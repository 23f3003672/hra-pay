"""Training loop for the learned policies.

    python -m hrapay.train --agent flat --steps 60000

Design notes that matter for the comparison being fair:

* The agent trains against the environment WITHOUT the PolicyGuard in the loop.
  The guard is a deployment-time constraint, not a training signal — training
  behind it would teach the policy that fraud retries are impossible rather than
  undesirable, and the moment the guard changed, the policy would be wrong. It
  therefore has to learn the economics itself, and the guard remains a genuine
  independent check at evaluation time rather than a crutch.

* Epsilon-greedy exploration, Double-DQN targets, a target network, and a
  uniform replay buffer. Nothing exotic: every hyperparameter here is shared
  verbatim with the branched agent so the only thing that differs between the
  two runs is the action-space architecture.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hrapay.agents.bdq import BranchingQNetwork, branch_activity_mask
from hrapay.agents.flat_dqn import FlatQNetwork, enumerate_flat_actions
from hrapay.agents.replay import ReplayBuffer
from hrapay.env.demo import load_priors
from hrapay.env.retry_env import RetryEnv
from hrapay.env.spec import MACRO_ACTIONS, EnvSpec
from hrapay.rewards.friction_table import CalibratedFrictionTable
from hrapay.rewards.reward import CalibratedReward, RewardConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "configs" / "spec_train.yaml"
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
CHECKPOINTS = ROOT / "checkpoints"


@dataclass
class TrainConfig:
    """Shared by both agents. Changing anything here changes both."""

    steps: int = 60_000
    warmup: int = 2_000
    batch_size: int = 128
    gamma: float = 0.99
    lr: float = 3e-4
    hidden: int = 128
    buffer_size: int = 100_000
    target_sync_every: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 30_000
    grad_clip: float = 10.0
    eval_every: int = 5_000
    seed: int = 0


def sample_random_branched_action(rng: np.random.Generator, branch_sizes: list[int]) -> np.ndarray:
    """Uniform over each action dimension independently.

    Used by BOTH trainers, and that is the entire point. Sampling uniformly over
    the flat agent's 31 enumerated actions gives P(ABANDON) = 1/31 = 3.2%, while
    sampling uniformly over three branches gives P(ABANDON) = 1/3 = 33%. Left
    unaddressed, the two agents would explore wildly different distributions —
    the flat agent would barely ever try abandoning — and the comparison between
    them would be measuring exploration, not architecture.
    """
    return np.array([int(rng.integers(0, n)) for n in branch_sizes])


def canonicalise_for_flat(action: np.ndarray) -> tuple[int, int, int]:
    """Map a branched action onto the flat agent's enumeration.

    Timing is ignored when abandoning and channel is ignored unless switching,
    so several branched actions collapse onto one flat action — exactly the
    collapsing the environment already performs.
    """
    macro, timing, channel = (int(v) for v in action)
    if macro == MACRO_ACTIONS.index("ABANDON"):
        return (macro, 0, 0)
    if macro == MACRO_ACTIONS.index("RETRY"):
        return (macro, timing, 0)
    return (macro, timing, channel)


def epsilon_at(step: int, cfg: TrainConfig) -> float:
    frac = min(1.0, step / max(cfg.eps_decay_steps, 1))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def build_env(spec: EnvSpec, cfg_path: Path, seed: int) -> RetryEnv:
    reward_cfg = RewardConfig.load(cfg_path)
    friction = CalibratedFrictionTable.load()
    reward = CalibratedReward(reward_cfg, friction_table=friction)
    return RetryEnv(
        spec,
        seed=seed,
        channel_priors=load_priors(spec),
        friction_table=friction,
        reward_fn=reward,
    )


@torch.no_grad()
def greedy_return(
    select: Callable[[np.ndarray], np.ndarray],
    spec: EnvSpec,
    cfg_path: Path,
    *,
    n_episodes: int = 300,
    seed: int = 999_000,
) -> float:
    """Mean episode return under the GREEDY policy, on a fixed held-out seed block.

    Training return is measured with exploration still switched on, which makes
    it a poor guide to how the deployed (greedy) policy actually performs. This
    is the statistic model selection should use, and it is deliberately computed
    on seeds the training loop never visits.

    Takes a selection function rather than a network so the flat and branched
    agents are scored by identical code on identical episodes.
    """
    env = build_env(spec, cfg_path, seed)
    total = 0.0

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(select(obs))
            total += reward
            done = terminated or truncated

    return total / n_episodes


def _flat_selector(net: FlatQNetwork, flat_actions: list, device: str) -> Callable:
    def select(obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return np.array(flat_actions[int(net(x).argmax(dim=-1).item())])

    return select


def _bdq_selector(net: BranchingQNetwork, device: str) -> Callable:
    def select(obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return np.array([int(q.argmax(dim=-1).item()) for q in net(x)])

    return select


def train_flat(
    spec: EnvSpec, cfg: TrainConfig, cfg_path: Path, out: Path, device: str = "cpu"
) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = build_env(spec, cfg_path, cfg.seed)
    obs_dim = env.observation_space.shape[0]
    flat_actions = enumerate_flat_actions(spec)
    n_actions = len(flat_actions)
    branch_sizes = [len(MACRO_ACTIONS), len(spec.time_buckets), len(spec.channels)]

    online = FlatQNetwork(obs_dim, n_actions, cfg.hidden).to(device)
    target = FlatQNetwork(obs_dim, n_actions, cfg.hidden).to(device)
    target.load_state_dict(online.state_dict())
    optimiser = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, seed=cfg.seed)

    print(f"flat DQN: obs_dim={obs_dim}  n_actions={n_actions}  device={device}")

    action_index = {a: i for i, a in enumerate(flat_actions)}

    obs, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    returns: list[float] = []
    history: list[dict] = []
    best_greedy = float("-inf")
    best_step = 0
    best_state: dict | None = None
    started = time.time()

    for step in range(1, cfg.steps + 1):
        eps = epsilon_at(step, cfg)
        if len(buffer) < cfg.warmup or rng.random() < eps:
            # Same exploration distribution as the branched agent -- see
            # sample_random_branched_action for why this matters.
            a_idx = action_index[
                canonicalise_for_flat(sample_random_branched_action(rng, branch_sizes))
            ]
        else:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a_idx = int(online(x).argmax(dim=-1).item())

        action = np.array(flat_actions[a_idx])
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, terminated)
        episode_return += reward
        obs = next_obs

        if done:
            returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

        if len(buffer) >= cfg.warmup:
            batch = buffer.sample(cfg.batch_size)
            b_obs = torch.as_tensor(batch.obs, device=device)
            b_next = torch.as_tensor(batch.next_obs, device=device)
            b_rew = torch.as_tensor(batch.reward, device=device)
            b_done = torch.as_tensor(batch.done, device=device)

            b_act = torch.as_tensor(
                np.array([action_index[tuple(a)] for a in batch.action]), device=device
            )

            q = online(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # Double DQN: online net picks the action, target net scores it.
                next_a = online(b_next).argmax(dim=-1, keepdim=True)
                next_q = target(b_next).gather(1, next_a).squeeze(1)
                td_target = b_rew + cfg.gamma * (1.0 - b_done) * next_q

            loss = nn.functional.smooth_l1_loss(q, td_target)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
            optimiser.step()

        if step % cfg.target_sync_every == 0:
            target.load_state_dict(online.state_dict())

        if step % cfg.eval_every == 0:
            recent = returns[-200:] or [0.0]
            greedy = greedy_return(_flat_selector(online, flat_actions, device), spec, cfg_path)

            improved = greedy > best_greedy
            if improved:
                best_greedy, best_step = greedy, step
                best_state = {k: v.detach().clone() for k, v in online.state_dict().items()}

            row = {
                "step": step,
                "epsilon": round(eps, 4),
                "episodes": len(returns),
                "train_return_200": round(float(np.mean(recent)), 4),
                "greedy_return": round(greedy, 4),
                "is_best": improved,
            }
            history.append(row)
            print(
                f"  step {step:>6}  eps {eps:.3f}  train {row['train_return_200']:>7.4f}  "
                f"greedy {greedy:>7.4f}{'  <- best' if improved else ''}"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    final_greedy = greedy_return(_flat_selector(online, flat_actions, device), spec, cfg_path)
    if final_greedy > best_greedy:
        best_greedy, best_step = final_greedy, cfg.steps
        best_state = {k: v.detach().clone() for k, v in online.state_dict().items()}

    assert best_state is not None
    torch.save(
        {
            "state_dict": best_state,
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "hidden": cfg.hidden,
            "spec_version": spec.version,
            "selected_at_step": best_step,
            "greedy_return": round(best_greedy, 4),
            "train_config": asdict(cfg),
        },
        out,
    )
    elapsed = time.time() - started
    print(
        f"\nbest greedy return {best_greedy:.4f} at step {best_step:,} "
        f"(final step scored {final_greedy:.4f})"
    )
    print(f"saved {out}  ({elapsed:.0f}s, {len(returns)} episodes)")
    return {
        "history": history,
        "episodes": len(returns),
        "seconds": round(elapsed, 1),
        "best_greedy_return": round(best_greedy, 4),
        "best_step": best_step,
        "final_greedy_return": round(final_greedy, 4),
    }


def train_bdq(
    spec: EnvSpec, cfg: TrainConfig, cfg_path: Path, out: Path, device: str = "cpu"
) -> dict:
    """Train the Branching Dueling Q-Network.

    Deliberately a near-copy of train_flat. Every hyperparameter comes from the
    same TrainConfig, the replay buffer and epsilon schedule are identical, and
    the greedy evaluation runs on the same held-out seed block. If this function
    quietly did something smarter than train_flat, the comparison between them
    would measure my tuning effort rather than the architecture.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = build_env(spec, cfg_path, cfg.seed)
    obs_dim = env.observation_space.shape[0]
    branch_sizes = [len(MACRO_ACTIONS), len(spec.time_buckets), len(spec.channels)]

    online = BranchingQNetwork(obs_dim, branch_sizes, cfg.hidden).to(device)
    target = BranchingQNetwork(obs_dim, branch_sizes, cfg.hidden).to(device)
    target.load_state_dict(online.state_dict())
    optimiser = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, seed=cfg.seed)

    print(
        f"BDQ: obs_dim={obs_dim}  branches={branch_sizes}  "
        f"outputs={sum(branch_sizes)} (flat head would need "
        f"{len(enumerate_flat_actions(spec))})  device={device}"
    )

    obs, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    returns: list[float] = []
    history: list[dict] = []
    best_greedy = float("-inf")
    best_step = 0
    best_state: dict | None = None
    started = time.time()

    for step in range(1, cfg.steps + 1):
        eps = epsilon_at(step, cfg)
        if len(buffer) < cfg.warmup or rng.random() < eps:
            action = sample_random_branched_action(rng, branch_sizes)
        else:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = np.array([int(q.argmax(dim=-1).item()) for q in online(x)])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, terminated)
        episode_return += reward
        obs = next_obs

        if done:
            returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

        if len(buffer) >= cfg.warmup:
            batch = buffer.sample(cfg.batch_size)
            b_obs = torch.as_tensor(batch.obs, device=device)
            b_next = torch.as_tensor(batch.next_obs, device=device)
            b_rew = torch.as_tensor(batch.reward, device=device)
            b_done = torch.as_tensor(batch.done, device=device)
            b_act = torch.as_tensor(batch.action, device=device)
            b_mask = torch.as_tensor(branch_activity_mask(batch.action[:, 0]), device=device)

            online_q = online(b_obs)
            chosen = torch.stack(
                [online_q[d].gather(1, b_act[:, d : d + 1]).squeeze(1) for d in range(3)],
                dim=1,
            )

            with torch.no_grad():
                # Double DQN per branch: online net argmaxes, target net scores.
                next_online = online(b_next)
                next_target = target(b_next)
                next_per_branch = torch.stack(
                    [
                        next_target[d]
                        .gather(1, next_online[d].argmax(dim=-1, keepdim=True))
                        .squeeze(1)
                        for d in range(3)
                    ],
                    dim=1,
                )
                # One shared target, averaged across branches (Tavakoli et al.).
                # A single reward arrived for one joint action, so a single
                # target keeps the branches consistent rather than letting them
                # drift into three disagreeing value functions.
                next_value = next_per_branch.mean(dim=1)
                td_target = b_rew + cfg.gamma * (1.0 - b_done) * next_value

            per_branch_loss = nn.functional.smooth_l1_loss(
                chosen, td_target.unsqueeze(1).expand(-1, 3), reduction="none"
            )
            # Mask out branches that had no effect on this transition, so the
            # timing head is not trained on the reward from an ABANDON.
            loss = (per_branch_loss * b_mask).sum() / b_mask.sum().clamp(min=1.0)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
            optimiser.step()

        if step % cfg.target_sync_every == 0:
            target.load_state_dict(online.state_dict())

        if step % cfg.eval_every == 0:
            recent = returns[-200:] or [0.0]
            online.eval()
            greedy = greedy_return(_bdq_selector(online, device), spec, cfg_path)
            online.train()

            improved = greedy > best_greedy
            if improved:
                best_greedy, best_step = greedy, step
                best_state = {k: v.detach().clone() for k, v in online.state_dict().items()}

            history.append(
                {
                    "step": step,
                    "epsilon": round(eps, 4),
                    "episodes": len(returns),
                    "train_return_200": round(float(np.mean(recent)), 4),
                    "greedy_return": round(greedy, 4),
                    "is_best": improved,
                }
            )
            print(
                f"  step {step:>6}  eps {eps:.3f}  train {float(np.mean(recent)):>7.4f}  "
                f"greedy {greedy:>7.4f}{'  <- best' if improved else ''}"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    online.eval()
    final_greedy = greedy_return(_bdq_selector(online, device), spec, cfg_path)
    if final_greedy > best_greedy:
        best_greedy, best_step = final_greedy, cfg.steps
        best_state = {k: v.detach().clone() for k, v in online.state_dict().items()}

    assert best_state is not None
    torch.save(
        {
            "state_dict": best_state,
            "obs_dim": obs_dim,
            "branch_sizes": branch_sizes,
            "hidden": cfg.hidden,
            "spec_version": spec.version,
            "selected_at_step": best_step,
            "greedy_return": round(best_greedy, 4),
            "train_config": asdict(cfg),
        },
        out,
    )
    elapsed = time.time() - started
    print(
        f"\nbest greedy return {best_greedy:.4f} at step {best_step:,} "
        f"(final step scored {final_greedy:.4f})"
    )
    print(f"saved {out}  ({elapsed:.0f}s, {len(returns)} episodes)")
    return {
        "history": history,
        "episodes": len(returns),
        "seconds": round(elapsed, 1),
        "best_greedy_return": round(best_greedy, 4),
        "best_step": best_step,
        "final_greedy_return": round(final_greedy, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a learned retry policy.")
    ap.add_argument("--agent", choices=["flat", "bdq"], default="flat")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="train one checkpoint per seed. Three seeds is the minimum that says "
        "anything about training variance.",
    )
    args = ap.parse_args()

    spec = EnvSpec.load(args.spec)
    trainer = train_flat if args.agent == "flat" else train_bdq
    summaries: dict[str, dict] = {}

    for i, seed in enumerate(args.seeds, start=1):
        print(f"\n=== {args.agent}  seed {seed}  ({i}/{len(args.seeds)}) ===")
        cfg = TrainConfig(steps=args.steps, seed=seed)
        out = CHECKPOINTS / f"{args.agent}_seed{seed}.pt"
        summary = trainer(spec, cfg, args.config, out)
        out.with_suffix(".train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries[str(seed)] = summary

    if len(args.seeds) > 1:
        bests = [s["best_greedy_return"] for s in summaries.values()]
        spread = max(bests) - min(bests)
        print(f"\n=== {args.agent}: {len(args.seeds)} seeds ===")
        for seed, s in summaries.items():
            print(
                f"  seed {seed}: best greedy {s['best_greedy_return']:.4f} at step {s['best_step']:,}"
            )
        print(f"  mean {np.mean(bests):.4f}  std {np.std(bests):.4f}  spread {spread:.4f}")
        print(
            "  Any claimed difference between architectures smaller than this spread "
            "is not a result."
        )


if __name__ == "__main__":
    main()
