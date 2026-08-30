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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hrapay.agents.flat_dqn import FlatQNetwork, enumerate_flat_actions
from hrapay.agents.replay import ReplayBuffer
from hrapay.env.demo import load_priors
from hrapay.env.retry_env import RetryEnv
from hrapay.env.spec import EnvSpec
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
    net: FlatQNetwork,
    spec: EnvSpec,
    cfg_path: Path,
    flat_actions: list,
    *,
    n_episodes: int = 300,
    seed: int = 999_000,
    device: str = "cpu",
) -> float:
    """Mean episode return under the GREEDY policy, on a fixed held-out seed block.

    Training return is measured with exploration still switched on, which makes
    it a poor guide to how the deployed (greedy) policy actually performs. This
    is the statistic model selection should use, and it is deliberately computed
    on seeds the training loop never visits.
    """
    was_training = net.training
    net.eval()
    env = build_env(spec, cfg_path, seed)
    total = 0.0

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a_idx = int(net(x).argmax(dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(np.array(flat_actions[a_idx]))
            total += reward
            done = terminated or truncated

    if was_training:
        net.train()
    return total / n_episodes


def train_flat(
    spec: EnvSpec, cfg: TrainConfig, cfg_path: Path, out: Path, device: str = "cpu"
) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = build_env(spec, cfg_path, cfg.seed)
    obs_dim = env.observation_space.shape[0]
    flat_actions = enumerate_flat_actions(spec)
    n_actions = len(flat_actions)

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
            a_idx = int(rng.integers(0, n_actions))
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
            greedy = greedy_return(online, spec, cfg_path, flat_actions, device=device)

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
    final_greedy = greedy_return(online, spec, cfg_path, flat_actions, device=device)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a learned retry policy.")
    ap.add_argument("--agent", choices=["flat", "bdq"], default="flat")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    spec = EnvSpec.load(args.spec)
    cfg = TrainConfig(steps=args.steps, seed=args.seed)
    out = args.out or CHECKPOINTS / f"{args.agent}_seed{args.seed}.pt"

    if args.agent == "flat":
        summary = train_flat(spec, cfg, args.config, out)
    else:
        raise SystemExit("The branched agent lands on Day 5. Use --agent flat for now.")

    log = out.with_suffix(".train.json")
    log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {log}")


if __name__ == "__main__":
    main()
