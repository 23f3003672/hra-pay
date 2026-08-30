"""Evaluate every policy and print the comparison table.

    python -m hrapay.eval.cli --episodes 1000 --eval-seeds 3

Two sources of variance, and conflating them is how a submission ends up
claiming a result it does not have:

    environment variance   which transactions the policy happens to face.
                           Controlled by evaluating each checkpoint over several
                           evaluation seeds.

    training variance      which policy the same code happens to produce.
                           For DQN this is large. Controlled only by TRAINING
                           several checkpoints on different seeds.

An earlier version of this project reported a 0.9% gap between the flat and
branched agents from one training run each. Re-running on a second machine
flipped the sign of that gap. The lesson is baked into this file: learned
policies are aggregated across training seeds and reported as mean +/- std, and
`compare_families` refuses to call a difference real when it is smaller than the
spread across seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd

from hrapay.agents.base import Policy
from hrapay.agents.bdq import load_bdq_policy
from hrapay.agents.flat_dqn import load_flat_policy
from hrapay.agents.static import StaticSchedulePolicy, StaticWithChannelSwitchPolicy
from hrapay.audit.logger import AuditLogger
from hrapay.env.demo import load_priors
from hrapay.env.retry_env import RetryEnv
from hrapay.env.spec import EnvSpec
from hrapay.eval.metrics import Metrics, compute
from hrapay.eval.runner import EpisodeRunner
from hrapay.guard.policy_guard import GuardConfig, PolicyGuard
from hrapay.rewards.friction_table import CalibratedFrictionTable
from hrapay.rewards.reward import CalibratedReward, RewardConfig

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = ROOT / "configs" / "spec_train.yaml"
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"

HEADLINE = [
    "recovered_inr",
    "recovery_rate",
    "wasted_attempts",
    "issuer_risk_exposure",
    "mean_time_to_recovery_h",
    "correct_abandon_rate",
]

# Which direction is good. Not every metric improves by going up, and a
# comparison that assumes otherwise will confidently report the loser as the
# winner -- which is exactly what the first version of compare_families did for
# wasted_attempts.
HIGHER_IS_BETTER: dict[str, bool] = {
    "recovered_inr": True,
    "recovery_rate": True,
    "correct_abandon_rate": True,
    "wasted_attempts": False,
    "issuer_risk_exposure": False,
    "mean_time_to_recovery_h": False,
}


def discover_policies(spec: EnvSpec) -> list[tuple[str, int | None, Policy]]:
    """(family, train_seed, policy). Static policies have no training seed."""
    found: list[tuple[str, int | None, Policy]] = [
        ("static_schedule", None, StaticSchedulePolicy(spec)),
        ("static_with_switch", None, StaticWithChannelSwitchPolicy(spec)),
    ]
    loaders = {"flat": load_flat_policy, "bdq": load_bdq_policy}
    families = {"flat": "flat_dqn", "bdq": "bdq"}

    for prefix, loader in loaders.items():
        for ckpt in sorted(CHECKPOINTS.glob(f"{prefix}_seed*.pt")):
            seed = int(ckpt.stem.split("seed")[-1])
            found.append((families[prefix], seed, loader(spec, ckpt)))
    return found


def evaluate_one(
    policy: Policy,
    *,
    spec: EnvSpec,
    reward_cfg: RewardConfig,
    guard_cfg: GuardConfig,
    friction: CalibratedFrictionTable,
    priors: dict,
    episodes: int,
    eval_seed: int,
    audit_path: Path | None = None,
) -> Metrics:
    reward = CalibratedReward(reward_cfg, friction_table=friction)
    env = RetryEnv(spec, seed=eval_seed, channel_priors=priors, reward_fn=reward)
    guard = PolicyGuard(guard_cfg, timing_order=spec.time_buckets)

    audit = AuditLogger(audit_path, keep_in_memory=False) if audit_path else None
    runner = EpisodeRunner(env, guard, reward, audit=audit, run_id=f"{policy.name}_e{eval_seed}")
    results = runner.run_batch(policy, n_episodes=episodes, seed=eval_seed * 100_000)
    if audit:
        audit.close()
    return compute(results, policy.name)


def compare_families(summary: pd.DataFrame, a: str, b: str) -> list[str]:
    """State plainly whether a difference between two families is real.

    A gap smaller than the combined spread across training seeds is reported as
    indistinguishable. This is deliberately conservative: it is the check that
    would have stopped the earlier, wrong 'BDQ wins' claim.
    """
    lines: list[str] = []
    if a not in summary.index or b not in summary.index:
        return ["(not enough trained checkpoints to compare architectures)"]

    for metric in ("recovery_rate", "recovered_inr", "wasted_attempts", "mean_time_to_recovery_h"):
        ma, mb = summary.loc[a, f"{metric}_mean"], summary.loc[b, f"{metric}_mean"]
        sa, sb = summary.loc[a, f"{metric}_std"], summary.loc[b, f"{metric}_std"]
        gap = ma - mb
        noise = sa + sb

        if noise == 0:
            verdict = "single seed - no variance estimate, treat as unproven"
        elif abs(gap) > noise:
            a_ahead = gap > 0 if HIGHER_IS_BETTER[metric] else gap < 0
            verdict = f"{a} better" if a_ahead else f"{b} better"
        else:
            verdict = "INDISTINGUISHABLE (gap is inside the seed spread)"

        lines.append(
            f"  {metric:<26} {a}={ma:,.4f}+/-{sa:,.4f}  {b}={mb:,.4f}+/-{sb:,.4f}  -> {verdict}"
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retry policies.")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--tag", type=str, default="train", help="suffix for the results files")
    args = ap.parse_args()

    spec = EnvSpec.load(args.spec)
    reward_cfg = RewardConfig.load(args.config)
    guard_cfg = GuardConfig.load(args.config)
    friction = CalibratedFrictionTable.load()
    priors = load_priors(spec)
    RESULTS.mkdir(parents=True, exist_ok=True)

    policies = discover_policies(spec)
    learned = [f for f, s, _ in policies if s is not None]
    print(f"spec={spec.version}  episodes={args.episodes}  eval seeds={args.eval_seeds}")
    print(f"checkpoints found: {len(learned)}  ({', '.join(sorted(set(learned))) or 'none'})\n")

    rows: list[dict] = []
    for family, train_seed, policy in policies:
        for eval_seed in range(args.eval_seeds):
            audit_path = (
                RESULTS / f"audit_{family}.jsonl"
                if eval_seed == 0 and train_seed in (None, 0)
                else None
            )
            m = evaluate_one(
                policy,
                spec=spec,
                reward_cfg=reward_cfg,
                guard_cfg=guard_cfg,
                friction=friction,
                priors=priors,
                episodes=args.episodes,
                eval_seed=eval_seed,
                audit_path=audit_path,
            )
            rows.append(
                {"family": family, "train_seed": train_seed, "eval_seed": eval_seed, **m.to_row()}
            )

    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / f"runs_{args.tag}.csv", index=False)

    # Average over evaluation seeds first, so each training seed contributes one
    # number. Otherwise evaluation repeats would masquerade as extra evidence
    # about the architecture.
    per_train_seed = runs.groupby(["family", "train_seed"], dropna=False)[HEADLINE].mean()

    summary_rows = []
    for family, group in per_train_seed.groupby(level=0):
        row: dict = {"family": family, "train_seeds": len(group)}
        for metric in HEADLINE:
            values = group[metric].tolist()
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("family")
    summary.to_csv(RESULTS / f"summary_{args.tag}.csv")

    print("mean over training seeds (+/- std across those seeds)\n")
    display = pd.DataFrame(
        {
            "seeds": summary["train_seeds"],
            "recovered_inr": summary.apply(
                lambda r: f"{r['recovered_inr_mean']:,.0f} +/- {r['recovered_inr_std']:,.0f}",
                axis=1,
            ),
            "recovery_rate": summary.apply(
                lambda r: f"{r['recovery_rate_mean']:.3f} +/- {r['recovery_rate_std']:.3f}", axis=1
            ),
            "wasted": summary.apply(
                lambda r: f"{r['wasted_attempts_mean']:,.0f} +/- {r['wasted_attempts_std']:,.0f}",
                axis=1,
            ),
            "issuer_risk": summary["issuer_risk_exposure_mean"].map(lambda v: f"{v:.1f}"),
            "ttr_h": summary["mean_time_to_recovery_h_mean"].map(lambda v: f"{v:.1f}"),
            "correct_abandon": summary["correct_abandon_rate_mean"].map(lambda v: f"{v:.3f}"),
        }
    )
    print(display.to_string())

    print("\nflat vs branched, judged against seed spread:")
    for line in compare_families(summary, "flat_dqn", "bdq"):
        print(line)

    print(f"\nwrote {RESULTS / f'runs_{args.tag}.csv'}")
    print(f"wrote {RESULTS / f'summary_{args.tag}.csv'}")
    print(
        json.dumps({"spec": spec.version, "episodes": args.episodes, "eval_seeds": args.eval_seeds})
    )


if __name__ == "__main__":
    main()
