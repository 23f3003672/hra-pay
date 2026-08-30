"""Evaluate one or more policies and print the comparison table.

    python -m hrapay.eval.cli --episodes 2000 --seeds 5

Writes per-seed metrics and the aggregate to results/, and the full audit trail
to results/audit_<policy>.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hrapay.agents.base import Policy
from hrapay.agents.flat_dqn import load_flat_policy
from hrapay.agents.static import StaticSchedulePolicy, StaticWithChannelSwitchPolicy
from hrapay.audit.logger import AuditLogger
from hrapay.env.demo import load_priors
from hrapay.env.retry_env import RetryEnv
from hrapay.env.spec import EnvSpec
from hrapay.eval.metrics import Metrics, aggregate_seeds, compute
from hrapay.eval.runner import EpisodeRunner
from hrapay.guard.policy_guard import GuardConfig, PolicyGuard
from hrapay.rewards.friction_table import CalibratedFrictionTable
from hrapay.rewards.reward import CalibratedReward, RewardConfig

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = ROOT / "configs" / "spec_train.yaml"
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
RESULTS = ROOT / "results"


CHECKPOINTS = ROOT / "checkpoints"


def build_policies(spec: EnvSpec, *, checkpoint_seed: int = 0) -> dict[str, Policy]:
    policies: dict[str, Policy] = {
        "static_schedule": StaticSchedulePolicy(spec),
        "static_with_switch": StaticWithChannelSwitchPolicy(spec),
    }
    flat_ckpt = CHECKPOINTS / f"flat_seed{checkpoint_seed}.pt"
    if flat_ckpt.exists():
        policies["flat_dqn"] = load_flat_policy(spec, flat_ckpt)
    return policies


def evaluate(
    policy: Policy,
    *,
    spec: EnvSpec,
    reward_cfg: RewardConfig,
    guard_cfg: GuardConfig,
    priors: dict,
    episodes: int,
    seed: int,
    audit_path: Path | None,
) -> Metrics:
    reward = CalibratedReward(reward_cfg, friction_table=CalibratedFrictionTable.load())
    env = RetryEnv(spec, seed=seed, channel_priors=priors, reward_fn=reward)
    guard = PolicyGuard(guard_cfg, timing_order=spec.time_buckets)

    audit = AuditLogger(audit_path, keep_in_memory=False) if audit_path else None
    runner = EpisodeRunner(env, guard, reward, audit=audit, run_id=f"{policy.name}_s{seed}")
    results = runner.run_batch(policy, n_episodes=episodes, seed=seed * 100_000)
    if audit:
        audit.close()
    return compute(results, policy.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retry policies.")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--policy", type=str, default=None, help="run only this policy")
    args = ap.parse_args()

    spec = EnvSpec.load(args.spec)
    reward_cfg = RewardConfig.load(args.config)
    guard_cfg = GuardConfig.load(args.config)
    priors = load_priors(spec)
    RESULTS.mkdir(parents=True, exist_ok=True)

    policies = build_policies(spec)
    if args.policy:
        policies = {args.policy: policies[args.policy]}

    print(f"spec={spec.version}  episodes={args.episodes}  seeds={args.seeds}\n")

    per_seed_rows: list[dict] = []
    aggregate_rows: list[dict] = []

    for name, policy in policies.items():
        seed_metrics: list[Metrics] = []
        for seed in range(args.seeds):
            audit_path = RESULTS / f"audit_{name}.jsonl" if seed == 0 else None
            m = evaluate(
                policy,
                spec=spec,
                reward_cfg=reward_cfg,
                guard_cfg=guard_cfg,
                priors=priors,
                episodes=args.episodes,
                seed=seed,
                audit_path=audit_path,
            )
            seed_metrics.append(m)
            per_seed_rows.append({"seed": seed, **m.to_row()})
        aggregate_rows.append(aggregate_seeds(seed_metrics))

    per_seed = pd.DataFrame(per_seed_rows)
    aggregate = pd.DataFrame(aggregate_rows)
    per_seed.to_csv(RESULTS / "per_seed_metrics.csv", index=False)
    aggregate.to_csv(RESULTS / "aggregate_metrics.csv", index=False)

    headline = [
        "policy",
        "recovered_inr",
        "recovery_rate",
        "wasted_attempts",
        "issuer_risk_exposure",
        "mean_time_to_recovery_h",
        "correct_abandon_rate",
    ]
    summary = (
        per_seed.groupby("policy")[[c for c in headline if c != "policy"]].mean().reset_index()
    )
    print(summary.to_string(index=False))
    print(f"\nwrote {RESULTS / 'per_seed_metrics.csv'}")
    print(f"wrote {RESULTS / 'aggregate_metrics.csv'}")
    print(json.dumps({"spec": spec.version, "episodes": args.episodes, "seeds": args.seeds}))


if __name__ == "__main__":
    main()
