"""Tests for the reward function, the runner and the metrics.

The reward is the easiest part of an RL system to get quietly wrong, and the
metrics are what the whole submission is judged on, so both are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hrapay.agents.static import StaticSchedulePolicy, StaticWithChannelSwitchPolicy
from hrapay.audit.logger import AuditLogger
from hrapay.env.generator import Episode
from hrapay.env.retry_env import RetryEnv, StepContext
from hrapay.env.spec import EnvSpec
from hrapay.eval.metrics import aggregate_seeds, compute
from hrapay.eval.runner import EpisodeRunner
from hrapay.guard.policy_guard import GuardConfig, PolicyGuard
from hrapay.rewards.reward import CalibratedReward, RewardConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def spec() -> EnvSpec:
    return EnvSpec.load(ROOT / "configs" / "spec_train.yaml")


@pytest.fixture(scope="module")
def reward() -> CalibratedReward:
    return CalibratedReward(RewardConfig.load(CONFIG))


def episode(amount: float = 1000.0, *, terminal: bool = False) -> Episode:
    return Episode(
        episode_id="txn_test",
        decline_code="insufficient_funds",
        origin_channel="UPI",
        amount=amount,
        merchant_tier="MEDIUM",
        start_day_of_month=15,
        is_terminal=terminal,
    )


def ctx(**kw: object) -> StepContext:
    base = dict(
        episode=episode(),
        macro="RETRY",
        timing_bucket="NOW",
        target_channel="UPI",
        elapsed_hours_total=0.0,
        attempts_total=1,
        attempts_on_target_channel=1,
        friction_penalty=0.0,
        success=False,
        terminated_by_abandon=False,
    )
    base.update(kw)
    return StepContext(**base)  # type: ignore[arg-type]


# --- reward ----------------------------------------------------------------


def test_success_is_worth_about_one_unit_of_transaction_value(reward: CalibratedReward) -> None:
    r = reward(ctx(success=True))
    assert 0.9 < r < 1.0


def test_failed_attempt_costs_money(reward: CalibratedReward) -> None:
    assert reward(ctx(success=False)) < 0


def test_fixed_fee_makes_small_transactions_relatively_more_expensive(
    reward: CalibratedReward,
) -> None:
    """This is why transaction amount stays a meaningful input feature.

    A flat Rs 3 gateway fee is a large fraction of a Rs 100 payment and a
    rounding error on a Rs 100,000 one, so the cost/benefit of retrying really
    does depend on size even though reward is scale-free.
    """
    small = reward(ctx(episode=episode(100.0), success=False))
    large = reward(ctx(episode=episode(100_000.0), success=False))
    assert small < large

    # The ratio is bounded below by the variable rate, which applies equally at
    # every size — so the gap is large but finite, not unbounded.
    assert reward.retry_cost_fraction(100.0) > 5 * reward.retry_cost_fraction(100_000.0)

    # It is specifically the FIXED fee that creates the asymmetry.
    fee = reward.config.retry_cost.fixed_fee_inr
    assert (fee / 100.0) == pytest.approx(1000 * (fee / 100_000.0))


def test_reward_is_scale_free_apart_from_the_fixed_fee(reward: CalibratedReward) -> None:
    a = reward(ctx(episode=episode(50_000.0), success=True))
    b = reward(ctx(episode=episode(60_000.0), success=True))
    assert a == pytest.approx(b, abs=1e-4)


def test_waiting_longer_costs_more(reward: CalibratedReward) -> None:
    soon = reward(ctx(success=True, elapsed_hours_total=2.0))
    late = reward(ctx(success=True, elapsed_hours_total=240.0))
    assert late < soon


def test_abandon_is_rewarded_only_when_correct(reward: CalibratedReward) -> None:
    good = reward(ctx(episode=episode(terminal=True), terminated_by_abandon=True))
    bad = reward(ctx(episode=episode(terminal=False), terminated_by_abandon=True))
    assert good > 0 > bad


def test_premature_abandon_hurts_more_than_one_wasted_retry(reward: CalibratedReward) -> None:
    """Otherwise the cheapest policy is to give up on everything immediately."""
    give_up = reward(ctx(episode=episode(terminal=False), terminated_by_abandon=True))
    one_retry = reward(ctx(success=False))
    assert give_up < one_retry


def test_explain_terms_sum_to_the_reward(reward: CalibratedReward) -> None:
    for c in (ctx(success=True), ctx(success=False), ctx(terminated_by_abandon=True)):
        terms = reward.explain(c)
        assert terms["total"] == pytest.approx(reward(c), abs=1e-9)


# --- runner ----------------------------------------------------------------


def build_runner(spec: EnvSpec, reward: CalibratedReward, audit: AuditLogger | None = None):
    env = RetryEnv(spec, seed=0, reward_fn=reward)
    guard = PolicyGuard(GuardConfig.load(CONFIG), timing_order=spec.time_buckets)
    return EpisodeRunner(env, guard, reward, audit=audit, run_id="test")


def test_runner_produces_one_result_per_episode(spec: EnvSpec, reward: CalibratedReward) -> None:
    runner = build_runner(spec, reward)
    results = runner.run_batch(StaticSchedulePolicy(spec), n_episodes=25, seed=0)
    assert len(results) == 25
    assert all(r.attempts >= 0 for r in results)


def test_guard_holds_across_a_full_batch(spec: EnvSpec, reward: CalibratedReward) -> None:
    """End-to-end version of the safety claim: zero fraud attempts, ever."""
    runner = build_runner(spec, reward)
    results = runner.run_batch(StaticWithChannelSwitchPolicy(spec), n_episodes=400, seed=0)
    assert sum(r.high_risk_attempts for r in results) == 0


def test_attempt_ceiling_holds_across_a_full_batch(spec: EnvSpec, reward: CalibratedReward) -> None:
    cap = GuardConfig.load(CONFIG).max_total_attempts
    runner = build_runner(spec, reward)
    results = runner.run_batch(StaticWithChannelSwitchPolicy(spec), n_episodes=300, seed=1)
    assert max(r.attempts for r in results) <= cap


def test_audit_records_every_decision(spec: EnvSpec, reward: CalibratedReward) -> None:
    audit = AuditLogger(None)
    runner = build_runner(spec, reward, audit)
    results = runner.run_batch(StaticSchedulePolicy(spec), n_episodes=30, seed=0)
    assert len(audit.records) == sum(len(r.steps) for r in results)
    assert all(r.policy == "static_schedule" for r in audit.records)


def test_audit_captures_blocked_actions_not_just_taken_ones(
    spec: EnvSpec, reward: CalibratedReward
) -> None:
    """The point of the audit trail: what the policy WANTED, and what stopped it."""
    audit = AuditLogger(None)
    runner = build_runner(spec, reward, audit)
    runner.run_batch(StaticSchedulePolicy(spec), n_episodes=300, seed=0)

    interventions = audit.interventions()
    assert interventions, "expected the guard to fire at least once in 300 episodes"
    for rec in interventions:
        assert rec.guard["proposed"] != rec.guard["final"]
        assert rec.guard["reason"]


def test_runs_are_reproducible(spec: EnvSpec, reward: CalibratedReward) -> None:
    def go() -> list[float]:
        runner = build_runner(spec, reward)
        return [
            r.total_reward
            for r in runner.run_batch(StaticWithChannelSwitchPolicy(spec), n_episodes=50, seed=3)
        ]

    assert go() == go()


# --- metrics ---------------------------------------------------------------


def test_recovery_rate_excludes_unrecoverable_transactions(
    spec: EnvSpec, reward: CalibratedReward
) -> None:
    """A policy must not look better simply because its population was easier."""
    runner = build_runner(spec, reward)
    results = runner.run_batch(StaticWithChannelSwitchPolicy(spec), n_episodes=400, seed=0)
    m = compute(results, "static_with_switch")
    assert m.n_recoverable < m.n_episodes
    assert 0.0 <= m.recovery_rate <= 1.0
    assert m.recovered_inr <= m.recoverable_inr


def test_channel_switching_baseline_beats_the_naive_one(
    spec: EnvSpec, reward: CalibratedReward
) -> None:
    """Sanity check on the environment, and the reason we report against BOTH.

    Comparing the learned agent only against the weaker baseline would inflate
    its apparent advantage.
    """
    naive = compute(
        build_runner(spec, reward).run_batch(StaticSchedulePolicy(spec), n_episodes=500, seed=0),
        "static_schedule",
    )
    better = compute(
        build_runner(spec, reward).run_batch(
            StaticWithChannelSwitchPolicy(spec), n_episodes=500, seed=0
        ),
        "static_with_switch",
    )
    assert better.recovery_rate > naive.recovery_rate


def test_aggregate_reports_spread_not_just_the_mean(
    spec: EnvSpec, reward: CalibratedReward
) -> None:
    per_seed = [
        compute(
            build_runner(spec, reward).run_batch(
                StaticSchedulePolicy(spec), n_episodes=100, seed=s
            ),
            "static_schedule",
        )
        for s in range(3)
    ]
    agg = aggregate_seeds(per_seed)
    assert agg["seeds"] == 3
    assert "recovery_rate_mean" in agg
    assert "recovery_rate_std" in agg


def test_metrics_reject_an_empty_run() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute([], "nobody")


# --- comparison logic ------------------------------------------------------


def test_comparison_knows_which_direction_is_good() -> None:
    """Not every metric improves by going up.

    The first version of compare_families assumed higher was always better and
    duly reported the agent with MORE wasted retries as the winner.
    """
    import pandas as pd

    from hrapay.eval.cli import HIGHER_IS_BETTER, compare_families

    summary = pd.DataFrame(
        {
            "recovery_rate_mean": [0.70, 0.70],
            "recovery_rate_std": [0.001, 0.001],
            "recovered_inr_mean": [1000.0, 1000.0],
            "recovered_inr_std": [1.0, 1.0],
            "wasted_attempts_mean": [1500.0, 900.0],  # b is clearly better here
            "wasted_attempts_std": [10.0, 10.0],
            "mean_time_to_recovery_h_mean": [50.0, 50.0],
            "mean_time_to_recovery_h_std": [0.1, 0.1],
        },
        index=["a", "b"],
    )
    lines = "\n".join(compare_families(summary, "a", "b"))
    wasted_line = next(ln for ln in lines.splitlines() if "wasted_attempts" in ln)
    assert "b better" in wasted_line

    assert HIGHER_IS_BETTER["recovery_rate"] is True
    assert HIGHER_IS_BETTER["wasted_attempts"] is False
    assert HIGHER_IS_BETTER["issuer_risk_exposure"] is False


def test_comparison_refuses_to_call_a_difference_inside_the_noise() -> None:
    """The check that would have stopped the earlier, wrong 'BDQ wins' claim."""
    import pandas as pd

    from hrapay.eval.cli import compare_families

    summary = pd.DataFrame(
        {
            "recovery_rate_mean": [0.688, 0.681],  # the real gap we saw
            "recovery_rate_std": [0.022, 0.026],  # dwarfed by the seed spread
            "recovered_inr_mean": [1.0, 1.0],
            "recovered_inr_std": [0.1, 0.1],
            "wasted_attempts_mean": [1.0, 1.0],
            "wasted_attempts_std": [0.1, 0.1],
            "mean_time_to_recovery_h_mean": [1.0, 1.0],
            "mean_time_to_recovery_h_std": [0.1, 0.1],
        },
        index=["bdq", "flat_dqn"],
    )
    line = next(ln for ln in compare_families(summary, "bdq", "flat_dqn") if "recovery_rate" in ln)
    assert "INDISTINGUISHABLE" in line


def test_single_seed_is_reported_as_unproven() -> None:
    import pandas as pd

    from hrapay.eval.cli import compare_families

    summary = pd.DataFrame(
        {
            "recovery_rate_mean": [0.90, 0.50],  # a huge gap...
            "recovery_rate_std": [0.0, 0.0],  # ...from one seed each
            "recovered_inr_mean": [1.0, 1.0],
            "recovered_inr_std": [0.0, 0.0],
            "wasted_attempts_mean": [1.0, 1.0],
            "wasted_attempts_std": [0.0, 0.0],
            "mean_time_to_recovery_h_mean": [1.0, 1.0],
            "mean_time_to_recovery_h_std": [0.0, 0.0],
        },
        index=["a", "b"],
    )
    assert all("unproven" in ln for ln in compare_families(summary, "a", "b"))
