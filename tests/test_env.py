"""Tests for the synthetic environment.

These are not decorative. The whole submission rests on the claim that the
environment is a fair test bed, so the properties that make it fair are asserted
here: seeded reproducibility, bounded probabilities, guaranteed termination, and
the specific issuer behaviours the spec claims to model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hrapay.env.generator import EpisodeGenerator
from hrapay.env.retry_env import RetryEnv
from hrapay.env.spec import EnvSpec

SPEC_PATH = Path(__file__).resolve().parents[1] / "configs" / "spec_train.yaml"


@pytest.fixture(scope="module")
def spec() -> EnvSpec:
    return EnvSpec.load(SPEC_PATH)


# --- spec integrity --------------------------------------------------------


def test_spec_loads_and_validates(spec: EnvSpec) -> None:
    assert spec.version.startswith("spec_train/")
    assert len(spec.decline_codes) >= 5
    assert len(spec.channels) == 5


def test_success_probability_is_bounded(spec: EnvSpec) -> None:
    """No combination of multipliers may push probability outside [0, cap]."""
    for code in spec.decline_codes:
        for channel in spec.channels:
            for bucket in spec.time_buckets:
                for day in (1, 15, 30):
                    for attempts in range(6):
                        p = spec.success_probability(
                            decline_code=code,
                            target_channel=channel,
                            origin_channel="UPI",
                            time_bucket=bucket,
                            day_of_month=day,
                            merchant_tier="LOW",
                            attempts_on_target_channel=attempts,
                        )
                        assert 0.0 <= p <= spec.success_probability_cap


def test_fatigue_is_monotonically_decreasing(spec: EnvSpec) -> None:
    values = [spec.fatigue(i) for i in range(8)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


# --- the issuer behaviours the spec claims to model ------------------------


def test_insufficient_funds_recovers_better_after_a_delay(spec: EnvSpec) -> None:
    kw = dict(
        decline_code="insufficient_funds",
        target_channel="UPI",
        origin_channel="UPI",
        day_of_month=15,
        merchant_tier="MEDIUM",
        attempts_on_target_channel=0,
    )
    now = spec.success_probability(time_bucket="NOW", **kw)
    later = spec.success_probability(time_bucket="PLUS_24H", **kw)
    assert later > now * 2


def test_payday_lifts_only_payday_sensitive_codes(spec: EnvSpec) -> None:
    kw = dict(
        target_channel="UPI",
        origin_channel="UPI",
        time_bucket="PLUS_24H",
        merchant_tier="MEDIUM",
        attempts_on_target_channel=0,
    )
    sensitive_off = spec.success_probability(
        decline_code="insufficient_funds", day_of_month=15, **kw
    )
    sensitive_on = spec.success_probability(decline_code="insufficient_funds", day_of_month=1, **kw)
    assert sensitive_on > sensitive_off

    other_off = spec.success_probability(decline_code="do_not_honor", day_of_month=15, **kw)
    other_on = spec.success_probability(decline_code="do_not_honor", day_of_month=1, **kw)
    assert other_on == pytest.approx(other_off)


def test_expired_card_is_hopeless_on_same_instrument_but_not_on_upi(spec: EnvSpec) -> None:
    kw = dict(
        decline_code="expired_card",
        origin_channel="CREDIT_CARD",
        time_bucket="PLUS_24H",
        day_of_month=15,
        merchant_tier="MEDIUM",
        attempts_on_target_channel=0,
    )
    same = spec.success_probability(target_channel="CREDIT_CARD", **kw)
    switched = spec.success_probability(target_channel="UPI", **kw)
    assert same < 0.02
    assert switched > 0.30


def test_terminal_codes_are_near_hopeless_everywhere(spec: EnvSpec) -> None:
    for code in ("suspected_fraud", "account_closed"):
        for channel in spec.channels:
            p = spec.success_probability(
                decline_code=code,
                target_channel=channel,
                origin_channel="UPI",
                time_bucket="PLUS_72H",
                day_of_month=1,
                merchant_tier="LOW",
                attempts_on_target_channel=0,
            )
            assert p < 0.05


# --- generator -------------------------------------------------------------


def test_generator_is_reproducible(spec: EnvSpec) -> None:
    a = EpisodeGenerator(spec, seed=7).sample_batch(50)
    b = EpisodeGenerator(spec, seed=7).sample_batch(50)
    assert a == b


def test_different_seeds_give_different_populations(spec: EnvSpec) -> None:
    a = EpisodeGenerator(spec, seed=1).sample_batch(50)
    b = EpisodeGenerator(spec, seed=2).sample_batch(50)
    assert a != b


def test_generated_amounts_respect_bounds(spec: EnvSpec) -> None:
    for ep in EpisodeGenerator(spec, seed=3).sample_batch(500):
        assert spec.amount.min <= ep.amount <= spec.amount.max
        assert 1 <= ep.start_day_of_month <= 28


def test_account_closed_is_always_terminal(spec: EnvSpec) -> None:
    eps = EpisodeGenerator(spec, seed=11).sample_batch(3000)
    closed = [e for e in eps if e.decline_code == "account_closed"]
    assert closed, "expected some account_closed episodes in 3000 samples"
    assert all(e.is_terminal for e in closed)


def test_population_contains_both_recoverable_and_terminal(spec: EnvSpec) -> None:
    eps = EpisodeGenerator(spec, seed=5).sample_batch(1000)
    terminal_share = sum(e.is_terminal for e in eps) / len(eps)
    assert 0.05 < terminal_share < 0.45, f"terminal share {terminal_share:.2%} is degenerate"


# --- environment -----------------------------------------------------------


def test_observation_matches_declared_space(spec: EnvSpec) -> None:
    env = RetryEnv(spec, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert len(env.feature_names) == obs.shape[0]


def test_abandon_terminates_immediately(spec: EnvSpec) -> None:
    env = RetryEnv(spec, seed=0)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step([2, 0, 0])
    assert terminated and not truncated
    assert info["outcome"] == "ABANDONED"


def test_episode_always_terminates(spec: EnvSpec) -> None:
    """Never-abandon, always-retry-now is the worst case for termination."""
    env = RetryEnv(spec, seed=0)
    for i in range(200):
        env.reset(seed=100 + i)
        steps = 0
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step([0, 0, 0])
            done = terminated or truncated
            steps += 1
            assert steps <= spec.max_attempts_per_episode + 1
        assert steps <= spec.max_attempts_per_episode


def test_switch_to_current_channel_is_recorded_as_retry(spec: EnvSpec) -> None:
    env = RetryEnv(spec, seed=0)
    env.reset(seed=0)
    current_idx = spec.channels.index(env._current_channel)  # noqa: SLF001
    _, _, _, _, info = env.step([1, 0, current_idx])
    assert info["macro"] == "RETRY"


def test_terminal_episodes_never_succeed(spec: EnvSpec) -> None:
    spec_env = RetryEnv(spec, seed=0)
    gen = EpisodeGenerator(spec, seed=42)
    checked = 0
    for i in range(2000):
        ep = gen.sample(i)
        if not ep.is_terminal:
            continue
        spec_env.reset(seed=i, options={"episode": ep})
        done = False
        while not done:
            _, _, terminated, truncated, info = spec_env.step([0, 3, 0])
            assert not info["success"]
            done = terminated or truncated
        checked += 1
        if checked >= 40:
            break
    assert checked >= 40


def test_env_rollout_is_reproducible(spec: EnvSpec) -> None:
    def rollout() -> list[float]:
        env = RetryEnv(spec, seed=99)
        env.action_space.seed(99)
        rewards: list[float] = []
        for i in range(20):
            env.reset(seed=99 + i)
            done = False
            while not done:
                _, r, terminated, truncated, _ = env.step(env.action_space.sample())
                rewards.append(r)
                done = terminated or truncated
        return rewards

    assert rollout() == rollout()
