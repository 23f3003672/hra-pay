"""Tests for PolicyGuard.

The guard is the system's safety claim. If these tests do not hold, the claim
that "no Q-value can authorise a fraud retry" is not true, and the submission
should not make it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hrapay.env.spec import EnvSpec
from hrapay.guard.policy_guard import GuardConfig, GuardState, PolicyGuard, RuleClass

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spec() -> EnvSpec:
    return EnvSpec.load(ROOT / "configs" / "spec_train.yaml")


@pytest.fixture(scope="module")
def guard(spec: EnvSpec) -> PolicyGuard:
    cfg = GuardConfig.load(ROOT / "configs" / "default.yaml")
    return PolicyGuard(cfg, timing_order=spec.time_buckets)


def state(**kw: object) -> GuardState:
    base = dict(
        decline_code="insufficient_funds",
        attempts_total=0,
        attempts_by_channel={},
        current_channel="UPI",
    )
    base.update(kw)
    return GuardState(**base)  # type: ignore[arg-type]


# --- compliance ------------------------------------------------------------


def test_fraud_retry_is_always_blocked(guard: PolicyGuard) -> None:
    """The headline safety property. No inputs may produce a fraud retry."""
    for macro in ("RETRY", "SWITCH_CHANNEL"):
        for timing in ("NOW", "PLUS_72H"):
            for attempts in (0, 1, 3):
                v = guard.review(
                    (macro, timing, "UPI"),
                    state(decline_code="suspected_fraud", attempts_total=attempts),
                )
                assert v.final[0] == "ABANDON"
                assert v.intervened
                assert v.rule_class is RuleClass.COMPLIANCE


def test_compliance_block_outranks_every_other_rule(guard: PolicyGuard) -> None:
    """When several rules apply, the audit log must name the most serious."""
    v = guard.review(
        ("RETRY", "NOW", "UPI"),
        state(
            decline_code="suspected_fraud",
            attempts_total=99,
            attempts_by_channel={"UPI": 99},
        ),
    )
    assert v.rule == "compliance_blocked_code"


def test_futility_block_for_closed_accounts(guard: PolicyGuard) -> None:
    v = guard.review(("RETRY", "PLUS_24H", "UPI"), state(decline_code="account_closed"))
    assert v.final[0] == "ABANDON"
    assert v.rule_class is RuleClass.FUTILITY


# --- budgets ---------------------------------------------------------------


def test_total_attempt_ceiling(guard: PolicyGuard) -> None:
    cap = guard.config.max_total_attempts
    allowed = guard.review(("RETRY", "PLUS_24H", "UPI"), state(attempts_total=cap - 1))
    assert not allowed.intervened

    blocked = guard.review(("RETRY", "PLUS_24H", "UPI"), state(attempts_total=cap))
    assert blocked.final[0] == "ABANDON"
    assert blocked.rule == "max_total_attempts"


def test_per_channel_cap(guard: PolicyGuard) -> None:
    cap = guard.config.max_attempts_per_channel
    v = guard.review(
        ("RETRY", "PLUS_24H", "UPI"),
        state(attempts_total=cap, attempts_by_channel={"UPI": cap}, current_channel="UPI"),
    )
    assert v.final[0] == "ABANDON"
    assert v.rule == "max_attempts_per_channel"


# --- velocity --------------------------------------------------------------


def test_velocity_escalates_rather_than_blocking(guard: PolicyGuard) -> None:
    """Velocity is a delay, not a veto — it must not destroy a good retry."""
    v = guard.review(
        ("RETRY", "NOW", "UPI"),
        state(attempts_total=guard.config.velocity_attempt_threshold),
    )
    assert v.intervened
    assert v.final[0] == "RETRY"
    assert v.final[1] == guard.config.velocity_min_timing
    assert v.rule_class is RuleClass.VELOCITY


def test_velocity_leaves_already_patient_timing_alone(guard: PolicyGuard) -> None:
    v = guard.review(
        ("RETRY", "PLUS_72H", "UPI"),
        state(attempts_total=guard.config.velocity_attempt_threshold),
    )
    assert not v.intervened


def test_velocity_inactive_early_in_the_episode(guard: PolicyGuard) -> None:
    v = guard.review(("RETRY", "NOW", "UPI"), state(attempts_total=0))
    assert not v.intervened


# --- invariants ------------------------------------------------------------


def test_abandon_is_never_overridden(guard: PolicyGuard) -> None:
    """The guard exists to stop spending, never to force it."""
    for code in ("suspected_fraud", "account_closed", "insufficient_funds"):
        v = guard.review(("ABANDON", "NOW", None), state(decline_code=code, attempts_total=0))
        assert v.final[0] == "ABANDON"
        assert not v.intervened


def test_every_intervention_carries_a_reason(guard: PolicyGuard) -> None:
    """An unexplained override is not an audit trail."""
    cases = [
        (("RETRY", "NOW", "UPI"), state(decline_code="suspected_fraud")),
        (("RETRY", "NOW", "UPI"), state(decline_code="account_closed")),
        (("RETRY", "NOW", "UPI"), state(attempts_total=99)),
        (("RETRY", "NOW", "UPI"), state(attempts_total=2, attempts_by_channel={"UPI": 9})),
        (("RETRY", "NOW", "UPI"), state(attempts_total=2)),
    ]
    for proposed, st in cases:
        v = guard.review(proposed, st)
        assert v.intervened
        assert v.rule and v.reason and v.rule_class
        assert len(v.reason) > 20


def test_verdict_records_what_the_policy_wanted(guard: PolicyGuard) -> None:
    proposed = ("RETRY", "NOW", "UPI")
    v = guard.review(proposed, state(decline_code="suspected_fraud"))
    assert v.proposed == proposed
    assert v.final != proposed
    assert v.to_dict()["proposed"]["macro"] == "RETRY"
    assert v.to_dict()["final"]["macro"] == "ABANDON"
