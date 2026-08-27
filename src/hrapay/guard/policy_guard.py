"""PolicyGuard — deterministic limits enforced outside the learned policy.

The learned agent proposes an action. This layer decides whether that action is
permitted, and rewrites or blocks it if not. Nothing here is learned. No Q-value,
however confident, can override it.

This exists because a reinforcement learning policy trained on synthetic data
should not be trusted with an unbounded action that moves money. Separating the
two means the safety properties of the system are *provable by reading 120 lines
of rules*, rather than being an emergent hope about what the network learned.

Every intervention is recorded with the rule that fired and a human-readable
reason, and flows straight into the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RuleClass(str, Enum):
    """Why a rule exists. The distinction is not cosmetic.

    COMPLIANCE rules are not a matter of expected value. Retrying an
    authorisation the issuer flagged as fraud is not something a better reward
    estimate should ever be allowed to justify, so it is not left to the policy.

    FUTILITY rules are ordinary economics — don't spend fees and issuer goodwill
    on attempts that cannot succeed. In principle the agent could learn these;
    they are enforced anyway so that a half-trained or drifted policy still
    cannot burn money.
    """

    COMPLIANCE = "COMPLIANCE"
    FUTILITY = "FUTILITY"
    VELOCITY = "VELOCITY"
    BUDGET = "BUDGET"


@dataclass(frozen=True)
class GuardVerdict:
    """The guard's decision about one proposed action."""

    proposed: tuple[str, str, str | None]
    final: tuple[str, str, str | None]
    intervened: bool
    rule: str | None = None
    rule_class: RuleClass | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed": {
                "macro": self.proposed[0],
                "timing": self.proposed[1],
                "channel": self.proposed[2],
            },
            "final": {
                "macro": self.final[0],
                "timing": self.final[1],
                "channel": self.final[2],
            },
            "intervened": self.intervened,
            "rule": self.rule,
            "rule_class": self.rule_class.value if self.rule_class else None,
            "reason": self.reason,
        }


class GuardConfig(BaseModel):
    compliance_blocked_codes: list[str]
    futility_blocked_codes: list[str]
    max_total_attempts: int = Field(gt=0)
    max_attempts_per_channel: int = Field(gt=0)
    velocity_attempt_threshold: int = Field(ge=0)
    velocity_min_timing: str

    @classmethod
    def load(cls, path: str | Path) -> GuardConfig:
        raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw["guard"])


@dataclass
class GuardState:
    """Everything the guard needs to know about the episode so far."""

    decline_code: str
    attempts_total: int
    attempts_by_channel: dict[str, int]
    current_channel: str


ABANDON = "ABANDON"


class PolicyGuard:
    """Applies the rules in configs/default.yaml to a proposed action.

    Rules are evaluated in order of severity. The first one that fires wins, so
    the reason recorded in the audit log is always the most serious applicable
    reason rather than an arbitrary one.
    """

    def __init__(self, config: GuardConfig, timing_order: list[str]) -> None:
        self.config = config
        self.timing_order = timing_order

    def _escalate_timing(self, timing: str) -> str:
        """Push a timing choice out to at least the velocity minimum."""
        floor = self.config.velocity_min_timing
        if self.timing_order.index(timing) >= self.timing_order.index(floor):
            return timing
        return floor

    def review(
        self,
        proposed: tuple[str, str, str | None],
        state: GuardState,
    ) -> GuardVerdict:
        macro, timing, channel = proposed
        cfg = self.config

        # Abandoning is always permitted. The guard exists to stop the agent
        # spending money, never to force it to keep spending.
        if macro == ABANDON:
            return GuardVerdict(proposed, proposed, intervened=False)

        # --- 1. Compliance: hard block, highest severity -------------------
        if state.decline_code in cfg.compliance_blocked_codes:
            return GuardVerdict(
                proposed,
                (ABANDON, timing, None),
                intervened=True,
                rule="compliance_blocked_code",
                rule_class=RuleClass.COMPLIANCE,
                reason=(
                    f"'{state.decline_code}' is on the compliance block list. Retrying an "
                    f"authorisation the issuer flagged for risk is never permitted, "
                    f"regardless of expected value."
                ),
            )

        # --- 2. Futility: known-hopeless -----------------------------------
        if state.decline_code in cfg.futility_blocked_codes:
            return GuardVerdict(
                proposed,
                (ABANDON, timing, None),
                intervened=True,
                rule="futility_blocked_code",
                rule_class=RuleClass.FUTILITY,
                reason=(
                    f"'{state.decline_code}' cannot succeed on any channel. Every further "
                    f"attempt is pure cost."
                ),
            )

        # --- 3. Total attempt budget ---------------------------------------
        if state.attempts_total >= cfg.max_total_attempts:
            return GuardVerdict(
                proposed,
                (ABANDON, timing, None),
                intervened=True,
                rule="max_total_attempts",
                rule_class=RuleClass.BUDGET,
                reason=(
                    f"{state.attempts_total} attempts already made; the ceiling is "
                    f"{cfg.max_total_attempts}."
                ),
            )

        # --- 4. Per-channel attempt cap ------------------------------------
        target = channel if channel is not None else state.current_channel
        if state.attempts_by_channel.get(target, 0) >= cfg.max_attempts_per_channel:
            alternatives = [
                ch
                for ch in state.attempts_by_channel
                if state.attempts_by_channel.get(ch, 0) < cfg.max_attempts_per_channel
            ]
            return GuardVerdict(
                proposed,
                (ABANDON, timing, None),
                intervened=True,
                rule="max_attempts_per_channel",
                rule_class=RuleClass.BUDGET,
                reason=(
                    f"{target} has already been attempted "
                    f"{state.attempts_by_channel.get(target, 0)} times "
                    f"(cap {cfg.max_attempts_per_channel}); untried channels: "
                    f"{sorted(set(alternatives)) or 'none recorded'}."
                ),
            )

        # --- 5. Velocity control -------------------------------------------
        if state.attempts_total >= cfg.velocity_attempt_threshold:
            escalated = self._escalate_timing(timing)
            if escalated != timing:
                return GuardVerdict(
                    proposed,
                    (macro, escalated, channel),
                    intervened=True,
                    rule="velocity_min_gap",
                    rule_class=RuleClass.VELOCITY,
                    reason=(
                        f"After {state.attempts_total} attempts, timing is floored at "
                        f"{escalated} (proposed {timing}). Rapid repeated authorisations "
                        f"trigger issuer velocity throttling."
                    ),
                )

        return GuardVerdict(proposed, proposed, intervened=False)
