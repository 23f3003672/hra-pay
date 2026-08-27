"""Reward function.

Expressed as a fraction of transaction value rather than in rupees. See
configs/default.yaml for the reasoning and the tunable constants.

The friction penalty is supplied by a FrictionTable. Until Day 3 that is a
zero table; afterwards it is the LLM-calibrated, human-reviewed one. The reward
function itself does not change when the table is swapped, which is the point:
the calibration is data, not code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from hrapay.env.retry_env import FrictionTable, StepContext, ZeroFrictionTable


class RetryCostConfig(BaseModel):
    fixed_fee_inr: float = Field(ge=0.0)
    variable_rate: float = Field(ge=0.0)


class RewardConfig(BaseModel):
    success_value: float = Field(gt=0.0)
    retry_cost: RetryCostConfig
    friction_weight: float = Field(ge=0.0)
    time_decay_per_day: float = Field(ge=0.0)
    abandon_bonus: float = Field(ge=0.0)
    abandon_penalty: float = Field(ge=0.0)

    @classmethod
    def load(cls, path: str | Path) -> RewardConfig:
        raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw["reward"])


class CalibratedReward:
    """R_t, in units of transaction value.

    Retry:
        + 1.0                                    if the attempt succeeds
        - (fixed_fee / amount + variable_rate)   cost of making the attempt
        - friction_weight * penalty(code) / 10   issuer/compliance friction
        - time_decay_per_day * elapsed_days      cost of delay

    Abandon:
        + abandon_bonus     if the transaction was genuinely unrecoverable
        - abandon_penalty   if it was still recoverable

    The abandon branch is the only place the environment's latent `is_terminal`
    reaches the reward. That is deliberate and is what teaches a stopping rule:
    the agent is scored on whether giving up was *correct*, which it cannot
    observe at decision time and must therefore infer.
    """

    def __init__(self, config: RewardConfig, friction_table: FrictionTable | None = None) -> None:
        self.config = config
        self.friction_table: FrictionTable = friction_table or ZeroFrictionTable()

    def retry_cost_fraction(self, amount: float) -> float:
        c = self.config.retry_cost
        return c.fixed_fee_inr / max(amount, 1.0) + c.variable_rate

    def __call__(self, ctx: StepContext) -> float:
        cfg = self.config

        if ctx.terminated_by_abandon:
            return float(cfg.abandon_bonus if ctx.episode.is_terminal else -cfg.abandon_penalty)

        r = cfg.success_value if ctx.success else 0.0
        r -= self.retry_cost_fraction(ctx.episode.amount)
        r -= cfg.friction_weight * (
            self.friction_table.penalty_for(ctx.episode.decline_code) / 10.0
        )
        r -= cfg.time_decay_per_day * (ctx.elapsed_hours_total / 24.0)
        return float(r)

    def explain(self, ctx: StepContext) -> dict[str, float]:
        """Per-term breakdown, for the audit trail and the dashboard.

        A single reward number is not auditable. This makes every term visible
        so a reviewer can see exactly why an action scored the way it did.
        """
        cfg = self.config
        if ctx.terminated_by_abandon:
            value = cfg.abandon_bonus if ctx.episode.is_terminal else -cfg.abandon_penalty
            return {"abandon": float(value), "total": float(value)}

        terms = {
            "recovery": cfg.success_value if ctx.success else 0.0,
            "retry_cost": -self.retry_cost_fraction(ctx.episode.amount),
            "friction": -cfg.friction_weight
            * (self.friction_table.penalty_for(ctx.episode.decline_code) / 10.0),
            "time_decay": -cfg.time_decay_per_day * (ctx.elapsed_hours_total / 24.0),
        }
        terms["total"] = float(sum(terms.values()))
        return {k: float(v) for k, v in terms.items()}
