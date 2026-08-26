"""Ground-truth environment specification.

The spec is the generative model of the world: the probabilities that decide
whether a retry actually succeeds. The agent never observes it.

Everything is validated on load. A malformed spec should fail loudly at import
time, not silently produce a subtly wrong environment that we then train on for
six days and cannot explain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

SAME = "SAME"

MacroAction = Literal["RETRY", "SWITCH_CHANNEL", "ABANDON"]
MACRO_ACTIONS: tuple[str, ...] = ("RETRY", "SWITCH_CHANNEL", "ABANDON")


class DeclineCodeSpec(BaseModel):
    """How one decline reason behaves under retry."""

    weight: float = Field(gt=0.0, description="Relative frequency in the population.")
    terminal_prob: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "P(episode is fundamentally unrecoverable). LATENT — never observed by "
            "the agent. This is what makes ABANDON a real decision."
        ),
    )
    payday_sensitive: bool
    raw_reason_text: str = Field(
        min_length=20,
        description="Unstructured gateway string. The only thing the LLM calibrator reads.",
    )
    base_success: dict[str, float]
    time_multiplier: dict[str, float]

    @model_validator(mode="after")
    def _check_ranges(self) -> DeclineCodeSpec:
        for channel, p in self.base_success.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"base_success[{channel}]={p} outside [0, 1]")
        for bucket, m in self.time_multiplier.items():
            if m < 0.0:
                raise ValueError(f"time_multiplier[{bucket}]={m} is negative")
        if SAME not in self.base_success:
            raise ValueError(f"base_success must contain the '{SAME}' key")
        return self


class MerchantTierSpec(BaseModel):
    weight: float = Field(gt=0.0)
    success_multiplier: float = Field(gt=0.0)


class PaydaySpec(BaseModel):
    multiplier: float = Field(gt=0.0)
    days_of_month: list[int]

    @model_validator(mode="after")
    def _check_days(self) -> PaydaySpec:
        bad = [d for d in self.days_of_month if not 1 <= d <= 31]
        if bad:
            raise ValueError(f"payday.days_of_month contains invalid days: {bad}")
        return self


class AmountSpec(BaseModel):
    distribution: Literal["lognormal"]
    mean_log: float
    sigma_log: float = Field(gt=0.0)
    min: float = Field(gt=0.0)
    max: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_bounds(self) -> AmountSpec:
        if self.min >= self.max:
            raise ValueError("amount.min must be < amount.max")
        return self


class EnvSpec(BaseModel):
    """The complete generative specification of the retry environment."""

    version: str
    description: str

    channels: list[str]
    time_buckets: list[str]
    time_bucket_hours: dict[str, float]

    decline_codes: dict[str, DeclineCodeSpec]

    payday: PaydaySpec
    channel_fatigue: list[float]
    merchant_tiers: dict[str, MerchantTierSpec]
    amount: AmountSpec
    origin_channel_weights: dict[str, float]

    success_probability_cap: float = Field(gt=0.0, le=1.0)
    max_attempts_per_episode: int = Field(gt=0)
    episode_horizon_hours: float = Field(gt=0.0)

    # -- validation ---------------------------------------------------------

    @model_validator(mode="after")
    def _cross_check(self) -> EnvSpec:
        channels = set(self.channels)

        if set(self.time_bucket_hours) != set(self.time_buckets):
            raise ValueError("time_bucket_hours keys must match time_buckets exactly")

        if set(self.origin_channel_weights) != channels:
            raise ValueError("origin_channel_weights must cover every channel exactly")

        for code, dc in self.decline_codes.items():
            missing_ch = channels - set(dc.base_success)
            if missing_ch:
                raise ValueError(f"decline_codes.{code}.base_success missing {sorted(missing_ch)}")
            missing_tb = set(self.time_buckets) - set(dc.time_multiplier)
            if missing_tb:
                raise ValueError(
                    f"decline_codes.{code}.time_multiplier missing {sorted(missing_tb)}"
                )

        if not self.channel_fatigue:
            raise ValueError("channel_fatigue must not be empty")
        if any(f <= 0.0 for f in self.channel_fatigue):
            raise ValueError("channel_fatigue values must be positive")

        if not self.merchant_tiers:
            raise ValueError("merchant_tiers must not be empty")

        return self

    # -- convenience --------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> EnvSpec:
        raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    @property
    def decline_code_names(self) -> list[str]:
        """Stable, sorted ordering — used for one-hot encoding everywhere."""
        return sorted(self.decline_codes)

    @property
    def merchant_tier_names(self) -> list[str]:
        return sorted(self.merchant_tiers)

    def fatigue(self, attempts_on_channel: int) -> float:
        idx = min(attempts_on_channel, len(self.channel_fatigue) - 1)
        return self.channel_fatigue[idx]

    def is_payday(self, day_of_month: int) -> bool:
        return day_of_month in self.payday.days_of_month

    def success_probability(
        self,
        *,
        decline_code: str,
        target_channel: str,
        origin_channel: str,
        time_bucket: str,
        day_of_month: int,
        merchant_tier: str,
        attempts_on_target_channel: int,
    ) -> float:
        """Ground-truth P(retry succeeds). Oracle only — never given to the agent.

        Multiplicative model:
            base(code, channel) x time(code, bucket) x payday x fatigue x tier
        clipped to [0, success_probability_cap].
        """
        dc = self.decline_codes[decline_code]

        key = SAME if target_channel == origin_channel else target_channel
        p = dc.base_success[key]
        p *= dc.time_multiplier[time_bucket]

        if dc.payday_sensitive and self.is_payday(day_of_month):
            p *= self.payday.multiplier

        p *= self.fatigue(attempts_on_target_channel)
        p *= self.merchant_tiers[merchant_tier].success_multiplier

        return float(min(max(p, 0.0), self.success_probability_cap))
