"""Gymnasium environment for the payment-retry MDP.

One episode = one failed transaction. Each step the agent chooses a branched
action (macro, timing, channel); the environment advances the clock, samples
the retry outcome from the ground-truth spec, and returns a reward.

The action space is MultiDiscrete([3, 5, 5]) so that both the flat DQN baseline
and the branching agent can consume the same environment without modification.

Nothing in the observation is derived from the spec's probability tables. The
one exception is `channel_success_prior`, which is a Monte-Carlo estimate of
historical retry success — information any real merchant could compute from
their own logs. See generator.estimate_channel_priors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hrapay.env.generator import Episode, EpisodeGenerator
from hrapay.env.spec import MACRO_ACTIONS, EnvSpec

# Timing buckets used when computing "how long since the last attempt".
ELAPSED_BUCKETS: tuple[str, ...] = ("0-2h", "2-6h", "6-24h", "24-72h", "72h+")


def _elapsed_bucket(hours: float) -> int:
    if hours < 2:
        return 0
    if hours < 6:
        return 1
    if hours < 24:
        return 2
    if hours < 72:
        return 3
    return 4


@dataclass
class StepContext:
    """Everything a reward function may look at for one decision.

    Deliberately explicit rather than a dict: the reward function is the part of
    this system most likely to be wrong in a way that is hard to see, so its
    inputs are typed and named.
    """

    episode: Episode
    macro: str
    timing_bucket: str
    target_channel: str | None
    elapsed_hours_total: float
    attempts_total: int
    attempts_on_target_channel: int
    friction_penalty: float
    success: bool
    terminated_by_abandon: bool


RewardFn = Callable[[StepContext], float]


class _DefaultReward:
    """Placeholder reward used until the calibrated reward lands (Day 2/3).

    Kept intentionally simple and legible so that any behaviour the environment
    shows on Day 1 is attributable to the dynamics, not to a clever reward.
    """

    retry_cost_fraction = 0.015
    time_decay_per_day = 0.004
    abandon_bonus = 0.10
    abandon_penalty = 0.25

    def __call__(self, ctx: StepContext) -> float:
        amount = ctx.episode.amount
        if ctx.terminated_by_abandon:
            return amount * (
                self.abandon_bonus if ctx.episode.is_terminal else -self.abandon_penalty
            )

        r = amount if ctx.success else 0.0
        r -= amount * self.retry_cost_fraction
        r -= amount * ctx.friction_penalty * 0.01
        r -= amount * self.time_decay_per_day * (ctx.elapsed_hours_total / 24.0)
        return float(r)


class FrictionTable(Protocol):
    def penalty_for(self, decline_code: str) -> float: ...


class _ZeroFriction:
    """Day-1 stand-in. Replaced on Day 3 by the LLM-calibrated table."""

    def penalty_for(self, decline_code: str) -> float:  # noqa: ARG002
        return 0.0


class RetryEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-transaction payment-retry MDP."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        spec: EnvSpec,
        *,
        seed: int = 0,
        channel_priors: dict[str, dict[str, float]] | None = None,
        friction_table: FrictionTable | None = None,
        reward_fn: RewardFn | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.channel_priors = channel_priors or {}
        self.friction_table: FrictionTable = friction_table or _ZeroFriction()
        self.reward_fn: RewardFn = reward_fn or _DefaultReward()

        self._generator = EpisodeGenerator(spec, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._episode_index = 0

        self.action_space = spaces.MultiDiscrete(
            [len(MACRO_ACTIONS), len(spec.time_buckets), len(spec.channels)]
        )
        self.feature_names = self._build_feature_names()
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(len(self.feature_names),), dtype=np.float32
        )

        # Episode state
        self.episode: Episode
        self._current_channel: str = ""
        self._elapsed_hours: float = 0.0
        self._attempts_total: int = 0
        self._attempts_by_channel: dict[str, int] = {}
        self._channels_tried: set[str] = set()
        self._last_gap_hours: float = 0.0

    # -- observation --------------------------------------------------------

    def _build_feature_names(self) -> list[str]:
        s = self.spec
        names = [f"decline={c}" for c in s.decline_code_names]
        names.append("friction_penalty")
        names += [f"elapsed={b}" for b in ELAPSED_BUCKETS]
        names += ["retry_count_channel", "retry_count_total"]
        names += [f"tried={c}" for c in s.channels]
        names += [f"current={c}" for c in s.channels]
        names.append("amount_norm")
        names += [f"tier={t}" for t in s.merchant_tier_names]
        names.append("is_likely_payday")
        names += [f"prior={c}" for c in s.channels]
        return names

    def _observe(self) -> np.ndarray:
        s = self.spec
        ep = self.episode
        v: list[float] = []

        v += [1.0 if c == ep.decline_code else 0.0 for c in s.decline_code_names]
        v.append(self.friction_table.penalty_for(ep.decline_code) / 10.0)

        bucket = _elapsed_bucket(self._last_gap_hours)
        v += [1.0 if i == bucket else 0.0 for i in range(len(ELAPSED_BUCKETS))]

        v.append(self._attempts_by_channel.get(self._current_channel, 0) / 4.0)
        v.append(self._attempts_total / float(s.max_attempts_per_episode))

        v += [1.0 if c in self._channels_tried else 0.0 for c in s.channels]
        v += [1.0 if c == self._current_channel else 0.0 for c in s.channels]

        v.append(float(np.log1p(ep.amount) / 12.0))
        v += [1.0 if t == ep.merchant_tier else 0.0 for t in s.merchant_tier_names]
        v.append(1.0 if s.is_payday(self._current_day_of_month()) else 0.0)

        priors = self.channel_priors.get(ep.decline_code, {})
        v += [float(priors.get(c, 0.0)) for c in s.channels]

        return np.asarray(v, dtype=np.float32)

    def _current_day_of_month(self) -> int:
        day = self.episode.start_day_of_month + int(self._elapsed_hours // 24)
        return ((day - 1) % 31) + 1

    # -- gym API ------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._generator.reset(seed)
            self._episode_index = 0

        forced: Episode | None = (options or {}).get("episode")
        if forced is not None:
            self.episode = forced
        else:
            self.episode = self._generator.sample(self._episode_index)
            self._episode_index += 1

        self._current_channel = self.episode.origin_channel
        self._elapsed_hours = 0.0
        self._last_gap_hours = 0.0
        self._attempts_total = 0
        self._attempts_by_channel = {self.episode.origin_channel: 1}
        self._channels_tried = {self.episode.origin_channel}

        return self._observe(), {"episode": self.episode.to_dict()}

    def decode_action(self, action: np.ndarray | list[int]) -> tuple[str, str, str | None]:
        """Turn the raw branch indices into a human-readable action triple."""
        a = np.asarray(action, dtype=int).ravel()
        macro = MACRO_ACTIONS[int(a[0])]
        timing = self.spec.time_buckets[int(a[1])]
        channel = self.spec.channels[int(a[2])]

        if macro == "ABANDON":
            return macro, timing, None
        if macro == "RETRY":
            return macro, timing, self._current_channel
        if channel == self._current_channel:
            # Switching to the channel we are already on is just a retry.
            return "RETRY", timing, self._current_channel
        return macro, timing, channel

    def step(
        self, action: np.ndarray | list[int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        spec = self.spec
        ep = self.episode
        macro, timing, target = self.decode_action(action)

        info: dict[str, Any] = {
            "episode_id": ep.episode_id,
            "macro": macro,
            "timing": timing,
            "target_channel": target,
            "attempts_total_before": self._attempts_total,
        }

        # --- ABANDON: terminate immediately, no clock advance ---------------
        if macro == "ABANDON":
            ctx = StepContext(
                episode=ep,
                macro=macro,
                timing_bucket=timing,
                target_channel=None,
                elapsed_hours_total=self._elapsed_hours,
                attempts_total=self._attempts_total,
                attempts_on_target_channel=0,
                friction_penalty=self.friction_table.penalty_for(ep.decline_code),
                success=False,
                terminated_by_abandon=True,
            )
            reward = self.reward_fn(ctx)
            info.update(
                {
                    "outcome": "ABANDONED",
                    "success": False,
                    "p_success_ORACLE": 0.0,
                    "is_terminal_ORACLE": ep.is_terminal,
                    "abandon_was_correct_ORACLE": ep.is_terminal,
                    "elapsed_hours": self._elapsed_hours,
                }
            )
            return self._observe(), reward, True, False, info

        assert target is not None
        gap = spec.time_bucket_hours[timing]
        self._elapsed_hours += gap
        self._last_gap_hours = gap

        attempts_on_target = self._attempts_by_channel.get(target, 0)

        p = 0.0
        if not ep.is_terminal:
            p = spec.success_probability(
                decline_code=ep.decline_code,
                target_channel=target,
                origin_channel=ep.origin_channel,
                time_bucket=timing,
                day_of_month=self._current_day_of_month(),
                merchant_tier=ep.merchant_tier,
                attempts_on_target_channel=attempts_on_target,
            )
        success = bool(self._rng.random() < p)

        self._attempts_total += 1
        self._attempts_by_channel[target] = attempts_on_target + 1
        self._channels_tried.add(target)
        self._current_channel = target

        ctx = StepContext(
            episode=ep,
            macro=macro,
            timing_bucket=timing,
            target_channel=target,
            elapsed_hours_total=self._elapsed_hours,
            attempts_total=self._attempts_total,
            attempts_on_target_channel=attempts_on_target,
            friction_penalty=self.friction_table.penalty_for(ep.decline_code),
            success=success,
            terminated_by_abandon=False,
        )
        reward = self.reward_fn(ctx)

        terminated = success
        truncated = (not success) and (
            self._attempts_total >= spec.max_attempts_per_episode
            or self._elapsed_hours >= spec.episode_horizon_hours
        )

        info.update(
            {
                "outcome": "RECOVERED" if success else ("EXHAUSTED" if truncated else "PENDING"),
                "success": success,
                "p_success_ORACLE": round(p, 6),
                "is_terminal_ORACLE": ep.is_terminal,
                "elapsed_hours": self._elapsed_hours,
                "attempts_total": self._attempts_total,
                "amount": ep.amount,
            }
        )
        return self._observe(), reward, terminated, truncated, info

    def render(self) -> str:
        ep = self.episode
        return (
            f"{ep.episode_id} | {ep.decline_code} | Rs {ep.amount:,.0f} "
            f"| on {self._current_channel} | t+{self._elapsed_hours:.0f}h "
            f"| attempts={self._attempts_total}"
        )
