"""Synthetic failed-transaction generator.

Produces the population of failed payments the agent is asked to recover.
Everything is driven by a seeded numpy Generator so any reported result can be
reproduced exactly from (spec_version, seed, n_episodes).

Note on `is_terminal`: this is sampled here and stored on the episode, but it is
LATENT. The environment uses it to force P(success)=0; the agent never sees it.
Without it, ABANDON would be a dominated action and the "stopping rule" the
track asks for would be untestable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hrapay.env.spec import EnvSpec


@dataclass(frozen=True)
class Episode:
    """One failed transaction awaiting a recovery decision."""

    episode_id: str
    decline_code: str
    origin_channel: str
    amount: float
    merchant_tier: str
    start_day_of_month: int
    is_terminal: bool = field(repr=False)  # latent — hidden from the agent

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "decline_code": self.decline_code,
            "origin_channel": self.origin_channel,
            "amount": round(self.amount, 2),
            "merchant_tier": self.merchant_tier,
            "start_day_of_month": self.start_day_of_month,
            "is_terminal_ORACLE": self.is_terminal,
        }


def _weighted_choice(rng: np.random.Generator, options: dict[str, float]) -> str:
    keys = sorted(options)
    weights = np.array([options[k] for k in keys], dtype=np.float64)
    weights = weights / weights.sum()
    return str(rng.choice(keys, p=weights))


class EpisodeGenerator:
    """Samples Episodes from an EnvSpec."""

    def __init__(self, spec: EnvSpec, seed: int = 0) -> None:
        self.spec = spec
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    def sample(self, index: int) -> Episode:
        spec, rng = self.spec, self._rng

        code_weights = {name: dc.weight for name, dc in spec.decline_codes.items()}
        decline_code = _weighted_choice(rng, code_weights)
        dc = spec.decline_codes[decline_code]

        origin_channel = _weighted_choice(rng, spec.origin_channel_weights)

        tier_weights = {name: t.weight for name, t in spec.merchant_tiers.items()}
        merchant_tier = _weighted_choice(rng, tier_weights)

        amount = float(
            np.clip(
                rng.lognormal(mean=spec.amount.mean_log, sigma=spec.amount.sigma_log),
                spec.amount.min,
                spec.amount.max,
            )
        )

        start_day_of_month = int(rng.integers(1, 29))
        is_terminal = bool(rng.random() < dc.terminal_prob)

        return Episode(
            episode_id=f"txn_{self.seed:04d}_{index:06d}",
            decline_code=decline_code,
            origin_channel=origin_channel,
            amount=amount,
            merchant_tier=merchant_tier,
            start_day_of_month=start_day_of_month,
            is_terminal=is_terminal,
        )

    def sample_batch(self, n: int) -> list[Episode]:
        return [self.sample(i) for i in range(n)]


def estimate_channel_priors(
    spec: EnvSpec, n_samples: int = 40_000, seed: int = 12345
) -> dict[str, dict[str, float]]:
    """Empirical P(success | decline_code, channel), as a merchant could compute it.

    This is the one piece of "ground truth adjacent" information the agent IS
    allowed to see, because any real merchant can compute it from their own
    historical retry logs. It is estimated by Monte Carlo over the population
    rather than read off the spec, so it carries realistic sampling noise and
    is marginalised over timing, payday, tier and fatigue — exactly the way a
    real historical estimate would be.
    """
    rng = np.random.default_rng(seed)
    gen = EpisodeGenerator(spec, seed=seed)

    successes: dict[str, dict[str, float]] = {
        code: dict.fromkeys(spec.channels, 0.0) for code in spec.decline_codes
    }
    trials: dict[str, dict[str, float]] = {
        code: dict.fromkeys(spec.channels, 0.0) for code in spec.decline_codes
    }

    for i in range(n_samples):
        ep = gen.sample(i)
        channel = str(rng.choice(spec.channels))
        bucket = str(rng.choice(spec.time_buckets))
        attempts = int(rng.integers(0, 3))
        day = int(rng.integers(1, 29))

        p = 0.0
        if not ep.is_terminal:
            p = spec.success_probability(
                decline_code=ep.decline_code,
                target_channel=channel,
                origin_channel=ep.origin_channel,
                time_bucket=bucket,
                day_of_month=day,
                merchant_tier=ep.merchant_tier,
                attempts_on_target_channel=attempts,
            )

        trials[ep.decline_code][channel] += 1.0
        successes[ep.decline_code][channel] += float(rng.random() < p)

    priors: dict[str, dict[str, float]] = {}
    for code in spec.decline_codes:
        priors[code] = {
            ch: (successes[code][ch] / trials[code][ch]) if trials[code][ch] > 0 else 0.0
            for ch in spec.channels
        }
    return priors
