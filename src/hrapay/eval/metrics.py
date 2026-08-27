"""Metrics.

Plain arithmetic over EpisodeResults. Nothing generative goes anywhere near
this file — the scoreboard has to be something a reviewer can recompute by hand
from the audit log.

Five metrics, chosen so that no single one can be gamed without visibly
worsening another:

    recovered_inr          the headline. Money actually recovered.
    recovery_rate          share of RECOVERABLE transactions recovered.
                           Denominator excludes truly-unrecoverable ones, so a
                           policy cannot look good simply by being handed an
                           easier population.
    wasted_attempts        attempts that never led to a recovery. Rises if a
                           policy chases recovered_inr indiscriminately.
    issuer_risk_exposure   attempts made on high-risk decline codes. Should be
                           zero. Non-zero means the guard was bypassed.
    mean_time_to_recovery  hours until a successful retry. A policy can raise
                           recovery_rate by simply waiting longer, and this is
                           the metric that shows the price of doing so.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from hrapay.eval.runner import EpisodeResult


@dataclass
class Metrics:
    policy: str
    n_episodes: int
    n_recoverable: int

    recovered_inr: float
    recoverable_inr: float
    recovery_rate: float
    revenue_capture_rate: float

    total_attempts: int
    wasted_attempts: int
    attempts_per_episode: float

    issuer_risk_exposure: int
    guard_interventions: int

    mean_time_to_recovery_h: float
    correct_abandon_rate: float
    premature_abandon_rate: float

    mean_reward: float

    def to_row(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "episodes": self.n_episodes,
            "recovered_inr": round(self.recovered_inr, 2),
            "recovery_rate": round(self.recovery_rate, 4),
            "revenue_capture_rate": round(self.revenue_capture_rate, 4),
            "wasted_attempts": self.wasted_attempts,
            "attempts_per_episode": round(self.attempts_per_episode, 3),
            "issuer_risk_exposure": self.issuer_risk_exposure,
            "guard_interventions": self.guard_interventions,
            "mean_time_to_recovery_h": round(self.mean_time_to_recovery_h, 2),
            "correct_abandon_rate": round(self.correct_abandon_rate, 4),
            "premature_abandon_rate": round(self.premature_abandon_rate, 4),
            "mean_reward": round(self.mean_reward, 5),
        }


def compute(results: list[EpisodeResult], policy: str) -> Metrics:
    if not results:
        raise ValueError("cannot compute metrics over an empty result set")

    recoverable = [r for r in results if r.was_recoverable]
    recovered = [r for r in results if r.recovered]
    abandons = [r for r in results if r.abandoned]

    recoverable_inr = sum(r.amount_inr for r in recoverable)
    recovered_inr = sum(r.recovered_amount for r in results)

    return Metrics(
        policy=policy,
        n_episodes=len(results),
        n_recoverable=len(recoverable),
        recovered_inr=recovered_inr,
        recoverable_inr=recoverable_inr,
        recovery_rate=(len(recovered) / len(recoverable)) if recoverable else 0.0,
        revenue_capture_rate=(recovered_inr / recoverable_inr) if recoverable_inr else 0.0,
        total_attempts=sum(r.attempts for r in results),
        wasted_attempts=sum(r.wasted_attempts for r in results),
        attempts_per_episode=mean(r.attempts for r in results),
        issuer_risk_exposure=sum(r.high_risk_attempts for r in results),
        guard_interventions=sum(r.guard_interventions for r in results),
        mean_time_to_recovery_h=mean([r.elapsed_hours for r in recovered]) if recovered else 0.0,
        correct_abandon_rate=(
            sum(1 for r in abandons if r.is_terminal_ORACLE) / len(abandons) if abandons else 0.0
        ),
        premature_abandon_rate=(
            sum(1 for r in abandons if not r.is_terminal_ORACLE) / len(abandons)
            if abandons
            else 0.0
        ),
        mean_reward=mean(r.total_reward for r in results),
    )


def aggregate_seeds(per_seed: list[Metrics]) -> dict[str, Any]:
    """Mean and standard deviation across seeds.

    Single-seed numbers are not a result. Every headline figure in the README
    goes through this so the spread is reported alongside the mean.
    """
    if not per_seed:
        raise ValueError("no seeds to aggregate")

    rows = [m.to_row() for m in per_seed]
    out: dict[str, Any] = {"policy": per_seed[0].policy, "seeds": len(per_seed)}
    for key in rows[0]:
        if key == "policy":
            continue
        values = [float(r[key]) for r in rows]
        out[f"{key}_mean"] = round(mean(values), 4)
        out[f"{key}_std"] = round(pstdev(values) if len(values) > 1 else 0.0, 4)
    return out
