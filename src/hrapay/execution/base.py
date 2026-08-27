"""Executor interface — the boundary where a decision becomes a side effect.

Deliberately separated from the environment. The environment owns the ground
truth about whether a retry succeeds; the executor only *performs* the attempt
and returns a reference to it. Conflating the two would make it impossible to
swap in a real payment gateway later without also changing the simulation.

Two implementations:
    SimulatedExecutor      no side effect, synthetic reference id (default)
    RazorpayTestModeExecutor   creates a real test-mode order (Day 9, opt-in)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RetryRequest:
    """A concrete instruction to attempt recovery of one transaction."""

    episode_id: str
    amount_inr: float
    channel: str
    scheduled_in_hours: float
    decline_code: str
    attempt_number: int


@dataclass(frozen=True)
class RetryResult:
    """What the execution layer returned. Does NOT contain success/failure.

    Whether the payment succeeds is decided by the world (the environment in
    simulation, the issuer in production) and arrives asynchronously. All the
    executor reports is that the attempt was dispatched, and how to find it.
    """

    reference_id: str
    executor: str
    dispatched: bool
    detail: dict[str, Any] = field(default_factory=dict)


class Executor(Protocol):
    name: str

    def execute(self, request: RetryRequest) -> RetryResult: ...
