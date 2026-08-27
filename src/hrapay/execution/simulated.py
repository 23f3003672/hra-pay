"""In-memory executor. The default for training and evaluation.

Performs no side effect and calls no network. Its only job is to produce a
stable reference id so that the audit trail has the same shape whether or not a
real gateway is wired in — which is what makes the Day-9 swap to Razorpay
test mode a configuration change rather than a rewrite.
"""

from __future__ import annotations

from hrapay.execution.base import Executor, RetryRequest, RetryResult


class SimulatedExecutor(Executor):
    name = "simulated"

    def __init__(self) -> None:
        self._counter = 0

    def execute(self, request: RetryRequest) -> RetryResult:
        self._counter += 1
        return RetryResult(
            reference_id=f"sim_{request.episode_id}_{request.attempt_number:02d}",
            executor=self.name,
            dispatched=True,
            detail={
                "channel": request.channel,
                "amount_inr": round(request.amount_inr, 2),
                "scheduled_in_hours": request.scheduled_in_hours,
                "note": "simulated — no network call, no side effect",
            },
        )
