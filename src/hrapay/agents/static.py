"""Static retry schedule — the industry-standard baseline.

This is what the agent has to beat, and it is modelled on the publicly
documented behaviour of production smart-retry systems: a fixed sequence of
delays on the original instrument, then stop. No channel switching, no
conditioning on the decline reason.

It is implemented as a real Policy rather than special-cased in the harness so
that it runs through exactly the same guard, executor and audit path as the
learned agents. A baseline that gets an easier code path is not a baseline.

Two variants exist on purpose:
    StaticSchedulePolicy         fixed delays, original channel only
    StaticWithChannelSwitchPolicy  fixed delays, falls back to UPI once

The second is the stronger, fairer baseline. Reporting only against the weaker
one would inflate the agent's apparent advantage.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hrapay.agents.base import Decision, Policy
from hrapay.env.spec import MACRO_ACTIONS, EnvSpec

RETRY = MACRO_ACTIONS.index("RETRY")
SWITCH = MACRO_ACTIONS.index("SWITCH_CHANNEL")
ABANDON = MACRO_ACTIONS.index("ABANDON")


class StaticSchedulePolicy(Policy):
    """Retry the original instrument at fixed delays, then give up."""

    name = "static_schedule"

    def __init__(self, spec: EnvSpec, schedule: list[str] | None = None) -> None:
        self.spec = spec
        self.schedule = schedule or ["PLUS_24H", "PLUS_72H", "PLUS_72H"]
        self._step = 0

    def reset_episode(self) -> None:
        self._step = 0

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> Decision:  # noqa: ARG002
        if self._step >= len(self.schedule):
            return Decision(
                np.array([ABANDON, 0, 0]),
                {"rule": "schedule exhausted", "step": self._step},
            )

        timing = self.schedule[self._step]
        timing_idx = self.spec.time_buckets.index(timing)
        self._step += 1
        return Decision(
            np.array([RETRY, timing_idx, 0]),
            {"rule": f"scheduled attempt {self._step} at {timing}", "step": self._step},
        )


class StaticWithChannelSwitchPolicy(Policy):
    """Fixed delays, with a single hard-coded fallback to UPI.

    Represents the better class of production retry logic: someone has noticed
    that an alternate rail sometimes works, and hard-coded one fallback. It
    still cannot condition the choice on the decline reason, which is the gap
    the learned policy is supposed to exploit.
    """

    name = "static_with_switch"

    def __init__(
        self,
        spec: EnvSpec,
        schedule: list[str] | None = None,
        fallback_channel: str = "UPI",
        switch_after: int = 1,
    ) -> None:
        self.spec = spec
        self.schedule = schedule or ["PLUS_24H", "PLUS_24H", "PLUS_72H"]
        self.fallback_channel = fallback_channel
        self.switch_after = switch_after
        self._step = 0

    def reset_episode(self) -> None:
        self._step = 0

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> Decision:  # noqa: ARG002
        if self._step >= len(self.schedule):
            return Decision(
                np.array([ABANDON, 0, 0]),
                {"rule": "schedule exhausted", "step": self._step},
            )

        timing_idx = self.spec.time_buckets.index(self.schedule[self._step])
        channel_idx = self.spec.channels.index(self.fallback_channel)

        if self._step == self.switch_after:
            macro, rule = SWITCH, f"fallback to {self.fallback_channel}"
        else:
            macro, rule = RETRY, f"scheduled attempt {self._step + 1}"

        self._step += 1
        return Decision(
            np.array([macro, timing_idx, channel_idx]),
            {"rule": rule, "step": self._step},
        )
