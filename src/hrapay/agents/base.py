"""Policy interface shared by every agent, learned or not.

All three policies compared in the results table — static schedule, flat DQN,
branched BDQ — implement this. That is what makes the comparison honest: they
see the same observations, run through the same guard, and are scored by the
same harness. No policy gets a private code path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Decision:
    """A policy's chosen action plus whatever it wants on the record.

    `diagnostics` is where a learned agent puts its Q-values. It goes straight
    into the audit trail, which is what turns "the model decided" into "the
    model scored SWITCH_CHANNEL at 0.41 against RETRY at 0.22".
    """

    action: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Policy(ABC):
    """Maps an observation to a branched action [macro, timing, channel]."""

    name: str = "unnamed"

    def reset_episode(self) -> None:  # noqa: B027
        """Clear any per-episode internal state. Called on every env.reset().

        Intentionally concrete-and-empty rather than abstract: a stateless
        policy should not be forced to write an empty override.
        """

    @abstractmethod
    def act(self, obs: np.ndarray, info: dict[str, Any]) -> Decision: ...
