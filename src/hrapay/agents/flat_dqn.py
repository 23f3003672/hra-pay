"""Flat dueling DQN — the non-hierarchical learned baseline.

This is the ablation that isolates the contribution of the branched action
space. It is a *dueling* DQN, not a plain one, and it uses the same trunk width,
the same replay buffer, the same optimiser and the same schedule as the branched
agent. The single difference is the action space:

    flat     one head over every valid (macro, timing, channel) combination
    branched three heads, one per action dimension, sharing a trunk

Enumerating the combinations gives 31 actions:

    ABANDON                                    1
    RETRY        x 5 timings                   5
    SWITCH_CHANNEL x 5 timings x 5 channels   25

31 is tractable. The point is what happens next: adding a sixth payment rail
takes the flat head to 37 and the branched agent to 14 outputs, and adding a
second decision dimension (say, which retry message to send) multiplies the flat
head while merely extending the branched one. This is the growth argument from
Tavakoli et al. (2018), made concrete on this problem.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from hrapay.agents.base import Decision, Policy
from hrapay.env.spec import MACRO_ACTIONS, EnvSpec


def enumerate_flat_actions(spec: EnvSpec) -> list[tuple[int, int, int]]:
    """Every distinct branched action, as (macro_idx, timing_idx, channel_idx).

    Channel is fixed to 0 for ABANDON and RETRY because it is ignored there —
    including it would create duplicate encodings of the same behaviour and
    split the Q-value estimate across them.
    """
    abandon = MACRO_ACTIONS.index("ABANDON")
    retry = MACRO_ACTIONS.index("RETRY")
    switch = MACRO_ACTIONS.index("SWITCH_CHANNEL")

    actions: list[tuple[int, int, int]] = [(abandon, 0, 0)]
    actions += [(retry, t, 0) for t in range(len(spec.time_buckets))]
    actions += [
        (switch, t, c) for t in range(len(spec.time_buckets)) for c in range(len(spec.channels))
    ]
    return actions


class DuelingHead(nn.Module):
    """Value/advantage decomposition over one set of actions."""

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.value = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.advantage = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=-1, keepdim=True)


class FlatQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = DuelingHead(hidden, n_actions, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class FlatDQNPolicy(Policy):
    """Greedy policy over a trained FlatQNetwork."""

    name = "flat_dqn"

    def __init__(
        self,
        spec: EnvSpec,
        network: FlatQNetwork,
        *,
        device: str = "cpu",
    ) -> None:
        self.spec = spec
        self.network = network.to(device)
        self.device = device
        self.flat_actions = enumerate_flat_actions(spec)
        self.labels = [self._label(macro, t, c) for (macro, t, c) in self.flat_actions]

    def _label(self, macro_idx: int, timing_idx: int, channel_idx: int) -> str:
        macro = MACRO_ACTIONS[macro_idx]
        if macro == "ABANDON":
            return "ABANDON"
        timing = self.spec.time_buckets[timing_idx]
        if macro == "RETRY":
            return f"RETRY@{timing}"
        return f"SWITCH>{self.spec.channels[channel_idx]}@{timing}"

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.network(x).squeeze(0).cpu().numpy()

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> Decision:  # noqa: ARG002
        q = self.q_values(obs)
        best = int(np.argmax(q))
        macro, timing, channel = self.flat_actions[best]

        order = np.argsort(-q)[:4]
        return Decision(
            np.array([macro, timing, channel]),
            {
                "chosen": self.labels[best],
                "q_chosen": round(float(q[best]), 4),
                "top_actions": [
                    {"action": self.labels[i], "q": round(float(q[i]), 4)} for i in order
                ],
                "n_flat_actions": len(self.flat_actions),
            },
        )


def load_flat_policy(spec: EnvSpec, checkpoint: str | Any, *, device: str = "cpu") -> FlatDQNPolicy:
    """Rebuild a trained policy from a checkpoint written by hrapay.train."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt.get("spec_version") != spec.version:
        raise ValueError(
            f"checkpoint was trained on spec '{ckpt.get('spec_version')}' but the "
            f"loaded spec is '{spec.version}'. Retrain, or load the matching spec."
        )
    net = FlatQNetwork(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return FlatDQNPolicy(spec, net, device=device)
