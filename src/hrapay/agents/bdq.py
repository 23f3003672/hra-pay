"""Branching Dueling Q-Network.

Tavakoli, Pardo & Kormushev (2018), "Action Branching Architectures for Deep
Reinforcement Learning", AAAI 32(1), arXiv:1711.08946.

A shared trunk feeds three independent output branches — macro, timing, channel —
so the network's output size grows *additively* with the number of decisions
rather than multiplicatively:

    flat head      1 + 5 + (5 x 5)          = 31 outputs
    branched       3 + 5 + 5                = 13 outputs

Following the paper, there is one **shared** state-value estimator V(s) and a
per-branch advantage stream, aggregated as

    Q_d(s, a_d) = V(s) + A_d(s, a_d) - mean_{a'} A_d(s, a')

and a single shared TD target formed by averaging the per-branch target values.
The paper found a common V and a mean-aggregated target to be the most stable of
the variants it tested, which is why those are the choices here rather than
per-branch value heads.

Everything else — trunk width, optimiser, replay buffer, epsilon schedule,
Double-DQN targets — is identical to the flat baseline. The action-space
architecture is the only thing that differs between the two runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from hrapay.agents.base import Decision, Policy
from hrapay.env.spec import MACRO_ACTIONS, EnvSpec

ABANDON_IDX = MACRO_ACTIONS.index("ABANDON")
SWITCH_IDX = MACRO_ACTIONS.index("SWITCH_CHANNEL")


def branch_activity_mask(macro_indices: np.ndarray | torch.Tensor) -> np.ndarray:
    """Which branches actually influenced the outcome, per transition.

    The environment canonicalises actions: timing is ignored when the macro is
    ABANDON, and channel is ignored unless the macro is SWITCH_CHANNEL. Without
    masking, the timing branch would be trained on the reward from every ABANDON
    transition — a gradient signal for a choice that had no effect on anything.
    That is pure noise injected into a head we need to be sharp.

    Returns a (B, 3) float mask over [macro, timing, channel].
    """
    m = np.asarray(macro_indices).reshape(-1)
    mask = np.zeros((len(m), 3), dtype=np.float32)
    mask[:, 0] = 1.0  # macro always matters
    mask[:, 1] = m != ABANDON_IDX  # timing matters unless abandoning
    mask[:, 2] = m == SWITCH_IDX  # channel matters only when switching
    return mask


class BranchingQNetwork(nn.Module):
    """Shared trunk, shared value head, one advantage head per action dimension."""

    def __init__(self, obs_dim: int, branch_sizes: list[int], hidden: int = 128) -> None:
        super().__init__()
        self.branch_sizes = branch_sizes

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # One shared state-value estimator across all branches (per the paper).
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.advantages = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n))
                for n in branch_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns one (B, n_actions_d) tensor of Q-values per branch."""
        z = self.trunk(x)
        v = self.value(z)
        return [v + a(z) - a(z).mean(dim=-1, keepdim=True) for a in self.advantages]

    @property
    def n_outputs(self) -> int:
        """13 for this problem, against the flat head's 31."""
        return sum(self.branch_sizes)


class BDQPolicy(Policy):
    """Greedy policy over a trained BranchingQNetwork."""

    name = "bdq"

    def __init__(self, spec: EnvSpec, network: BranchingQNetwork, *, device: str = "cpu") -> None:
        self.spec = spec
        self.network = network.to(device)
        self.device = device
        self.branch_labels = [
            list(MACRO_ACTIONS),
            list(spec.time_buckets),
            list(spec.channels),
        ]

    @torch.no_grad()
    def branch_q_values(self, obs: np.ndarray) -> list[np.ndarray]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return [q.squeeze(0).cpu().numpy() for q in self.network(x)]

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> Decision:  # noqa: ARG002
        qs = self.branch_q_values(obs)
        choice = [int(np.argmax(q)) for q in qs]

        # Per-branch Q-values go into the audit trail. This is the readable
        # advantage of the branched design: the log can show *separately* what
        # the agent thought about whether to retry, when, and on which rail —
        # rather than one score over an opaque 31-way combination.
        diagnostics: dict[str, Any] = {"n_branch_outputs": self.network.n_outputs}
        for name, labels, q, picked in zip(
            ("macro", "timing", "channel"), self.branch_labels, qs, choice, strict=True
        ):
            diagnostics[name] = {
                "chosen": labels[picked],
                "q": round(float(q[picked]), 4),
                "all": {lbl: round(float(v), 4) for lbl, v in zip(labels, q, strict=True)},
            }
        return Decision(np.array(choice), diagnostics)


def load_bdq_policy(spec: EnvSpec, checkpoint: str | Any, *, device: str = "cpu") -> BDQPolicy:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt.get("spec_version") != spec.version:
        raise ValueError(
            f"checkpoint was trained on spec '{ckpt.get('spec_version')}' but the "
            f"loaded spec is '{spec.version}'. Retrain, or load the matching spec."
        )
    net = BranchingQNetwork(ckpt["obs_dim"], ckpt["branch_sizes"], ckpt["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return BDQPolicy(spec, net, device=device)
