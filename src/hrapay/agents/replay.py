"""Uniform experience replay.

Deliberately plain. Prioritised replay would probably help both agents, but it
would also be a second thing changing between the flat and branched runs, and
the entire point of the flat baseline is that exactly one thing differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray
    action: np.ndarray  # (B, 3) branch indices
    reward: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, *, seed: int = 0) -> None:
        self.capacity = capacity
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, 3), dtype=np.int64)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._idx = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._idx

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._idx
        self._obs[i] = obs
        self._action[i] = action
        self._reward[i] = reward
        self._next_obs[i] = next_obs
        self._done[i] = float(done)
        self._idx = (i + 1) % self.capacity
        self._full = self._full or self._idx == 0

    def sample(self, batch_size: int) -> Batch:
        idx = self._rng.integers(0, len(self), size=batch_size)
        return Batch(
            obs=self._obs[idx],
            action=self._action[idx],
            reward=self._reward[idx],
            next_obs=self._next_obs[idx],
            done=self._done[idx],
        )
