"""Episode runner: the single code path every policy goes through.

    policy proposes -> guard reviews -> executor dispatches -> env resolves
                    -> audit records everything

There is exactly one of these, shared by all policies. If the static baseline
and the learned agent were run by different loops, any difference in the
results table could be an artefact of the loop rather than the policy, and the
whole comparison would be worthless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hrapay.agents.base import Policy
from hrapay.audit.logger import AuditLogger, AuditRecord
from hrapay.env.retry_env import RetryEnv
from hrapay.env.spec import MACRO_ACTIONS
from hrapay.execution.base import Executor, RetryRequest
from hrapay.execution.simulated import SimulatedExecutor
from hrapay.guard.policy_guard import GuardState, PolicyGuard
from hrapay.rewards.reward import CalibratedReward


@dataclass
class EpisodeResult:
    """Outcome of one transaction, in the terms the metrics care about."""

    episode_id: str
    decline_code: str
    amount_inr: float
    recovered: bool
    abandoned: bool
    is_terminal_ORACLE: bool
    attempts: int
    wasted_attempts: int
    elapsed_hours: float
    total_reward: float
    guard_interventions: int
    high_risk_attempts: int
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def recovered_amount(self) -> float:
        return self.amount_inr if self.recovered else 0.0

    @property
    def was_recoverable(self) -> bool:
        return not self.is_terminal_ORACLE


class EpisodeRunner:
    def __init__(
        self,
        env: RetryEnv,
        guard: PolicyGuard,
        reward: CalibratedReward,
        *,
        executor: Executor | None = None,
        audit: AuditLogger | None = None,
        run_id: str = "run",
        high_risk_codes: tuple[str, ...] = ("suspected_fraud", "account_closed"),
    ) -> None:
        self.env = env
        self.guard = guard
        self.reward = reward
        self.executor = executor or SimulatedExecutor()
        self.audit = audit
        self.run_id = run_id
        self.high_risk_codes = high_risk_codes

        # The env computes reward through whatever function it was given; make
        # sure it is the same calibrated one the runner reports on.
        self.env.reward_fn = reward
        self.env.friction_table = reward.friction_table

    def _encode(self, macro: str, timing: str, channel: str | None) -> np.ndarray:
        spec = self.env.spec
        return np.array(
            [
                MACRO_ACTIONS.index(macro),
                spec.time_buckets.index(timing),
                spec.channels.index(channel) if channel else 0,
            ]
        )

    def run_episode(self, policy: Policy, *, seed: int) -> EpisodeResult:
        env, spec = self.env, self.env.spec
        obs, info = env.reset(seed=seed)
        policy.reset_episode()

        episode = env.episode
        total_reward = 0.0
        interventions = 0
        high_risk_attempts = 0
        attempts = 0
        steps: list[dict[str, Any]] = []
        recovered = abandoned = False
        step_idx = 0
        done = False

        while not done:
            decision = policy.act(obs, info)
            proposed = env.decode_action(decision.action)

            state = GuardState(
                decline_code=episode.decline_code,
                attempts_total=attempts,
                attempts_by_channel=dict(env._retries_by_channel),  # noqa: SLF001
                current_channel=env._current_channel,  # noqa: SLF001
            )
            verdict = self.guard.review(proposed, state)
            macro, timing, channel = verdict.final
            if verdict.intervened:
                interventions += 1

            execution = None
            if macro != "ABANDON":
                if episode.decline_code in self.high_risk_codes:
                    high_risk_attempts += 1
                execution = self.executor.execute(
                    RetryRequest(
                        episode_id=episode.episode_id,
                        amount_inr=episode.amount,
                        channel=channel or env._current_channel,  # noqa: SLF001
                        scheduled_in_hours=spec.time_bucket_hours[timing],
                        decline_code=episode.decline_code,
                        attempt_number=attempts + 1,
                    )
                ).__dict__

            obs, reward, terminated, truncated, info = env.step(
                self._encode(macro, timing, channel)
            )
            total_reward += reward
            done = terminated or truncated

            if macro != "ABANDON":
                attempts += 1
            if info["outcome"] == "RECOVERED":
                recovered = True
            if info["outcome"] == "ABANDONED":
                abandoned = True

            record = AuditRecord(
                run_id=self.run_id,
                policy=policy.name,
                episode_id=episode.episode_id,
                step=step_idx,
                decline_code=episode.decline_code,
                amount_inr=round(episode.amount, 2),
                current_channel=state.current_channel,
                elapsed_hours=info["elapsed_hours"],
                attempts_total=attempts,
                friction_penalty=self.reward.friction_table.penalty_for(episode.decline_code),
                proposed={"macro": proposed[0], "timing": proposed[1], "channel": proposed[2]},
                policy_diagnostics=decision.diagnostics or None,
                guard=verdict.to_dict(),
                final={"macro": macro, "timing": timing, "channel": channel},
                execution=execution,
                outcome=info["outcome"],
                reward=round(reward, 6),
                reward_breakdown=info.get("reward_breakdown", {}),
                p_success_ORACLE=info.get("p_success_ORACLE"),
                is_terminal_ORACLE=episode.is_terminal,
            )
            if self.audit is not None:
                self.audit.log(record)
            steps.append(
                {
                    "step": step_idx,
                    "proposed": record.proposed,
                    "final": record.final,
                    "guard": record.guard,
                    "outcome": record.outcome,
                    "reward": record.reward,
                    "p_success_ORACLE": record.p_success_ORACLE,
                }
            )
            step_idx += 1

        wasted = attempts if not recovered else max(attempts - 1, 0)

        return EpisodeResult(
            episode_id=episode.episode_id,
            decline_code=episode.decline_code,
            amount_inr=episode.amount,
            recovered=recovered,
            abandoned=abandoned,
            is_terminal_ORACLE=episode.is_terminal,
            attempts=attempts,
            wasted_attempts=wasted,
            elapsed_hours=info["elapsed_hours"],
            total_reward=total_reward,
            guard_interventions=interventions,
            high_risk_attempts=high_risk_attempts,
            steps=steps,
        )

    def run_batch(self, policy: Policy, *, n_episodes: int, seed: int = 0) -> list[EpisodeResult]:
        return [self.run_episode(policy, seed=seed + i) for i in range(n_episodes)]
