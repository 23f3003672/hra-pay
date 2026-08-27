"""Audit trail.

One JSON object per decision, appended to a JSONL file. Greppable, diffable,
no database, and readable by a human or by pandas without any tooling.

The record deliberately captures the *proposed* action alongside the *final*
one. An audit trail that only records what happened cannot answer the question
a reviewer actually cares about — what did the model want to do, and what
stopped it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "audit/v1"


@dataclass
class AuditRecord:
    """A single decision, fully reconstructible."""

    schema_version: str = SCHEMA_VERSION

    run_id: str = ""
    policy: str = ""
    episode_id: str = ""
    step: int = 0

    # Situation at decision time
    decline_code: str = ""
    amount_inr: float = 0.0
    current_channel: str = ""
    elapsed_hours: float = 0.0
    attempts_total: int = 0
    friction_penalty: float = 0.0

    # What the policy wanted, why, and what it was allowed to do
    proposed: dict[str, Any] = field(default_factory=dict)
    policy_diagnostics: dict[str, Any] | None = None
    guard: dict[str, Any] = field(default_factory=dict)
    final: dict[str, Any] = field(default_factory=dict)

    # Consequence
    execution: dict[str, Any] | None = None
    outcome: str = ""
    reward: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)

    # Oracle fields — ground truth used for evaluation only. Suffixed so that
    # nobody can mistake them for something the agent had access to.
    p_success_ORACLE: float | None = None
    is_terminal_ORACLE: bool | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class AuditLogger:
    """Appends AuditRecords to a JSONL file.

    Also keeps them in memory so the dashboard can replay an episode without
    re-reading from disk.
    """

    def __init__(self, path: str | Path | None = None, *, keep_in_memory: bool = True) -> None:
        self.path = Path(path) if path else None
        self.keep_in_memory = keep_in_memory
        self.records: list[AuditRecord] = []
        self._fh = None

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")

    def log(self, record: AuditRecord) -> None:
        if self.keep_in_memory:
            self.records.append(record)
        if self._fh is not None:
            self._fh.write(record.to_json() + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reading back -------------------------------------------------------

    @staticmethod
    def read(path: str | Path) -> list[dict[str, Any]]:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def interventions(self) -> list[AuditRecord]:
        """Every decision the guard overrode. The 'bounded and gated' evidence."""
        return [r for r in self.records if r.guard.get("intervened")]
