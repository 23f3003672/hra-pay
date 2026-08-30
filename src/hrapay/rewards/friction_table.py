"""Loads the calibrated penalty table and serves it to the reward function.

Refuses to load a table that has not been human-reviewed. That refusal is the
whole mechanism behind the "bounded and gated" claim: an LLM-authored number
cannot silently reach the reward function that trains a policy which moves
money. Someone has to have read it and said so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TABLE = Path(__file__).resolve().parent / "penalty_table.json"

MIN_PENALTY = 0.0
MAX_PENALTY = 10.0


class UnreviewedTableError(RuntimeError):
    """Raised when a calibration table has not been signed off by a human."""


class CalibratedFrictionTable:
    """Maps a decline code to its friction penalty, plus the reason for it."""

    def __init__(self, data: dict[str, Any], *, require_review: bool = True) -> None:
        self.data = data
        self.entries: dict[str, dict[str, Any]] = data["entries"]

        if require_review and not data.get("review", {}).get("reviewed", False):
            raise UnreviewedTableError(
                "penalty_table.json has not been reviewed.\n"
                "Read every justification, correct anything wrong via human_override, "
                "then set review.reviewed = true and review.reviewed_by to your name.\n"
                "An LLM-authored penalty must not reach the reward function unread."
            )

        for code, e in self.entries.items():
            p = e["friction_penalty"]
            if not MIN_PENALTY <= p <= MAX_PENALTY:
                raise ValueError(f"{code}: friction_penalty {p} outside [0, 10]")

    # -- construction -------------------------------------------------------

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_TABLE, *, require_review: bool = True
    ) -> CalibratedFrictionTable:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"No calibration table at {p}.\n"
                "Generate one with:  python -m hrapay.rewards.calibrator\n"
                "Or without an API key:  python -m hrapay.rewards.calibrator --deterministic"
            )
        return cls(json.loads(p.read_text(encoding="utf-8")), require_review=require_review)

    # -- the interface the reward function uses -----------------------------

    def penalty_for(self, decline_code: str) -> float:
        entry = self.entries.get(decline_code)
        if entry is None:
            # An unseen decline code is treated as high friction, not free.
            # Optimism about a reason nobody has calibrated is the wrong default
            # when the downside is spending money on a doomed retry. This path
            # is exercised by the held-out environment, which introduces a
            # decline code the table has never seen.
            return 7.0
        return float(entry["friction_penalty"])

    # -- explanation, for the audit trail and dashboard ---------------------

    def explain(self, decline_code: str) -> dict[str, Any]:
        entry = self.entries.get(decline_code)
        if entry is None:
            return {
                "friction_penalty": 7.0,
                "severity": "UNKNOWN",
                "retry_advisable": False,
                "justification": (
                    f"'{decline_code}' was not present when this table was calibrated. "
                    f"Defaulting to high friction rather than assuming it is safe."
                ),
                "uncalibrated": True,
            }
        return {**entry, "uncalibrated": False}

    def overrides(self) -> dict[str, Any]:
        """Every place a human disagreed with the model. The gating evidence."""
        return {
            code: {
                "llm": e["llm_raw"],
                "human": e["human_override"],
                "effective": e["friction_penalty"],
            }
            for code, e in self.entries.items()
            if e.get("human_override") is not None
        }

    @property
    def source(self) -> str:
        return str(self.data.get("source", "unknown"))

    @property
    def reviewed_by(self) -> str | None:
        return self.data.get("review", {}).get("reviewed_by")
