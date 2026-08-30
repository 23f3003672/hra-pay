"""Report how far each policy degrades under distribution shift.

    python -m hrapay.eval.shift

Reads results/summary_train.csv and results/summary_holdout.csv and prints the
drop for each policy, plus the one number that actually settles the "you
designed the world your agent wins in" objection:

    RELATIVE degradation of the learned agents against the static baselines.

Every policy is expected to get worse on the held-out spec — the population is
harder there by construction, so an absolute drop proves nothing. What would be
damning is the learned agents degrading MORE than the static ones, because that
is the signature of having memorised the training distribution rather than
learned a policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"

LEARNED = ("flat_dqn", "bdq")
STATIC = ("static_schedule", "static_with_switch")


def load(tag: str) -> pd.DataFrame:
    path = RESULTS / f"summary_{tag}.csv"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun:\n"
            "  python -m hrapay.eval.cli --tag train\n"
            "  python -m hrapay.eval.cli --spec configs/spec_holdout.yaml --tag holdout"
        )
    return pd.read_csv(path).set_index("family")


def main() -> None:
    ap = argparse.ArgumentParser(description="Distribution-shift degradation report.")
    ap.add_argument("--metric", default="recovery_rate")
    args = ap.parse_args()

    train, holdout = load("train"), load("holdout")
    col = f"{args.metric}_mean"
    families = [f for f in train.index if f in holdout.index]

    rows = []
    for family in families:
        before, after = float(train.loc[family, col]), float(holdout.loc[family, col])
        rows.append(
            {
                "policy": family,
                "train": round(before, 4),
                "holdout": round(after, 4),
                "abs_drop": round(before - after, 4),
                "rel_drop_pct": round(100.0 * (before - after) / before, 1) if before else 0.0,
            }
        )

    table = pd.DataFrame(rows).set_index("policy")
    print(f"metric: {args.metric}\n")
    print(table.to_string())

    learned = [r for r in rows if r["policy"] in LEARNED]
    static = [r for r in rows if r["policy"] in STATIC]
    if not learned or not static:
        return

    learned_drop = sum(r["rel_drop_pct"] for r in learned) / len(learned)
    static_drop = sum(r["rel_drop_pct"] for r in static) / len(static)

    print(f"\nmean relative drop:  learned {learned_drop:.1f}%   static {static_drop:.1f}%")
    if learned_drop > static_drop + 5:
        verdict = (
            "The learned agents degrade materially MORE than the static baselines. "
            "That is evidence of overfitting to the training distribution and must "
            "be reported as such."
        )
    elif learned_drop > static_drop:
        verdict = (
            "The learned agents degrade slightly more than the static baselines, "
            "within the range a fixed schedule would be expected to enjoy from "
            "having nothing to overfit."
        )
    else:
        verdict = (
            "The learned agents degrade no more than the static baselines, so the "
            "advantage is not an artefact of memorising the training spec."
        )
    print(f"\n{verdict}")

    # Does the ranking survive? A policy that wins on the training distribution
    # and loses on the shifted one has not learned anything transferable.
    best_train = max(rows, key=lambda r: r["train"])["policy"]
    best_holdout = max(rows, key=lambda r: r["holdout"])["policy"]
    print(f"\nbest on train: {best_train}   best on holdout: {best_holdout}", end="")
    print("   (ranking holds)" if best_train == best_holdout else "   (RANKING CHANGED)")


if __name__ == "__main__":
    main()
