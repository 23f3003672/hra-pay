"""HRA-Pay dashboard.

    streamlit run app/dashboard.py

Four views, in the order a reviewer would want them:

    Episode Explorer   one transaction, decision by decision, with the agent's
                       Q-values, the guardrail verdict, and the reward breakdown
    Results            train vs held-out, with the seed spread visible
    Reward Calibration the LLM's scores and every place a human overrode them
    Audit Trail        the raw decision log

The Episode Explorer is the point of the whole thing. A results table says the
system recovers money; only a step-by-step trace shows *why it did what it did*
and *what stopped it* — which is what "explainable, bounded and gated" has to
mean if it means anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrapay.agents.bdq import load_bdq_policy  # noqa: E402
from hrapay.agents.flat_dqn import load_flat_policy  # noqa: E402
from hrapay.agents.static import (  # noqa: E402
    StaticSchedulePolicy,
    StaticWithChannelSwitchPolicy,
)
from hrapay.audit.logger import AuditLogger  # noqa: E402
from hrapay.env.demo import load_priors  # noqa: E402
from hrapay.env.retry_env import RetryEnv  # noqa: E402
from hrapay.env.spec import EnvSpec  # noqa: E402
from hrapay.eval.runner import EpisodeRunner  # noqa: E402
from hrapay.guard.policy_guard import GuardConfig, PolicyGuard  # noqa: E402
from hrapay.rewards.friction_table import CalibratedFrictionTable  # noqa: E402
from hrapay.rewards.reward import CalibratedReward, RewardConfig  # noqa: E402

CONFIGS = ROOT / "configs"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"

# Categorical slots 1 and 2 from the validated palette. Two series only, so the
# adjacent-pair checks apply; both carry direct value labels because slot
# contrast against a light surface sits below 3:1.
C_TRAIN = "#2a78d6"
C_HOLDOUT = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"

st.set_page_config(page_title="HRA-Pay", page_icon="₹", layout="wide")


# --- loading ---------------------------------------------------------------


@st.cache_resource
def load_specs() -> dict[str, EnvSpec]:
    return {
        "train": EnvSpec.load(CONFIGS / "spec_train.yaml"),
        "holdout": EnvSpec.load(CONFIGS / "spec_holdout.yaml"),
    }


@st.cache_resource
def load_pieces() -> tuple[RewardConfig, GuardConfig, CalibratedFrictionTable, dict]:
    train = load_specs()["train"]
    return (
        RewardConfig.load(CONFIGS / "default.yaml"),
        GuardConfig.load(CONFIGS / "default.yaml"),
        CalibratedFrictionTable.load(),
        load_priors(train),
    )


@st.cache_resource
def load_policies() -> dict[str, object]:
    train = load_specs()["train"]
    out: dict[str, object] = {
        "static_schedule": StaticSchedulePolicy(train),
        "static_with_switch": StaticWithChannelSwitchPolicy(train),
    }
    for ckpt in sorted(CHECKPOINTS.glob("flat_seed*.pt")):
        out[f"flat_dqn (seed {ckpt.stem.split('seed')[-1]})"] = load_flat_policy(train, ckpt)
    for ckpt in sorted(CHECKPOINTS.glob("bdq_seed*.pt")):
        out[f"bdq (seed {ckpt.stem.split('seed')[-1]})"] = load_bdq_policy(train, ckpt)
    return out


def run_one_episode(policy, spec_name: str, seed: int):
    specs = load_specs()
    spec = specs[spec_name]
    train = specs["train"]
    reward_cfg, guard_cfg, friction, priors = load_pieces()

    reward = CalibratedReward(reward_cfg, friction_table=friction)
    env = RetryEnv(
        spec,
        seed=seed,
        channel_priors=priors,
        reward_fn=reward,
        observation_codes=train.decline_code_names,
    )
    guard = PolicyGuard(guard_cfg, timing_order=spec.time_buckets)
    audit = AuditLogger(None)
    runner = EpisodeRunner(env, guard, reward, audit=audit, run_id="dashboard")
    result = runner.run_episode(policy, seed=seed)
    return result, audit.records, friction


# --- shared bits -----------------------------------------------------------


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items, strict=True):
        col.metric(label, value, help=help_text)


def grouped_bar(df: pd.DataFrame, metric: str, title: str, fmt: str) -> go.Figure:
    """Train vs held-out, one group per policy, with direct value labels.

    Direct labels are not decoration: the palette check flags these slots as
    below 3:1 against a light surface, and the relief rule requires visible
    labels or a table view. This ships both.
    """
    fig = go.Figure()
    for name, colour, col in (
        ("Training", C_TRAIN, f"{metric}_train"),
        ("Held-out", C_HOLDOUT, f"{metric}_holdout"),
    ):
        fig.add_bar(
            name=name,
            x=df["policy"],
            y=df[col],
            marker_color=colour,
            marker_line_width=2,
            marker_line_color="#fcfcfb",  # 2px surface gap between adjacent fills
            text=[fmt.format(v) for v in df[col]],
            textposition="outside",
            textfont={"color": INK, "size": 12},
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:,.3f}}<extra></extra>",
        )
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": INK}},
        barmode="group",
        bargap=0.3,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": INK_MUTED, "size": 12},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        height=380,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    return fig


# --- tab 1: episode explorer ----------------------------------------------


def tab_episode() -> None:
    st.subheader("Episode Explorer")
    st.caption(
        "One failed transaction, decision by decision. Everything the agent saw, "
        "everything it wanted, and everything the guardrails allowed."
    )

    policies = load_policies()
    left, mid, right = st.columns([2, 2, 1])
    policy_name = left.selectbox("Policy", list(policies))
    spec_name = mid.selectbox(
        "Distribution", ["train", "holdout"], help="'holdout' is the shifted spec no agent trained on"
    )
    seed = right.number_input("Episode seed", min_value=0, max_value=99_999, value=7, step=1)

    result, records, friction = run_one_episode(policies[policy_name], spec_name, int(seed))
    entry = friction.explain(result.decline_code)

    st.divider()
    metric_row(
        [
            ("Transaction", f"Rs {result.amount_inr:,.0f}", None),
            ("Decline reason", result.decline_code, entry["justification"]),
            ("Friction penalty", f"{entry['friction_penalty']:.1f} / 10", "From the calibrated table"),
            ("Outcome", "RECOVERED" if result.recovered else ("ABANDONED" if result.abandoned else "EXHAUSTED"), None),
            ("Attempts", str(result.attempts), None),
            ("Elapsed", f"{result.elapsed_hours:.0f}h", None),
        ]
    )

    if entry.get("uncalibrated"):
        st.warning(
            f"**{result.decline_code}** was never calibrated — it did not exist when this "
            f"model was trained. It reads as an all-zero one-hot, has no channel history, "
            f"and falls through to the default high friction penalty. This is the realistic "
            f"deployment failure the held-out spec is designed to produce."
        )

    st.divider()
    for i, rec in enumerate(records, start=1):
        guard = rec.guard
        blocked = guard["intervened"]
        header = (
            f"Step {i} — proposed {rec.proposed['macro']} {rec.proposed['timing']}"
            f"{' -> ' + rec.proposed['channel'] if rec.proposed['channel'] else ''}"
            f"{'   [GUARD OVERRODE]' if blocked else ''}"
        )
        with st.expander(header, expanded=(i == 1 or blocked)):
            a, b = st.columns(2)

            with a:
                st.markdown("**What the policy wanted**")
                diag = rec.policy_diagnostics or {}
                if "macro" in diag and isinstance(diag.get("macro"), dict):
                    for branch in ("macro", "timing", "channel"):
                        st.caption(f"{branch} branch — chose **{diag[branch]['chosen']}**")
                        st.dataframe(
                            pd.DataFrame(
                                {"action": list(diag[branch]["all"]), "Q": list(diag[branch]["all"].values())}
                            ),
                            hide_index=True,
                            width="stretch",
                        )
                elif "top_actions" in diag:
                    st.caption(f"chose **{diag['chosen']}** of {diag['n_flat_actions']} flat actions")
                    st.dataframe(pd.DataFrame(diag["top_actions"]), hide_index=True, width="stretch")
                else:
                    st.caption(f"rule-based: {diag.get('rule', 'n/a')}")

            with b:
                st.markdown("**What the guardrails decided**")
                if blocked:
                    st.error(
                        f"**{guard['rule']}** ({guard['rule_class']})\n\n{guard['reason']}\n\n"
                        f"Rewritten to **{guard['final']['macro']}**."
                    )
                else:
                    st.success("No rule fired. The proposed action was permitted unchanged.")

                if rec.execution:
                    st.markdown("**Executor**")
                    st.code(
                        f"{rec.execution['executor']}  ref={rec.execution['reference_id']}\n"
                        f"channel={rec.execution['detail'].get('channel')}  "
                        f"in {rec.execution['detail'].get('scheduled_in_hours')}h",
                        language="text",
                    )

                if rec.reward_breakdown:
                    st.markdown("**Reward**")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "term": list(rec.reward_breakdown),
                                "value": [round(v, 4) for v in rec.reward_breakdown.values()],
                            }
                        ),
                        hide_index=True,
                        width="stretch",
                    )

            with st.expander("Oracle (ground truth — NOT visible to the agent)"):
                st.caption(
                    "Shown for analysis only. The agent never receives these; "
                    "`is_terminal` in particular is what makes ABANDON a real inference."
                )
                st.json(
                    {
                        "p_success": rec.p_success_ORACLE,
                        "is_terminal": rec.is_terminal_ORACLE,
                        "outcome": rec.outcome,
                    }
                )


# --- tab 2: results --------------------------------------------------------


def tab_results() -> None:
    st.subheader("Results")
    train_p, hold_p = RESULTS / "summary_train.csv", RESULTS / "summary_holdout.csv"
    if not (train_p.exists() and hold_p.exists()):
        st.info("Run `python tasks.py eval` to generate the results tables.")
        return

    train = pd.read_csv(train_p).set_index("family")
    hold = pd.read_csv(hold_p).set_index("family")
    families = [f for f in train.index if f in hold.index]

    rows = []
    for f in families:
        rows.append(
            {
                "policy": f,
                "seeds": int(train.loc[f, "train_seeds"]),
                "recovery_rate_train": float(train.loc[f, "recovery_rate_mean"]),
                "recovery_rate_holdout": float(hold.loc[f, "recovery_rate_mean"]),
                "recovery_rate_train_std": float(train.loc[f, "recovery_rate_std"]),
                "recovered_inr_train": float(train.loc[f, "recovered_inr_mean"]),
                "recovered_inr_holdout": float(hold.loc[f, "recovered_inr_mean"]),
                "wasted_attempts_train": float(train.loc[f, "wasted_attempts_mean"]),
                "wasted_attempts_holdout": float(hold.loc[f, "wasted_attempts_mean"]),
                "issuer_risk": float(train.loc[f, "issuer_risk_exposure_mean"]),
            }
        )
    df = pd.DataFrame(rows).sort_values("recovery_rate_holdout", ascending=False)
    df["rel_drop_pct"] = 100 * (df.recovery_rate_train - df.recovery_rate_holdout) / df.recovery_rate_train

    st.plotly_chart(
        grouped_bar(df, "recovery_rate", "Recovery rate — training vs held-out", "{:.3f}"),
        width="stretch",
    )

    st.markdown("**The table the chart is drawn from**")
    st.dataframe(
        df[
            [
                "policy",
                "seeds",
                "recovery_rate_train",
                "recovery_rate_holdout",
                "rel_drop_pct",
                "wasted_attempts_train",
                "wasted_attempts_holdout",
                "issuer_risk",
            ]
        ].round(3),
        hide_index=True,
        width="stretch",
    )

    st.divider()
    st.markdown("#### What survives, and what does not")
    st.markdown(
        """
- **Both learned agents beat both static baselines**, on the training distribution
  and under shift. This is the stable result.
- **On the training distribution the two architectures are indistinguishable** —
  their gap is smaller than the spread across training seeds.
- **Under distribution shift they are not.** The flat agent degrades less and the
  ranking between the two flips. Measuring only on the training distribution
  would have produced the wrong recommendation.
- **Off-distribution, both learned agents spend far more retries** than the static
  baselines. They stay ahead on revenue by spending more attempts, and that gap
  widens under shift.
- **Issuer-risk exposure is zero everywhere.** No policy ever retried a
  fraud-flagged authorisation, because the guard does not let it.
"""
    )


# --- tab 3: calibration ----------------------------------------------------


def tab_calibration() -> None:
    st.subheader("Reward Calibration")
    st.caption(
        "An LLM read each raw decline-reason string — and nothing else — and scored how "
        "much friction a retry carries. Then a human reviewed it. Both are on the record."
    )

    path = ROOT / "src" / "hrapay" / "rewards" / "penalty_table.json"
    if not path.exists():
        st.info("Run `python tasks.py calibrate` first.")
        return
    table = json.loads(path.read_text(encoding="utf-8"))

    metric_row(
        [
            ("Model", table["source"], None),
            ("Reviewed by", table["review"].get("reviewed_by") or "-", None),
            ("Entries", str(len(table["entries"])), None),
            ("Human overrides", str(sum(1 for e in table["entries"].values() if e["human_override"])), None),
        ]
    )

    st.divider()
    for code, e in table["entries"].items():
        overridden = e["human_override"] is not None
        with st.expander(
            f"{code} — {e['friction_penalty']:.1f}/10"
            + (f"  (LLM said {e['llm_raw']['friction_penalty']:.1f})" if overridden else ""),
            expanded=overridden,
        ):
            st.markdown(f"*{e['justification']}*")
            if overridden:
                st.error(
                    f"**Overridden by a human.** {e['llm_raw']['friction_penalty']:.1f} → "
                    f"{e['friction_penalty']:.1f}\n\n{e['human_override']['reason']}"
                )
            else:
                st.success("Accepted as scored.")

    st.info(
        "The reward function refuses to load this table until `review.reviewed` is true. "
        "An LLM-authored number cannot reach a policy that moves money without a person "
        "having read it."
    )


# --- tab 4: audit ----------------------------------------------------------


def tab_audit() -> None:
    st.subheader("Audit Trail")
    st.caption("One JSON record per decision, capturing what was proposed as well as what happened.")

    files = sorted(RESULTS.glob("audit_*.jsonl"))
    if not files:
        st.info("Run `python tasks.py eval` to generate audit logs.")
        return

    chosen = st.selectbox("Log", [f.name for f in files])
    records = AuditLogger.read(RESULTS / chosen)
    interventions = [r for r in records if r["guard"]["intervened"]]

    metric_row(
        [
            ("Decisions logged", f"{len(records):,}", None),
            ("Guard interventions", f"{len(interventions):,}", None),
            (
                "Intervention rate",
                f"{100 * len(interventions) / max(len(records), 1):.1f}%",
                None,
            ),
        ]
    )

    if interventions:
        by_rule = pd.Series([r["guard"]["rule"] for r in interventions]).value_counts()
        st.markdown("**Which rules fired**")
        st.dataframe(
            pd.DataFrame({"rule": by_rule.index, "times": by_rule.values}),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**A blocked decision, in full**")
        st.json(interventions[0])

    st.markdown("**Raw records**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "episode": r["episode_id"],
                    "step": r["step"],
                    "decline": r["decline_code"],
                    "amount": r["amount_inr"],
                    "proposed": f"{r['proposed']['macro']} {r['proposed']['timing']}",
                    "final": f"{r['final']['macro']} {r['final']['timing']}",
                    "guard": r["guard"]["rule"] or "",
                    "outcome": r["outcome"],
                    "reward": r["reward"],
                }
                for r in records[:500]
            ]
        ),
        hide_index=True,
        width="stretch",
        height=380,
    )


# --- main ------------------------------------------------------------------

st.title("HRA-Pay")
st.caption(
    "Hierarchical reinforcement learning for payment retry optimisation. "
    "Razorpay Buildathon 2026 — Track 03, AI Revenue Recovery."
)

t1, t2, t3, t4 = st.tabs(["Episode Explorer", "Results", "Reward Calibration", "Audit Trail"])
with t1:
    tab_episode()
with t2:
    tab_results()
with t3:
    tab_calibration()
with t4:
    tab_audit()
