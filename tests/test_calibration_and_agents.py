"""Tests for the LLM calibration table and the learned agent.

The calibration tests do not call an LLM. They assert the properties the table
must have *whatever* produced it — which is the useful thing, because the whole
risk of an LLM-authored reward component is that nobody checks its output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hrapay.agents.flat_dqn import FlatDQNPolicy, FlatQNetwork, enumerate_flat_actions
from hrapay.env.spec import MACRO_ACTIONS, EnvSpec
from hrapay.rewards.calibrator import (
    DETERMINISTIC_TABLE,
    build_table,
    build_user_prompt,
    deterministic_entries,
)
from hrapay.rewards.friction_table import CalibratedFrictionTable, UnreviewedTableError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spec() -> EnvSpec:
    return EnvSpec.load(ROOT / "configs" / "spec_train.yaml")


@pytest.fixture(scope="module")
def table(spec: EnvSpec) -> dict:
    return build_table(spec, deterministic_entries(spec), model="test_fixture", fingerprint="test")


# --- the prompt must not leak the answer -----------------------------------


def test_prompt_contains_no_success_probabilities(spec: EnvSpec) -> None:
    """The single most important test in this file.

    If the numeric success probabilities reached the prompt, the 'calibration'
    would just be laundering the ground truth into the reward, and any result
    derived from it would be meaningless.
    """
    prompt = build_user_prompt(spec)

    for code, dc in spec.decline_codes.items():
        for channel, p in dc.base_success.items():
            assert f"{p}" not in prompt, f"leaked base_success[{code}][{channel}]"
        for bucket, m in dc.time_multiplier.items():
            if m not in (0.0, 1.0):  # 1.0 is too common a string to test on
                assert f"{m}" not in prompt, f"leaked time_multiplier[{code}][{bucket}]"
        assert str(dc.terminal_prob) not in prompt, f"leaked terminal_prob[{code}]"


def test_prompt_contains_every_decline_reason(spec: EnvSpec) -> None:
    prompt = build_user_prompt(spec)
    for code in spec.decline_code_names:
        assert code in prompt


# --- table integrity -------------------------------------------------------


def test_every_decline_code_is_calibrated(spec: EnvSpec, table: dict) -> None:
    assert set(table["entries"]) == set(spec.decline_code_names)


def test_penalties_are_in_range(table: dict) -> None:
    for code, e in table["entries"].items():
        assert 0.0 <= e["friction_penalty"] <= 10.0, code


def test_fraud_and_closed_accounts_score_severe(table: dict) -> None:
    """Policy encoded as a test.

    A calibration pass that scored suspected_fraud as benign would be wrong no
    matter how fluent its justification, so the threshold is asserted rather
    than trusted.
    """
    for code in ("suspected_fraud", "account_closed"):
        assert table["entries"][code]["friction_penalty"] >= 8.0, code
        assert table["entries"][code]["retry_advisable"] is False, code


def test_routine_declines_are_not_over_penalised(table: dict) -> None:
    for code in ("insufficient_funds", "issuer_unavailable"):
        assert table["entries"][code]["friction_penalty"] <= 4.0, code
        assert table["entries"][code]["retry_advisable"] is True, code


def test_every_entry_has_a_reviewable_justification(table: dict) -> None:
    for code, e in table["entries"].items():
        assert len(e["justification"]) > 40, f"{code}: justification too thin to review"


def test_raw_model_output_is_preserved_for_diffing(table: dict) -> None:
    """The human_override vs llm_raw diff is the gating evidence."""
    for code, e in table["entries"].items():
        assert "llm_raw" in e, code
        assert "human_override" in e, code


def test_table_is_generated_unreviewed(table: dict) -> None:
    assert table["review"]["reviewed"] is False


def test_missing_code_is_rejected_loudly(spec: EnvSpec) -> None:
    partial = deterministic_entries(spec)[:-1]
    with pytest.raises(SystemExit, match="no entry for"):
        build_table(spec, partial, model="test", fingerprint="test")


# --- the review gate -------------------------------------------------------


def test_unreviewed_table_refuses_to_load(table: dict) -> None:
    with pytest.raises(UnreviewedTableError):
        CalibratedFrictionTable(table)


def test_reviewed_table_loads(table: dict) -> None:
    reviewed = json.loads(json.dumps(table))
    reviewed["review"]["reviewed"] = True
    ft = CalibratedFrictionTable(reviewed)
    assert ft.penalty_for("suspected_fraud") >= 8.0


def test_human_override_wins_and_is_reported(table: dict) -> None:
    reviewed = json.loads(json.dumps(table))
    reviewed["review"]["reviewed"] = True
    reviewed["entries"]["do_not_honor"]["human_override"] = {
        "friction_penalty": 7.0,
        "reason": "test override",
    }
    reviewed["entries"]["do_not_honor"]["friction_penalty"] = 7.0

    ft = CalibratedFrictionTable(reviewed)
    assert ft.penalty_for("do_not_honor") == 7.0
    assert "do_not_honor" in ft.overrides()


def test_uncalibrated_code_defaults_to_high_friction(table: dict) -> None:
    """Distribution shift on Day 7 introduces a code this table has never seen.

    The safe default when nobody has assessed a decline reason is to treat
    retrying it as expensive, not free.
    """
    reviewed = json.loads(json.dumps(table))
    reviewed["review"]["reviewed"] = True
    ft = CalibratedFrictionTable(reviewed)

    assert ft.penalty_for("some_code_invented_in_2027") >= 5.0
    assert ft.explain("some_code_invented_in_2027")["uncalibrated"] is True


def test_deterministic_fallback_covers_the_whole_spec(spec: EnvSpec) -> None:
    """The no-API-key path must not silently omit a decline code."""
    assert set(DETERMINISTIC_TABLE) >= set(spec.decline_code_names)


# --- flat agent ------------------------------------------------------------


def test_flat_action_space_has_no_duplicates(spec: EnvSpec) -> None:
    """Duplicate encodings of one behaviour split its Q-value across outputs."""
    actions = enumerate_flat_actions(spec)
    assert len(actions) == len(set(actions))


def test_flat_action_count_matches_the_growth_argument(spec: EnvSpec) -> None:
    """1 abandon + 5 retry timings + (5 timings x 5 channels) = 31.

    Against the branched agent's 3 + 5 + 5 = 13 outputs. This is the concrete
    version of the additive-vs-multiplicative claim in the README.
    """
    n_t, n_c = len(spec.time_buckets), len(spec.channels)
    assert len(enumerate_flat_actions(spec)) == 1 + n_t + n_t * n_c == 31

    branched_outputs = len(MACRO_ACTIONS) + n_t + n_c
    assert branched_outputs == 13
    assert branched_outputs < len(enumerate_flat_actions(spec))


def test_flat_policy_emits_valid_actions_and_diagnostics(spec: EnvSpec) -> None:
    net = FlatQNetwork(36, len(enumerate_flat_actions(spec)), 32)
    policy = FlatDQNPolicy(spec, net)
    obs = np.zeros(36, dtype=np.float32)

    decision = policy.act(obs, {})
    assert decision.action.shape == (3,)
    assert 0 <= decision.action[0] < len(MACRO_ACTIONS)
    assert 0 <= decision.action[1] < len(spec.time_buckets)
    assert 0 <= decision.action[2] < len(spec.channels)

    # Diagnostics must be rich enough for the audit trail to explain the choice.
    assert "q_chosen" in decision.diagnostics
    assert len(decision.diagnostics["top_actions"]) == 4


def test_flat_policy_is_deterministic_at_evaluation(spec: EnvSpec) -> None:
    net = FlatQNetwork(36, len(enumerate_flat_actions(spec)), 32).eval()
    policy = FlatDQNPolicy(spec, net)
    obs = np.random.default_rng(0).random(36).astype(np.float32)
    a = policy.act(obs, {}).action
    b = policy.act(obs, {}).action
    assert np.array_equal(a, b)


# --- branched agent --------------------------------------------------------


def test_bdq_output_count_is_additive_not_multiplicative(spec: EnvSpec) -> None:
    """The architectural claim, asserted rather than described.

    13 branched outputs against 31 flat ones on this problem -- and the gap
    widens with every action dimension added.
    """
    from hrapay.agents.bdq import BranchingQNetwork

    branch_sizes = [len(MACRO_ACTIONS), len(spec.time_buckets), len(spec.channels)]
    net = BranchingQNetwork(36, branch_sizes, 32)

    assert net.n_outputs == sum(branch_sizes) == 13
    assert net.n_outputs < len(enumerate_flat_actions(spec))


def test_bdq_emits_one_q_vector_per_branch(spec: EnvSpec) -> None:
    from hrapay.agents.bdq import BranchingQNetwork

    branch_sizes = [len(MACRO_ACTIONS), len(spec.time_buckets), len(spec.channels)]
    net = BranchingQNetwork(36, branch_sizes, 32)
    out = net(torch.zeros(4, 36))

    assert len(out) == 3
    for q, n in zip(out, branch_sizes, strict=True):
        assert q.shape == (4, n)


def test_bdq_policy_returns_per_branch_diagnostics(spec: EnvSpec) -> None:
    """The readable advantage of branching: the audit log can separate the
    decision to retry from when, and from on which rail."""
    from hrapay.agents.bdq import BDQPolicy, BranchingQNetwork

    branch_sizes = [len(MACRO_ACTIONS), len(spec.time_buckets), len(spec.channels)]
    policy = BDQPolicy(spec, BranchingQNetwork(36, branch_sizes, 32).eval())
    decision = policy.act(np.zeros(36, dtype=np.float32), {})

    assert decision.action.shape == (3,)
    for name, size in zip(("macro", "timing", "channel"), branch_sizes, strict=True):
        assert decision.diagnostics[name]["chosen"]
        assert len(decision.diagnostics[name]["all"]) == size


def test_branch_mask_silences_branches_that_had_no_effect(spec: EnvSpec) -> None:
    """Without this, the timing head trains on the reward from every ABANDON --
    a gradient for a choice that changed nothing."""
    from hrapay.agents.bdq import ABANDON_IDX, SWITCH_IDX, branch_activity_mask

    retry = MACRO_ACTIONS.index("RETRY")
    mask = branch_activity_mask(np.array([ABANDON_IDX, retry, SWITCH_IDX]))

    assert mask.tolist() == [
        [1.0, 0.0, 0.0],  # abandon: only the macro choice mattered
        [1.0, 1.0, 0.0],  # retry: macro and timing, channel is implicit
        [1.0, 1.0, 1.0],  # switch: all three
    ]


def test_both_agents_explore_the_same_distribution() -> None:
    """The fairness property the whole flat-vs-branched comparison rests on.

    Uniform over the flat agent's 31 actions gives P(ABANDON) = 3.2%; uniform
    over three branches gives 33%. If the two agents used their own natural
    samplers, the comparison would measure exploration, not architecture.
    """
    from hrapay.train import canonicalise_for_flat, sample_random_branched_action

    rng = np.random.default_rng(0)
    n = 20_000
    branched = [sample_random_branched_action(rng, [3, 5, 5]) for _ in range(n)]

    macro_share = sum(int(a[0]) == MACRO_ACTIONS.index("ABANDON") for a in branched) / n
    flat_share = (
        sum(canonicalise_for_flat(a)[0] == MACRO_ACTIONS.index("ABANDON") for a in branched) / n
    )
    assert abs(macro_share - flat_share) < 1e-9
    assert 0.30 < flat_share < 0.36


def test_canonicalise_drops_branches_the_env_ignores() -> None:
    from hrapay.train import canonicalise_for_flat

    abandon = MACRO_ACTIONS.index("ABANDON")
    retry = MACRO_ACTIONS.index("RETRY")
    switch = MACRO_ACTIONS.index("SWITCH_CHANNEL")

    assert canonicalise_for_flat(np.array([abandon, 3, 4])) == (abandon, 0, 0)
    assert canonicalise_for_flat(np.array([retry, 3, 4])) == (retry, 3, 0)
    assert canonicalise_for_flat(np.array([switch, 3, 4])) == (switch, 3, 4)
