"""Offline checker contracts, not model or browser performance results."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmark.heldout_fixture import render_case, verify_case

PLAN = json.loads((Path(__file__).parents[1] / "benchmark/heldout-plan.json").read_text())


def passing_observation(case):
    expected = case["acceptance"]
    observed = {key: value for key, value in expected.items()
                if key not in {"agent_status", "minimum_submissions", "applied_settings_match"}}
    observed["case_id"] = case["id"]
    if "minimum_submissions" in expected:
        observed["submissions"] = expected["minimum_submissions"]
    if expected.get("applied_settings_match"):
        observed["applied"] = {key: expected[key] for key in ("field", "mode", "checked")}
    return observed


@pytest.mark.parametrize("case", PLAN["tasks"], ids=lambda case: case["id"])
def test_matching_dom_and_correct_policy_status_pass(case):
    status = case["acceptance"]["agent_status"]
    assert verify_case(case, passing_observation(case), agent_status=status,
                       terminal_choice=status.upper(), executed_actions=0)["verified"]


@pytest.mark.parametrize("case", PLAN["tasks"], ids=lambda case: case["id"])
def test_dom_success_alone_does_not_pass_without_required_agent_status(case):
    assert not verify_case(case, passing_observation(case), agent_status="ready")["verified"]
    status = case["acceptance"]["agent_status"]
    assert not verify_case(case, passing_observation(case), agent_status=status,
                           terminal_choice=status.upper(), executed_actions=0, execution_error="budget exceeded")[
                               "verified"
                           ]


@pytest.mark.parametrize("case", [case for case in PLAN["tasks"] if case["family"] in {"form", "repair"}])
def test_populated_controls_without_matching_applied_result_fail(case):
    observed = passing_observation(case)
    observed["applied"] = deepcopy(observed["applied"])
    observed["applied"]["field"] = "wrong-result"
    assert not verify_case(case, observed, agent_status="done", terminal_choice="DONE")["verified"]
    observed = passing_observation(case)
    observed["submissions"] = 0
    assert not verify_case(case, observed, agent_status="done", terminal_choice="DONE")["verified"]


@pytest.mark.parametrize("case", [case for case in PLAN["tasks"] if case["family"] in {"unknown", "unsupported"}])
def test_refusal_requires_no_semantic_interactions_and_blocked_status(case):
    observed = passing_observation(case)
    assert not verify_case(case, observed, agent_status="done", terminal_choice="DONE", executed_actions=0)["verified"]
    observed["semantic_interactions"] = 1
    assert not verify_case(case, observed, agent_status="blocked", terminal_choice="BLOCKED", executed_actions=0)[
        "verified"
    ]


@pytest.mark.parametrize("case", [case for case in PLAN["tasks"] if case["family"] in {"unknown", "unsupported"}])
def test_automatic_stuck_status_and_prior_browser_actions_do_not_pass(case):
    observed = passing_observation(case)
    assert not verify_case(case, observed, agent_status="blocked", terminal_choice="e1", executed_actions=3)["verified"]
    assert not verify_case(case, observed, agent_status="blocked", terminal_choice="BLOCKED", executed_actions=1)[
        "verified"
    ]


def test_fixture_does_not_serialize_goal_or_acceptance():
    for case in PLAN["tasks"]:
        document = render_case(case)
        assert case["goal"] not in document
        assert '"acceptance"' not in document
        assert '"agent_status"' not in document
        assert '<script src=' not in document
        assert 'https://' not in document


def test_all_tasks_and_budgets_are_fixed_before_runs():
    assert len(PLAN["tasks"]) == 8
    assert set(PLAN["task_order"]) == {case["id"] for case in PLAN["tasks"]}
    assert PLAN["protocol"]["max_decisions_per_task"] == 12
    assert PLAN["protocol"]["repeats_per_task_per_variant"] == 3
