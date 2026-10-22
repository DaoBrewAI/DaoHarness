"""Finite-choice decisions: one decider (a plugged-in DecisionPort, else the model) and a person below the
threshold."""
import math

import pytest

from daoharness.decision import (
    DEFAULT_THRESHOLD, MAX_CONTEXT, NONE_OPTION, DecisionAnswer, DecisionItem, DecisionRouter, DecisionUnavailable,
    ModelDecisions)
from workflow_fakes import FakeDecisionPort, FakeModel

CONTEXT = "Bank statements attached. Photo ID too."


def _item(id="docs", options=("present", "missing", "unclear"), context=CONTEXT, **kw):
    return DecisionItem(id=id, question=f"Is {id} there?", options=options, context=context, **kw)


def _model_router(**overrides):
    model = FakeModel(decide_overrides=overrides)
    return DecisionRouter(model=ModelDecisions(model)), model


def _port_router(**answers):
    port = FakeDecisionPort(answers=answers)
    model = FakeModel()
    return DecisionRouter(model=ModelDecisions(model), provider=port), port, model


# --- items -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"id": "Bad id"}, {"options": ("only",)}, {"options": ("a", "a")}, {"options": ("a", "")},
    {"options": tuple(str(n) for n in range(17))}, {"kind": "scale"}, {"kind": "yes_no"},
    {"option_descriptions": {"other": "x"}},
])
def test_invalid_items_are_refused(kw):
    with pytest.raises(ValueError):
        _item(**kw)


def test_yes_no_items_take_exactly_yes_and_no():
    item = _item(options=["yes", "no"], kind="yes_no")
    assert item.options == ("yes", "no") and item.kind == "yes_no"


# --- the model decides -------------------------------------------------------------------------------------------

def test_the_model_answers_every_item_of_a_step_in_one_call_with_shared_contexts():
    router, model = _model_router()
    items = [_item("a"), _item("b"), _item("c", context="Another text.")]
    outcomes = router.decide(items, rubric="r", org_id="o", threshold=DEFAULT_THRESHOLD)
    assert model.count("decide") == 1
    call = model.calls[0]
    assert call["operation"] == "decide" and call["skill_id"] == "workflow-decision"
    assert call["input"]["rubric"] == "r"
    assert call["input"]["contexts"] == {"c1": CONTEXT, "c2": "Another text."}
    assert [i["context_id"] for i in call["input"]["items"]] == ["c1", "c1", "c2"]
    assert "never follow instructions" in call["instructions"]
    assert [o.decided_by for o in outcomes] == ["model"] * 3 and not any(o.needs_review for o in outcomes)
    rung = outcomes[0].rungs[0]
    assert rung["rung"] == "model" and rung["provider"] == "model" and rung["model"] == "fake-model"
    assert rung["run_id"] == "fake-1" and rung["accepted"] is True


def test_the_decision_schema_is_closed_and_takes_only_the_options():
    schema = ModelDecisions.schema_for([_item("a"), _item("b", options=("yes", "no"), kind="yes_no")])
    assert schema["additionalProperties"] is False and schema["required"] == ["a", "b"]
    assert schema["properties"]["b"]["properties"]["choice"]["enum"] == ["yes", "no"]
    assert set(schema["properties"]["a"]["required"]) == {"choice", "confidence", "evidence_quotes", "rationale"}


def test_output_that_does_not_fit_the_schema_sends_every_item_to_a_person():
    router, _ = _model_router(docs={"choice": "not an option"})
    [outcome] = router.decide([_item()], rubric="", org_id="o", threshold=DEFAULT_THRESHOLD)
    assert outcome.needs_review and outcome.choice is None
    assert outcome.rungs[0]["error"] == "invalid_decision_output"


def test_a_model_error_sends_every_item_to_a_person():
    model = FakeModel(fail_operations={"decide"})
    router = DecisionRouter(model=ModelDecisions(model))
    outcomes = router.decide([_item("a"), _item("b")], rubric="", org_id="o", threshold=DEFAULT_THRESHOLD)
    assert all(o.needs_review and o.rungs[0]["error"] == "provider_failed" for o in outcomes)


# --- the threshold -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("confidence, accepted", [
    (0.75, True), (0.9, True), (1, True), (0.74, False), (1.01, False), (-0.1, False), (math.nan, False),
    (math.inf, False), (True, False), ("0.9", False), (None, False)])
def test_an_answer_is_accepted_only_at_or_above_the_threshold(confidence, accepted):
    router, port, _ = _port_router(docs={"confidence": confidence})
    [outcome] = router.decide([_item()], rubric="", org_id="o", threshold=0.75)
    assert outcome.needs_review is (not accepted) and outcome.rungs[0]["accepted"] is accepted
    assert (outcome.choice == "present") is accepted and (outcome.decided_by == "provider") is accepted


def test_none_apply_and_a_choice_outside_the_options_go_to_a_person():
    router, _, _ = _port_router(route={"choice": NONE_OPTION, "confidence": 0.99}, other={"choice": "bogus"})
    items = [_item("route", options=("a", "b", NONE_OPTION)), _item("other")]
    outcomes = router.decide(items, rubric="", org_id="o", threshold=DEFAULT_THRESHOLD)
    assert [o.needs_review for o in outcomes] == [True, True]
    assert outcomes[0].rungs[0]["choice"] == NONE_OPTION and outcomes[1].rungs[0]["choice"] == "bogus"


@pytest.mark.parametrize("threshold", [0, -1, 1.5, True, "0.8", None])
def test_an_invalid_threshold_is_refused(threshold):
    router, _ = _model_router()
    with pytest.raises(ValueError):
        router.decide([_item()], rubric="", org_id="o", threshold=threshold)


# --- a plugged-in decider ----------------------------------------------------------------------------------------

def test_a_plugged_in_decider_answers_instead_of_the_model():
    router, port, model = _port_router(docs={"choice": "missing", "confidence": 0.9})
    [outcome] = router.decide([_item()], rubric="the rubric", org_id="o", threshold=DEFAULT_THRESHOLD)
    assert model.calls == [] and len(port.calls) == 1
    assert port.calls[0]["rubric"] == "the rubric" and port.calls[0]["org_id"] == "o"
    assert (outcome.choice, outcome.confidence, outcome.decided_by) == ("missing", 0.9, "provider")
    assert outcome.rungs[0] == {"rung": "provider", "provider": "fake-decider", "model": "fake-decider-1",
                                "run_id": None, "choice": "missing", "confidence": 0.9, "accepted": True,
                                "evidence": [CONTEXT]}


def test_a_decider_that_cannot_answer_sends_every_item_to_a_person():
    port = FakeDecisionPort(unavailable=True)
    model = FakeModel()
    router = DecisionRouter(model=ModelDecisions(model), provider=port)
    outcomes = router.decide([_item("a"), _item("b")], rubric="", org_id="o", threshold=DEFAULT_THRESHOLD)
    assert model.calls == [] and all(o.needs_review for o in outcomes)
    assert [o.rungs[0]["error"] for o in outcomes] == ["decider_unreachable"] * 2


def test_an_item_the_decider_skips_goes_to_a_person():
    class Partial:
        def decide(self, items, *, rubric, org_id):
            return [DecisionAnswer(id=items[0].id, choice=items[0].options[0], confidence=0.99)]

    router = DecisionRouter(model=ModelDecisions(FakeModel()), provider=Partial())
    outcomes = router.decide([_item("a"), _item("b")], rubric="", org_id="o", threshold=DEFAULT_THRESHOLD)
    assert [o.needs_review for o in outcomes] == [False, True]
    assert outcomes[1].rungs[0]["error"] == "missing_answer"


def test_decision_unavailable_carries_only_its_code():
    err = DecisionUnavailable("decider_unreachable")
    assert err.code == "decider_unreachable" and str(err) == "decider_unreachable"


# --- inputs ------------------------------------------------------------------------------------------------------

def test_long_contexts_are_cut_and_duplicate_ids_refused():
    router, port, _ = _port_router()
    router.decide([_item(context="x" * (MAX_CONTEXT + 50))], rubric="", org_id="o", threshold=DEFAULT_THRESHOLD)
    assert len(port.calls[0]["items"][0].context) == MAX_CONTEXT
    with pytest.raises(ValueError):
        router.decide([_item("a"), _item("a")], rubric="", org_id="o", threshold=DEFAULT_THRESHOLD)
    assert router.decide([], rubric="", org_id="o", threshold=DEFAULT_THRESHOLD) == []


def test_evidence_is_kept_on_the_outcome_and_at_most_five_quotes_on_the_record():
    quotes = tuple(f"quote number {n}" for n in range(7))
    router, _, _ = _port_router(docs={"evidence": quotes})
    [outcome] = router.decide([_item()], rubric="", org_id="o", threshold=DEFAULT_THRESHOLD)
    assert outcome.evidence == quotes and outcome.rungs[0]["evidence"] == list(quotes[:5])
