"""Finite-choice decisions: one decider (a plugged-in DecisionPort, else the model) and a person below the
threshold."""
import copy
import math

import pytest

from daoharness.decision import (
    DEFAULT_THRESHOLD, MAX_CONTEXT, NONE_OPTION, DecisionAnswer, DecisionItem, DecisionRouter, DecisionUnavailable,
    ModelDecisions)
from daoharness.ports import ModelReceipt
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


# --- a model that is slightly off ------------------------------------------------------------------------------------

class Replying:
    """A model that answers every decide call with the given reply, as it is."""

    def __init__(self, reply, repairs=None):
        self.reply, self.repairs, self.calls = reply, repairs, 0

    def generate(self, *, org_id, operation, skill_id, input_data, instructions, evidence, schema):
        self.calls += 1
        return copy.deepcopy(self.reply), ModelReceipt(run_id="r1", provider="p", model="m", output_digest="d",
                                                        repairs=self.repairs)


def _answer(**fields):
    return {"choice": "present", "confidence": 0.9, "evidence_quotes": [], "rationale": "r", **fields}


def _decide(reply, ids=("a", "b", "c"), threshold=DEFAULT_THRESHOLD, repairs=None, **item_kw):
    router = DecisionRouter(model=ModelDecisions(Replying(reply, repairs)))
    items = [_item(i, **item_kw) for i in ids]
    return router.decide(items, rubric="", org_id="o", threshold=threshold)


def _batch(**overrides):
    return {i: _answer(**overrides.get(i, {})) for i in ("a", "b", "c")}


@pytest.mark.parametrize("written, read", [("Present", "present"), ("PRESENT", "present"), (" present ", "present"),
                                           ("Missing\n", "missing")])
def test_a_choice_in_another_case_is_matched_to_its_option_and_the_record_says_so(written, read):
    [a, b, c] = _decide(_batch(a={"choice": written}))
    assert (a.choice, a.needs_review, a.decided_by) == (read, False, "model")
    assert a.rungs[0]["choice"] == read and a.rungs[0]["normalized"] == [f"choice {written!r} read as {read!r}"]
    assert "normalized" not in b.rungs[0] and (b.choice, c.choice) == ("present", "present")


@pytest.mark.parametrize("written, read, note", [
    ("0.9", 0.9, "confidence '0.9' read as 0.9"), (" 0.9 ", 0.9, "confidence ' 0.9 ' read as 0.9"),
    ("92%", 0.92, "confidence '92%' read as 0.92"), ("92 %", 0.92, "confidence '92 %' read as 0.92"),
    ("87.5%", 0.875, "confidence '87.5%' read as 0.875"), ("100%", 1.0, "confidence '100%' read as 1.0"),
    (92, 0.92, "confidence 92 read as 0.92"), (87.5, 0.875, "confidence 87.5 read as 0.875"),
    (100, 1.0, "confidence 100 read as 1.0"), ("92", 0.92, "confidence '92' read as 0.92")])
def test_a_confidence_written_as_text_or_a_percentage_becomes_a_number_from_0_to_1(written, read, note):
    [a, *_] = _decide(_batch(a={"confidence": written}))
    assert (a.confidence, a.needs_review) == (read, False)
    assert a.rungs[0]["confidence"] == read and a.rungs[0]["normalized"] == [note]


@pytest.mark.parametrize("written", [0.9, 1, 1.0, 0, 0.0, 0.75])
def test_a_confidence_that_is_already_from_0_to_1_is_left_alone(written):
    [a, *_] = _decide(_batch(a={"confidence": written}), threshold=0.5)
    assert a.rungs[0]["confidence"] == written and "normalized" not in a.rungs[0]


def test_the_threshold_applies_to_the_confidence_after_it_is_read():
    outcomes = _decide(_batch(a={"confidence": "75%"}, b={"confidence": "74%"}, c={"confidence": 60}))
    assert [o.needs_review for o in outcomes] == [False, True, True]
    assert [o.rungs[0]["confidence"] for o in outcomes] == [0.75, 0.74, 0.6]


@pytest.mark.parametrize("confidence", ["high", "", "nan", "inf", "0.9 or so", "9e1", "1_0", "-5%", "101%", "0,9",
                                        None, True, [0.9], {"v": 1}])
def test_a_confidence_that_cannot_be_read_sends_only_that_item_to_a_person(confidence):
    [a, b, c] = _decide(_batch(a={"confidence": confidence}))
    assert a.needs_review and a.choice is None and a.rungs[0]["error"] == "invalid_decision_output"
    assert [b.needs_review, c.needs_review] == [False, False] and (b.decided_by, c.decided_by) == ("model", "model")


@pytest.mark.parametrize("confidence", [101, 250.5, -0.1, -5, math.inf, math.nan])
def test_a_number_outside_0_to_100_is_never_accepted(confidence):
    [a, *_] = _decide(_batch(a={"confidence": confidence}))
    assert a.needs_review and a.choice is None and "normalized" not in a.rungs[0]


@pytest.mark.parametrize("choice", ["not an option", "", "presen", None, 3, ["present"]])
def test_a_choice_that_matches_no_option_sends_only_that_item_to_a_person(choice):
    outcomes = _decide(_batch(b={"choice": choice}))
    assert [o.needs_review for o in outcomes] == [False, True, False]
    assert outcomes[1].rungs[0]["error"] == "invalid_decision_output" and outcomes[1].rungs[0]["accepted"] is False
    assert outcomes[1].rungs[0]["choice"] is None and outcomes[1].rungs[0]["confidence"] is None
    assert [outcomes[0].choice, outcomes[2].choice] == ["present", "present"]


def test_a_choice_that_fits_two_options_only_by_case_is_not_guessed():
    options = ("Yes", "YES", "no")
    outcomes = _decide({i: _answer(choice="yes") for i in "abc"}, options=options)
    assert all(o.needs_review and o.rungs[0]["error"] == "invalid_decision_output" for o in outcomes)
    exact = _decide({i: _answer(choice="YES") for i in "abc"}, options=options)
    assert [o.choice for o in exact] == ["YES"] * 3 and not any("normalized" in o.rungs[0] for o in exact)


def test_an_item_that_is_missing_has_extra_fields_or_lacks_one_goes_to_a_person_on_its_own():
    reply = {"a": _answer(), "b": _answer(mood="sure"), "c": {k: v for k, v in _answer().items() if k != "rationale"}}
    outcomes = _decide(reply, ids=("a", "b", "c", "d"))
    assert [o.needs_review for o in outcomes] == [False, True, True, True]
    assert [o.rungs[0].get("error") for o in outcomes] == [
        None, "invalid_decision_output", "invalid_decision_output", "invalid_decision_output"]


@pytest.mark.parametrize("evidence", ["a quote", [1], ["x"] * 6, ["x" * 1001]])
def test_evidence_and_rationale_are_still_checked(evidence):
    [a, b, c] = _decide(_batch(a={"evidence_quotes": evidence}, b={"rationale": 5}, c={"rationale": "x" * 2001}))
    assert all(o.needs_review and o.rungs[0]["error"] == "invalid_decision_output" for o in (a, b, c))


@pytest.mark.parametrize("reply", [[], "answer", None, {**_batch(), "extra": _answer()}])
def test_a_reply_that_is_not_the_requested_object_sends_every_item_to_a_person(reply):
    outcomes = _decide(reply)
    assert all(o.needs_review and o.rungs[0]["error"] == "invalid_decision_output" for o in outcomes)


def test_a_partly_wrong_batch_still_costs_one_call_and_keeps_its_receipt():
    model = Replying(_batch(b={"choice": "bogus"}))
    answers, receipt = ModelDecisions(model).decide([_item(i) for i in "abc"], rubric="", org_id="o")
    assert model.calls == 1 and receipt.run_id == "r1"
    assert [a.id for a in answers] == ["a", "b", "c"]
    assert [a.error for a in answers] == [None, "invalid_decision_output", None]
    assert answers[1].choice is None and answers[1].confidence is None
    router = DecisionRouter(model=ModelDecisions(Replying(_batch(b={"choice": "bogus"}))))
    outcomes, receipt = router.decide_with_receipt([_item(i) for i in "abc"], rubric="", org_id="o",
                                                   threshold=DEFAULT_THRESHOLD)
    assert receipt.run_id == "r1" and outcomes[1].rungs[0]["run_id"] == "r1"


def test_what_was_tidied_in_the_reply_is_on_every_answer_it_covers():
    outcomes = _decide(_batch(a={"choice": "Present"}), repairs=("removed a code fence",))
    assert outcomes[0].rungs[0]["normalized"] == ["reply tidied: removed a code fence",
                                                  "choice 'Present' read as 'present'"]
    assert outcomes[1].rungs[0]["normalized"] == ["reply tidied: removed a code fence"]


def test_a_plugged_in_decider_is_not_read_generously():
    # Only the model's free-form JSON is tidied; a DecisionPort returns typed answers and is held to them.
    router, port, _ = _port_router(docs={"choice": "Present"})
    [outcome] = router.decide([_item()], rubric="", org_id="o", threshold=DEFAULT_THRESHOLD)
    assert outcome.needs_review and outcome.rungs[0]["choice"] == "Present"
