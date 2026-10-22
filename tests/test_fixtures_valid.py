import json

import pytest
from jsonschema import Draft202012Validator

from daoharness.decision import DEFAULT_THRESHOLD, DecisionItem, DecisionRouter, ModelDecisions
from daoharness.guardrails import flatten_text
from daoharness.ports import ModelError
from daoharness.primitives import builtin_catalog
from daoharness.spec import WorkflowSpec
from daoharness.validator import ValidationContext, validate_spec, workflow_metrics
from industry_fixtures import ACCOUNTING, RECRUITING
from workflow_fakes import (
    Clock, FakeDecisionPort, FakeModel, LocalMembership, RecordingNotifier, as_model_payload, default_step_output,
    fill_from_schema)

FIXTURES = {"accounting": ACCOUNTING, "recruiting": RECRUITING}
SKELETON = ["collect", "extract", "assess", "route", "questions", "outreach", "approval", "summary", "package"]


def _spec(fixture):
    return WorkflowSpec.from_dict(fixture["spec"])


# --- industry fixtures ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_spec_validates(name):
    report = validate_spec(_spec(FIXTURES[name]), ValidationContext(builtin_catalog(), lambda r: True))
    assert report.ok, report.errors
    assert report.gaps == ()


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_metrics(name):
    m = workflow_metrics(_spec(FIXTURES[name]), builtin_catalog())
    assert m == {"total_steps": 9, "automatic_steps": 8, "automation_ratio": 0.89}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_shape(name):
    fixture = FIXTURES[name]
    assert set(fixture) == {"description", "spec", "sample"}
    spec = _spec(fixture)
    assert [s.id for s in spec.steps] == SKELETON
    assert len(spec.completion_criteria) == 2 and spec.deliverables and spec.domain and spec.purpose
    assert set(fixture["sample"]) == {r.name for r in spec.input_requirements}
    assert all(isinstance(v, str) and v for v in fixture["sample"].values())
    collect = spec.step("collect")
    assert sorted(collect.inputs) == sorted(f"input:{r.name}" for r in spec.input_requirements)


def test_primitive_lists_differ_only_at_assess():
    a = [s.primitive for s in _spec(ACCOUNTING).steps]
    r = [s.primitive for s in _spec(RECRUITING).steps]
    assert a[2] == "checklist_match" and r[2] == "screen_score"
    assert a[:2] + a[3:] == r[:2] + r[3:]


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_decision_steps_produce_items_for_the_router(name):
    spec, catalog = _spec(FIXTURES[name]), builtin_catalog()
    router = DecisionRouter(model=ModelDecisions(FakeModel()), provider=FakeDecisionPort())
    inputs = {f"input:{k}": v for k, v in FIXTURES[name]["sample"].items()}
    for step in spec.steps:
        p = catalog.get(step.primitive)
        if p.executor != "decision":
            continue
        items = p.decision_items(step.config, inputs, step.instructions)
        assert items and len({i.id for i in items}) == len(items)
        outcomes = router.decide(items, rubric=p.rubric, org_id="org", threshold=DEFAULT_THRESHOLD)
        assert [o.item_id for o in outcomes] == [i.id for i in items]
        assert not any(o.needs_review for o in outcomes)


# --- fake model doubles -----------------------------------------------------------------------------------------

def _step_input(spec, step_id, inputs):
    return {"step": spec.step(step_id).to_dict(), "inputs": inputs,
            "completion_criteria": list(spec.completion_criteria)}


def _generate(model, operation, input_data, schema=None):
    return model.generate(org_id="org", operation=operation, skill_id="skill", input_data=input_data,
                          instructions="", evidence=[], schema=schema or {"type": "object"})


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_default_step_outputs_fit_output_schemas(name):
    spec, catalog = _spec(FIXTURES[name]), builtin_catalog()
    inputs = {"input:" + k: v for k, v in FIXTURES[name]["sample"].items()}
    for step in spec.steps:
        p = catalog.get(step.primitive)
        if p.executor != "model":
            continue
        schema = p.output_schema_for(step.config)
        refs = {ref: "Upstream text. More." for ref in step.inputs}
        out, receipt = _generate(FakeModel(), "step", _step_input(spec, step.id, {**inputs, **refs}), schema)
        assert not list(Draft202012Validator(schema).iter_errors(out)), (step.id, out)
        assert receipt.provider == "fake" and receipt.output_digest.startswith("sha256:")


def test_extract_default_covers_configured_fields_with_quotes():
    spec = _spec(ACCOUNTING)
    text = "Northwind LLC files. Other."
    inputs = {"step:collect": {"packet": [{"ref": "input:documents", "text": text}]}}
    out = default_step_output(_step_input(spec, "extract", inputs), {})
    assert [f["name"] for f in out["fields"]] == ["legal_name", "entity_type", "owner_name"]
    assert all(f["found"] and f["source_quotes"] == ["Northwind LLC files"] for f in out["fields"])
    assert all(q in " ".join(flatten_text(inputs)) for f in out["fields"] for q in f["source_quotes"])


def test_summary_default_cites_input_steps_and_assesses_criteria():
    spec = _spec(RECRUITING)
    data = _step_input(spec, "summary", {"step:extract": "a", "step:assess": "b", "step:questions": "c"})
    out = default_step_output(data, {})
    assert out["sections"][0]["source_steps"] == ["extract", "assess", "questions"]
    assert [c["criterion"] for c in out["criteria"]] == list(spec.completion_criteria)


def test_draft_default_contains_a_maskable_identifier():
    spec = _spec(ACCOUNTING)
    out = default_step_output(_step_input(spec, "outreach", {"step:assess": "x"}), {})
    assert "123-45-6789" in out["body"] and out["subject"]


def test_fill_from_schema():
    schema = {"type": "object", "properties": {
        "a": {"type": "string"}, "b": {"type": "integer", "minimum": 2}, "c": {"type": "boolean"},
        "d": {"type": "array", "items": {"type": "string", "enum": ["x", "y"]}}, "e": {"type": "number"}}}
    assert fill_from_schema(schema) == {"a": "fixture", "b": 2, "c": True, "d": ["x"], "e": 1}


def test_compile_outputs_are_consumed_in_order_and_may_be_callable():
    model = FakeModel(compile_outputs=[{"n": 1}, lambda data: {"echo": data["need"]}])
    assert _generate(model, "compile", {"need": "x"})[0] == {"n": 1}
    assert _generate(model, "compile", {"need": "y"})[0] == {"echo": "y"}
    assert model.count("compile") == 2 and model.compile_outputs == []


def test_step_override_and_count_by_step():
    spec = _spec(ACCOUNTING)
    model = FakeModel(step_overrides={"questions": lambda data: {"questions": []}})
    out, _ = _generate(model, "step", _step_input(spec, "questions", {"step:assess": "x"}))
    assert out == {"questions": []}
    assert model.count("step", "questions") == 1 and model.count(step_id="extract") == 0


def test_step_override_may_raise():
    def boom(_data):
        raise ModelError("provider_failed")
    model = FakeModel(step_overrides={"questions": boom})
    with pytest.raises(ModelError):
        _generate(model, "step", _step_input(_spec(ACCOUNTING), "questions", {"step:assess": "x"}))


def test_fail_operations_raise_model_error():
    with pytest.raises(ModelError) as info:
        _generate(FakeModel(fail_operations={"compile"}), "compile", {})
    assert info.value.code == "provider_failed"


def test_decide_is_supported():
    item = DecisionItem(id="docs", question="Q?", options=("present", "missing"), context="ctx")
    model = FakeModel(decide_overrides={"docs": {"choice": "missing"}})
    answers, receipt = ModelDecisions(model).decide([item], rubric="", org_id="org")
    assert [a.choice for a in answers] == ["missing"] and model.count("decide") == 1 and receipt.run_id == "fake-1"


def test_receipt_ids_are_sequential():
    model = FakeModel(compile_outputs=[{}, {}])
    assert [_generate(model, "compile", {})[1].run_id for _ in range(2)] == ["fake-1", "fake-2"]


def test_fake_decider_quotes_the_start_of_the_context():
    item = DecisionItem(id="docs", question="Q?", options=("present", "missing"),
                        context="Bank statements attached for the last three months, signed.")
    [answer] = FakeDecisionPort().decide([item], rubric="", org_id="org")
    assert answer.choice == "present" and answer.evidence == ("Bank statements attached for the last",)
    [blank] = FakeDecisionPort().decide([DecisionItem(id="x", question="Q?", options=("a", "b"), context=" ")],
                                        rubric="", org_id="org")
    assert blank.evidence == ()


def test_clock_membership_notifier():
    clock = Clock()
    assert clock() == "2026-01-01T00:00:01Z" and clock() == "2026-01-01T00:00:02Z"
    assert LocalMembership().resolve("org-1", "Lead") == "org-1"
    notifier = RecordingNotifier()
    notifier.approval_requested("org-1", {"approval_id": "a" * 32})
    assert notifier.events == [("org-1", "a" * 32)]


def test_as_model_payload_shapes_compiler_output():
    payload = as_model_payload(ACCOUNTING["spec"])
    assert set(payload) == {"spec", "declared_gaps"} and payload["declared_gaps"] == []
    steps = payload["spec"]["steps"]
    assert [s["id"] for s in steps] == SKELETON
    for raw, shaped in zip(ACCOUNTING["spec"]["steps"], steps):
        assert json.loads(shaped["config_json"]) == raw.get("config", {})
        assert not {"config", "primitive_version", "origin", "guardrails"} & set(shaped)
        assert {"required_sources", "approval", "inputs", "for_each"} <= set(shaped)
    gaps = [{"step_id": "outreach", "kind": "needs_capability", "detail": "send"}]
    assert as_model_payload(ACCOUNTING["spec"], gaps=gaps)["declared_gaps"] == gaps


# --- per-item fan-out fixture -------------------------------------------------------------------------------------

def test_fan_out_fixture_valid():
    from industry_fixtures import fan_out_fake_outputs, recruiting_fan_out_sample, recruiting_fan_out_spec

    spec = WorkflowSpec.from_dict(recruiting_fan_out_spec())
    report = validate_spec(spec, ValidationContext(builtin_catalog(), lambda r: True))
    assert report.ok, report.errors
    assert report.metrics["automation_ratio"] == 0.89
    assert report.metrics["fan_out"] == {"split_step": "split", "max_items": 20,
                                         "per_item_steps": ["extract", "assess", "route", "outreach"]}
    sample = recruiting_fan_out_sample()
    assert set(sample) == {r.name for r in spec.input_requirements}
    split = fan_out_fake_outputs()["split"]({"inputs": {"input:packet": sample["packet"]}})
    assert [i["label"] for i in split["items"]] == ["Priya Shah", "Marco Silva", "Li Wei"]
