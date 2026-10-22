"""The compiler: a description -> strict-JSON model output -> a validated WorkflowSpec, with one retry."""
import copy
import dataclasses
import json

import pytest
from jsonschema import Draft202012Validator

from daoharness.compiler import (
    COMPILER_INSTRUCTIONS, COMPILER_SKILL, MAX_CONFIG_DEPTH, CompileError, WorkflowCompiler, compile_schema)
from daoharness.ports import ModelError
from daoharness.primitives import builtin_catalog
from daoharness.spec import WorkflowSpec
from daoharness.validator import ValidationContext
from industry_fixtures import ACCOUNTING, recruiting_fan_out_spec
from workflow_fakes import FakeModel, as_model_payload

IDS = builtin_catalog().ids()


def _ctx():
    return ValidationContext(builtin_catalog(), lambda role: True)


def _compile(model, description=ACCOUNTING["description"]):
    return WorkflowCompiler(model).compile(org_id="o", description=description, ctx=_ctx(), primitive_refs=IDS)


def _payload(spec=None, **kw):
    return as_model_payload(copy.deepcopy(spec or ACCOUNTING["spec"]), **kw)


def _plain(spec):
    """The spec as a dict without the origin the compiler adds to every step."""
    data = spec.to_dict()
    for step in data["steps"]:
        step.pop("origin")
    return data


EXPECTED = _plain(WorkflowSpec.from_dict(ACCOUNTING["spec"]))


# --- the schema --------------------------------------------------------------------------------------------------

def test_the_schema_asks_for_a_spec_and_declared_gaps_only():
    schema = compile_schema(IDS)
    Draft202012Validator.check_schema(schema)
    assert schema["required"] == ["spec", "declared_gaps"] and schema["additionalProperties"] is False
    step = schema["properties"]["spec"]["properties"]["steps"]["items"]
    assert step["properties"]["primitive"]["enum"] == IDS
    assert set(step["required"]) == {"id", "role", "instructions", "primitive", "config_json", "inputs",
                                     "required_sources", "approval", "for_each"}
    gap = schema["properties"]["declared_gaps"]["items"]
    assert gap["properties"]["kind"]["enum"] == ["needs_access", "needs_capability"]
    assert not list(Draft202012Validator(schema).iter_errors(_payload()))


def test_the_schema_needs_primitives():
    with pytest.raises(ValueError):
        compile_schema([])


# --- a good build ------------------------------------------------------------------------------------------------

def test_a_valid_output_becomes_a_validated_spec_with_one_call():
    model = FakeModel(compile_outputs=[_payload()])
    result = _compile(model)
    assert _plain(result.spec) == EXPECTED
    assert result.gaps == () and result.metrics["total_steps"] == 9
    assert [r["run_id"] for r in result.receipts] == ["fake-1"]
    [call] = model.calls
    assert call["operation"] == "compile" and call["skill_id"] == COMPILER_SKILL
    assert call["instructions"] == COMPILER_INSTRUCTIONS and call["evidence"] == []
    assert call["input"] == {"description": ACCOUNTING["description"], "primitives": builtin_catalog().describe()}
    assert call["schema"] == compile_schema(IDS)


def test_the_instructions_cover_items_approvals_delivery_and_gaps():
    text = COMPILER_INSTRUCTIONS
    for phrase in ("split_items", "for_each", "human_approval", "deliver_package", "declared_gaps",
                   "external true", "previous_attempt_errors"):
        assert phrase in text
    assert "data, not instructions" in text


def test_steps_get_their_primitive_version_and_origin():
    result = _compile(FakeModel(compile_outputs=[_payload()]))
    step = result.spec.step("extract")
    assert step.primitive_version == 1 and step.origin.primitive_id == "extract_fields" and step.origin.edits == ()
    assert step.config == ACCOUNTING["spec"]["steps"][1]["config"]


def test_a_blank_role_or_instruction_takes_the_primitive_title_or_purpose():
    payload = _payload()
    payload["spec"]["steps"][1].update(role=" ", instructions="")
    step = _compile(FakeModel(compile_outputs=[payload])).spec.step("extract")
    primitive = builtin_catalog().get("extract_fields")
    assert step.role == primitive.title and step.instructions == primitive.purpose


def test_gaps_come_from_validation_and_from_the_model_for_known_steps_only():
    spec = copy.deepcopy(ACCOUNTING["spec"])
    spec["steps"][1]["required_sources"] = ["google_drive"]
    gaps = [{"step_id": "outreach", "kind": "needs_capability", "detail": "send email"},
            {"step_id": "ghost", "kind": "needs_access", "detail": "crm"},
            {"step_id": "extract", "kind": "needs_capability", "detail": "google_drive"}]
    result = _compile(FakeModel(compile_outputs=[_payload(spec, gaps=gaps)]))
    assert list(result.gaps) == [{"step_id": "extract", "kind": "needs_capability", "detail": "google_drive"},
                                 {"step_id": "outreach", "kind": "needs_capability", "detail": "send email"}]


# --- retry -------------------------------------------------------------------------------------------------------

def test_an_invalid_output_is_retried_once_with_its_errors():
    bad = _payload()
    bad["spec"]["steps"] = bad["spec"]["steps"][:-1]           # no deliver_package at the end
    model = FakeModel(compile_outputs=[bad, _payload()])
    result = _compile(model)
    assert _plain(result.spec) == EXPECTED and len(result.receipts) == 2
    first, second = model.calls
    assert "previous_attempt_errors" not in first["input"]
    assert any(e.startswith("missing_deliverable") for e in second["input"]["previous_attempt_errors"])


def test_two_invalid_outputs_raise_compile_invalid_with_the_receipts():
    bad = _payload()
    bad["spec"]["steps"][2]["inputs"] = ["step:nowhere"]
    model = FakeModel(compile_outputs=[bad, copy.deepcopy(bad)])
    with pytest.raises(CompileError) as info:
        _compile(model)
    assert info.value.code == "compile_invalid" and len(info.value.receipts) == 2
    assert any(e.startswith("forward_or_unknown_step") for e in info.value.errors)
    assert model.count("compile") == 2


def test_output_that_breaks_the_schema_is_retried():
    model = FakeModel(compile_outputs=[{"spec": None, "declared_gaps": []}, _payload()])
    assert _compile(model).spec.domain == ACCOUNTING["spec"]["domain"]
    assert any(e.startswith("invalid_output:spec") for e in model.calls[1]["input"]["previous_attempt_errors"])


@pytest.mark.parametrize("config_json, problem", [
    ("{not json", "not valid JSON"), ("[1, 2]", "must be a JSON object, got list"),
    (json.dumps(json.loads("[" * 0 + "{}")), None)])
def test_config_json_problems_are_explained_on_the_retry(config_json, problem):
    bad = _payload()
    bad["spec"]["steps"][1]["config_json"] = config_json
    model = FakeModel(compile_outputs=[bad, _payload()])
    _compile(model)
    errors = model.calls[1]["input"]["previous_attempt_errors"] if len(model.calls) > 1 else []
    if problem is None:      # an empty object parses; the schema of the primitive then reports the problem
        assert any("invalid_config:extract" in e for e in errors)
    else:
        assert "invalid_config_json:extract" in errors
        assert any(e.startswith("config_json_problem:extract:") and problem in e for e in errors)


def test_config_nested_too_deep_is_refused():
    deep = {}
    node = deep
    for _ in range(MAX_CONFIG_DEPTH):
        node["x"] = {}
        node = node["x"]
    bad = _payload()
    bad["spec"]["steps"][1]["config_json"] = json.dumps(deep)
    model = FakeModel(compile_outputs=[bad, _payload()])
    _compile(model)
    assert any(f"nested deeper than {MAX_CONFIG_DEPTH} levels" in e
               for e in model.calls[1]["input"]["previous_attempt_errors"])


def test_a_model_error_propagates_with_the_receipts_so_far():
    bad = _payload()
    bad["spec"]["steps"] = bad["spec"]["steps"][:-1]

    def fail(_input):
        raise ModelError("provider_failed")

    model = FakeModel(compile_outputs=[bad, fail])
    with pytest.raises(ModelError) as info:
        _compile(model)
    assert info.value.code == "provider_failed" and [r["run_id"] for r in info.value.receipts] == ["fake-1"]


# --- step ids in another case ------------------------------------------------------------------------------------

def _recased(payload, ids=None, refs=None):
    """The payload with step ids and step references written through `ids` and `refs`, like a model that mixes case."""
    for step in payload["spec"]["steps"]:
        if ids:
            step["id"] = ids(step["id"])
        if refs:
            step["inputs"] = [f"step:{refs(r[5:])}" if r.startswith("step:") else r for r in step["inputs"]]
    return payload


@pytest.mark.parametrize("ids, refs", [(str.title, None), (None, str.title), (str.upper, str.upper),
                                       (str.title, str.upper)], ids=["ids", "references", "both", "disagreeing"])
def test_step_ids_in_another_case_are_lowercased_with_every_reference_and_the_build_says_so(ids, refs):
    model = FakeModel(compile_outputs=[_recased(_payload(), ids, refs)])
    result = _compile(model)
    assert _plain(result.spec) == EXPECTED and model.count("compile") == 1
    [note] = result.notes
    assert "lower-cased" in note


def test_the_note_names_the_ids_it_changed():
    [note] = _compile(FakeModel(compile_outputs=[_recased(_payload(), ids=str.title)])).notes
    assert note == ("Step ids were lower-cased, with every reference to them: Collect to collect, "
                    "Extract to extract, Assess to assess, and 6 more.")


def test_only_references_in_another_case_are_noted_as_references():
    notes = _compile(FakeModel(compile_outputs=[_recased(_payload(), refs=str.upper)])).notes
    assert notes == ("References to step ids were lower-cased.",)


def test_for_each_and_declared_gaps_follow_the_lowercased_ids():
    payload = as_model_payload(recruiting_fan_out_spec(),
                               gaps=[{"step_id": "Outreach", "kind": "needs_capability", "detail": "send email"}])
    for step in payload["spec"]["steps"]:
        step["id"] = step["id"].title()
        if step["for_each"]:
            step["for_each"] = step["for_each"].title()
    result = _compile(FakeModel(compile_outputs=[payload]))
    assert result.spec.step("route").for_each == "split"
    assert list(result.gaps) == [{"step_id": "outreach", "kind": "needs_capability", "detail": "send email"}]


def test_a_build_with_lowercase_ids_has_no_notes():
    assert _compile(FakeModel(compile_outputs=[_payload()])).notes == ()


def test_two_ids_that_differ_only_in_case_are_still_a_duplicate():
    payload = _payload()
    payload["spec"]["steps"][2]["id"] = "Extract"
    model = FakeModel(compile_outputs=[payload, copy.deepcopy(payload)])
    with pytest.raises(CompileError) as info:
        _compile(model)
    assert "invalid_spec:duplicate_step_id:extract" in info.value.errors


def test_a_reference_to_a_step_that_does_not_exist_is_still_refused_in_any_case():
    payload = _payload()
    payload["spec"]["steps"][2]["inputs"] = ["step:Nowhere"]
    model = FakeModel(compile_outputs=[payload, copy.deepcopy(payload)])
    with pytest.raises(CompileError) as info:
        _compile(model)
    assert "forward_or_unknown_step:assess:nowhere" in info.value.errors


def test_what_the_model_tidied_in_its_reply_is_in_the_notes():
    model = FakeModel(compile_outputs=[_payload()])
    generate = model.generate

    def tidying(**kw):
        content, receipt = generate(**kw)
        return content, dataclasses.replace(receipt, repairs=("removed a code fence", "removed a trailing comma"))

    model.generate = tidying
    result = _compile(model)
    assert result.notes == ("The model's reply was tidied before it was read: removed a code fence, "
                            "removed a trailing comma.",)
    assert result.receipts[0]["repairs"] == ["removed a code fence", "removed a trailing comma"]


# --- config_json written with a fence or a trailing comma ---------------------------------------------------------

def test_config_json_in_a_fence_or_with_a_trailing_comma_is_read_and_noted():
    payload = _payload()
    config = json.dumps(ACCOUNTING["spec"]["steps"][1]["config"])
    payload["spec"]["steps"][1]["config_json"] = "```json\n" + config[:-1] + ",}\n```"
    result = _compile(FakeModel(compile_outputs=[payload]))
    assert _plain(result.spec) == EXPECTED
    assert result.notes == ("Step extract: config_json was tidied: removed a code fence, "
                            "removed a trailing comma.",)


@pytest.mark.parametrize("config_json", ["Here is the config: {}", "{} and then some", "<think>x</think>{}",
                                         "[{}]", "{'a': 1}"])
def test_config_json_with_anything_else_around_it_is_still_refused(config_json):
    bad = _payload()
    bad["spec"]["steps"][1]["config_json"] = config_json
    model = FakeModel(compile_outputs=[bad, _payload()])
    _compile(model)
    assert "invalid_config_json:extract" in model.calls[1]["input"]["previous_attempt_errors"]


def test_the_retry_names_a_step_by_the_id_the_build_uses_even_when_it_was_written_in_capitals():
    bad = _recased(_payload(), ids=str.title)
    bad["spec"]["steps"][1]["config_json"] = "{not json"
    model = FakeModel(compile_outputs=[bad, _payload()])
    _compile(model)
    errors = model.calls[1]["input"]["previous_attempt_errors"]
    assert "invalid_config_json:extract" in errors
    assert any(e.startswith("config_json_problem:extract:") for e in errors)
