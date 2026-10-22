"""Compiler: per-item segments in the schema, the instructions and the build."""
from jsonschema import Draft202012Validator

from daoharness.compiler import COMPILER_INSTRUCTIONS, NAME_PATTERN, WorkflowCompiler, compile_schema
from daoharness.primitives import builtin_catalog
from daoharness.spec import WorkflowSpec
from daoharness.validator import ValidationContext
from industry_fixtures import ACCOUNTING, recruiting_fan_out_spec
from workflow_fakes import FakeModel, as_model_payload

BUILTIN_IDS = builtin_catalog().ids()


def _ctx():
    return ValidationContext(builtin_catalog(), lambda r: True)


def _compile(model, description="For each applicant in the packet, screen and reply."):
    return WorkflowCompiler(model).compile(org_id="org-1", description=description, ctx=_ctx(),
                                           primitive_refs=list(BUILTIN_IDS))


def test_schema_accepts_for_each_null_and_name():
    schema = compile_schema(list(BUILTIN_IDS))
    step = schema["properties"]["spec"]["properties"]["steps"]["items"]
    assert step["properties"]["for_each"] == {"anyOf": [NAME_PATTERN, {"type": "null"}]}
    assert "for_each" in step["required"]  # strict mode: every property is required, null when unused
    payload = as_model_payload(recruiting_fan_out_spec())
    assert not list(Draft202012Validator(schema).iter_errors(payload))
    assert payload["spec"]["steps"][2]["for_each"] == "split" and payload["spec"]["steps"][0]["for_each"] is None
    payload["spec"]["steps"][2]["for_each"] = "Split"
    assert list(Draft202012Validator(schema).iter_errors(payload))


def test_the_build_keeps_for_each():
    result = _compile(FakeModel(compile_outputs=[as_model_payload(recruiting_fan_out_spec())]))
    assert result.spec.step("extract").for_each == "split" and result.spec.step("split").for_each is None
    assert "for_each" not in result.spec.step("collect").to_dict()
    assert result.metrics["fan_out"]["per_item_steps"] == ["extract", "assess", "route", "outreach"]
    assert result.metrics["automation_ratio"] == 0.89


def test_single_case_build_has_no_for_each():
    result = _compile(FakeModel(compile_outputs=[as_model_payload(ACCOUNTING["spec"])]), ACCOUNTING["description"])
    assert all("for_each" not in s.to_dict() for s in result.spec.steps)
    assert "fan_out" not in result.metrics
    assert result.spec.digest() == WorkflowSpec.from_dict(result.spec.to_dict()).digest()


def test_an_invalid_segment_is_retried_with_its_errors():
    broken = recruiting_fan_out_spec()
    broken["steps"][4].pop("for_each")  # route between per-item steps: the segment is not contiguous
    model = FakeModel(compile_outputs=[as_model_payload(broken), as_model_payload(recruiting_fan_out_spec())])
    result = _compile(model)
    assert result.spec.step("route").for_each == "split" and model.count("compile") == 2
    errors = model.calls[1]["input"]["previous_attempt_errors"]
    assert any(e.startswith("segment_not_contiguous:") for e in errors)


def test_the_instructions_describe_per_item_steps():
    for phrase in ("split_items", "for_each", "one item each", "a single human_approval after them"):
        assert phrase in COMPILER_INSTRUCTIONS, phrase


def test_the_split_schema_says_the_separator_is_literal_text():
    described = builtin_catalog().get("split_items").describe()["config_schema"]["properties"]["separator"]
    description = described["description"].lower()
    assert "exact literal text" in description and "not a description" in description
