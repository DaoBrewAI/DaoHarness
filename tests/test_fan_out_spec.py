"""The optional per-item field `for_each` on a workflow step."""
import copy

import pytest

from daoharness.spec import WorkflowSpec, WorkflowSpecError, digest
from industry_fixtures import ACCOUNTING, RECRUITING

STEP_KEYS = {"id", "role", "instructions", "primitive", "primitive_version", "config", "inputs", "guardrails",
             "required_sources", "approval", "origin"}


def _step(id, primitive, inputs, config=None, **extra):
    return {"id": id, "role": f"{id} role", "instructions": f"Do {id}.", "primitive": primitive,
            "config": config or {}, "inputs": list(inputs), **extra}


def fan_out_spec_dict():
    return {
        "domain": "Packet review", "purpose": "Review every item in a packet.",
        "input_requirements": [{"name": "packet", "description": "Several items", "required": True}],
        "deliverables": ["Review"], "completion_criteria": ["Every item is reviewed"],
        "steps": [
            _step("split", "split_items", ["input:packet"], {"item_noun": "item"}),
            _step("extract", "extract_fields", ["step:split"],
                  {"fields": [{"name": "title", "description": "Title", "required": True}]}, for_each="split"),
            _step("package", "deliver_package", ["step:extract"]),
        ]}


def with_for_each(value):
    data = fan_out_spec_dict()
    data["steps"][1]["for_each"] = value
    return data


def test_for_each_round_trips_and_is_absent_when_unset():
    spec = WorkflowSpec.from_dict(fan_out_spec_dict())
    assert spec.step("extract").for_each == "split"
    assert spec.step("extract").to_dict()["for_each"] == "split"
    assert spec.step("split").for_each is None
    assert "for_each" not in spec.step("split").to_dict()
    assert WorkflowSpec.from_dict(spec.to_dict()) == spec


def test_explicit_null_is_absent():
    spec = WorkflowSpec.from_dict(with_for_each(None))
    assert spec.step("extract").for_each is None
    assert "for_each" not in spec.step("extract").to_dict()


def test_existing_spec_digest_is_unchanged():
    for fixture in (ACCOUNTING["spec"], RECRUITING["spec"]):
        spec = WorkflowSpec.from_dict(copy.deepcopy(fixture))
        data = spec.to_dict()
        assert all(set(step) == STEP_KEYS for step in data["steps"])
        assert spec.digest() == digest(data)


@pytest.mark.parametrize("bad", ["", "Split", "step:split", 3, True, ["split"]])
def test_for_each_must_be_a_step_name(bad):
    with pytest.raises(WorkflowSpecError):
        WorkflowSpec.from_dict(with_for_each(bad))
