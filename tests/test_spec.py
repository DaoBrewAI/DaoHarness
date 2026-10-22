import copy

import pytest

from daoharness.spec import WorkflowSpec, WorkflowSpecError, digest

BASE = {
    "domain": "Sample intake",
    "purpose": "Turn a submitted packet into a reviewed file.",
    "input_requirements": [{"name": "packet", "description": "Submitted text", "required": True}],
    "deliverables": ["Reviewed file"],
    "completion_criteria": ["Every item has a status"],
    "steps": [
        {"id": "collect", "role": "Coordinator", "instructions": "Gather inputs.", "primitive": "collect_inputs",
         "config": {}, "inputs": ["input:packet"]},
        {"id": "approve", "role": "Reviewer", "instructions": "Approve.", "primitive": "human_approval",
         "config": {}, "inputs": ["step:collect"],
         "approval": {"approver_role": "Lead", "what_is_approved": "Packet"}},
    ],
}


def test_round_trip_and_digest_are_stable():
    spec = WorkflowSpec.from_dict(BASE)
    again = WorkflowSpec.from_dict(spec.to_dict())
    assert again == spec
    assert spec.digest() == again.digest() == digest(spec.to_dict())
    assert spec.steps[1].approval.approver_role == "Lead"
    assert spec.steps[0].primitive_version == 1 and spec.steps[0].origin is None


@pytest.mark.parametrize("mutate,detail", [
    (lambda d: d["steps"].append(dict(d["steps"][0])), "duplicate_step_id"),
    (lambda d: d["steps"][0].update(id="Bad Id"), "steps[0].id"),
    (lambda d: d["steps"][0].update(inputs=["packet"]), "steps[0].inputs"),
    (lambda d: d.update(steps=[]), "steps"),
    (lambda d: d["input_requirements"].append(dict(d["input_requirements"][0])), "duplicate_input"),
    (lambda d: d["steps"][0].update(config={"x": float("nan")}), "steps[0].config"),
    (lambda d: d["steps"][0].update(primitive="Not A Primitive"), "steps[0].primitive"),
    (lambda d: d["steps"][0].update(guardrails=["unknown"]), "steps[0].guardrails"),
    (lambda d: d["steps"][0].update(id="collect\n"), "steps[0].id"),
    (lambda d: d["steps"][0].update(inputs=["input:packet\n"]), "steps[0].inputs"),
    (lambda d: d["steps"][0].update(primitive="collect_inputs\n"), "steps[0].primitive"),
    (lambda d: d["steps"][0].update(primitive="org.request_missing_items\n"), "steps[0].primitive"),
])
def test_structural_errors(mutate, detail):
    data = copy.deepcopy(BASE)
    mutate(data)
    with pytest.raises(WorkflowSpecError) as err:
        WorkflowSpec.from_dict(data)
    assert err.value.code == "invalid_spec" and detail in err.value.detail


def test_max_24_steps():
    data = copy.deepcopy(BASE)
    data["steps"] = [dict(BASE["steps"][0], id=f"s{i}") for i in range(25)]
    with pytest.raises(WorkflowSpecError):
        WorkflowSpec.from_dict(data)


@pytest.mark.parametrize("primitive", ["org.request_missing_items", "workflow:" + "a" * 32 + "@2", "a" * 65])
def test_primitive_refs_are_plain_ids(primitive):
    data = copy.deepcopy(BASE)
    data["steps"][0]["primitive"] = primitive
    with pytest.raises(WorkflowSpecError) as err:
        WorkflowSpec.from_dict(data)
    assert err.value.detail == "steps[0].primitive"


def test_origin_round_trips_and_rejects_non_integer_version():
    data = copy.deepcopy(BASE)
    data["steps"][0]["origin"] = {"primitive_id": "collect_inputs", "primitive_version": 3, "edits": ["renamed"]}
    spec = WorkflowSpec.from_dict(data)
    assert spec.steps[0].origin.primitive_version == 3 and spec.steps[0].origin.edits == ("renamed",)
    assert WorkflowSpec.from_dict(spec.to_dict()) == spec
    for bad in (True, 0, "2"):
        data["steps"][0]["origin"]["primitive_version"] = bad
        with pytest.raises(WorkflowSpecError) as err:
            WorkflowSpec.from_dict(data)
        assert "steps[0].origin" in err.value.detail


def test_unknown_step_keys_are_dropped_and_do_not_change_the_digest():
    data = copy.deepcopy(BASE)
    data["steps"][0]["estimate"] = 0.5
    spec = WorkflowSpec.from_dict(data)
    assert "estimate" not in spec.steps[0].to_dict()
    assert spec.digest() == WorkflowSpec.from_dict(BASE).digest()


def test_for_each_is_absent_until_set():
    data = copy.deepcopy(BASE)
    assert "for_each" not in WorkflowSpec.from_dict(data).steps[0].to_dict()
    data["steps"][1]["for_each"] = "collect"
    assert WorkflowSpec.from_dict(data).steps[1].to_dict()["for_each"] == "collect"
