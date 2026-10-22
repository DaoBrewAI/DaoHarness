import copy

from daoharness.diff import spec_diff
from daoharness.spec import WorkflowSpec
from test_spec import BASE  # same folder; pytest's default rootdir-relative import


def test_diff_added_removed_changed_reordered():
    old = WorkflowSpec.from_dict(BASE)
    data = copy.deepcopy(BASE)
    data["purpose"] = "New purpose"
    data["steps"][0]["instructions"] = "Gather all inputs."
    data["steps"].append({"id": "pack", "role": "Clerk", "instructions": "Pack.", "primitive": "deliver_package",
                          "config": {"title": "T"}, "inputs": ["step:approve"]})
    diff = spec_diff(old, WorkflowSpec.from_dict(data))
    assert diff == {"added": ["pack"], "removed": [], "changed": [{"step_id": "collect", "fields": ["instructions"]}],
                    "reordered": False, "meta_changed": ["purpose"]}


def test_removed_and_reordered():
    data = copy.deepcopy(BASE)
    data["steps"] = [data["steps"][1] | {"inputs": []}, data["steps"][0]]
    diff = spec_diff(WorkflowSpec.from_dict(BASE), WorkflowSpec.from_dict(data))
    assert diff["reordered"] is True and diff["changed"] == [{"step_id": "approve", "fields": ["inputs"]}]


def test_identical_specs_have_empty_diff():
    s = WorkflowSpec.from_dict(BASE)
    assert spec_diff(s, s) == {"added": [], "removed": [], "changed": [], "reordered": False, "meta_changed": []}


def test_removed_step_and_reorder_ignores_added_and_removed_ids():
    data = copy.deepcopy(BASE)
    data["steps"] = [dict(BASE["steps"][0], id="fresh"), data["steps"][0]]
    diff = spec_diff(WorkflowSpec.from_dict(BASE), WorkflowSpec.from_dict(data))
    assert diff["added"] == ["fresh"] and diff["removed"] == ["approve"]
    assert diff["reordered"] is False and diff["changed"] == []


def test_origin_is_ignored_and_changed_fields_are_sorted():
    data = copy.deepcopy(BASE)
    data["steps"][0]["origin"] = {"primitive_id": "collect_inputs", "primitive_version": 1, "edits": ["x"]}
    assert spec_diff(WorkflowSpec.from_dict(BASE), WorkflowSpec.from_dict(data))["changed"] == []
    data["steps"][0].update(role="Clerk", config={"a": 1}, guardrails=["citations"])
    diff = spec_diff(WorkflowSpec.from_dict(BASE), WorkflowSpec.from_dict(data))
    assert diff["changed"] == [{"step_id": "collect", "fields": ["config", "guardrails", "role"]}]


def test_meta_changes_reported_in_fixed_order():
    data = copy.deepcopy(BASE)
    data.update(completion_criteria=["Other"], domain="Other domain", deliverables=["X"])
    data["input_requirements"][0]["description"] = "Changed"
    diff = spec_diff(WorkflowSpec.from_dict(BASE), WorkflowSpec.from_dict(data))
    assert diff["meta_changed"] == ["domain", "input_requirements", "deliverables", "completion_criteria"]
