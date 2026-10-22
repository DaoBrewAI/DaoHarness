"""Validator rules for a per-item segment, and the fan_out compile metric."""
import copy

import pytest

from daoharness.primitives import builtin_catalog
from daoharness.spec import WorkflowSpec
from daoharness.validator import ValidationContext, validate_spec, workflow_metrics

FIELDS = {"fields": [{"name": "title", "description": "Title", "required": True}]}
CATEGORIES = {"categories": [{"id": "keep", "label": "Keep", "description": "Keep it"},
                             {"id": "drop", "label": "Drop", "description": "Drop it"}]}
APPROVAL = {"approver_role": "Lead", "what_is_approved": "Outbound replies"}


def _step(id, primitive, inputs, config=None, **extra):
    return {"id": id, "role": f"{id} role", "instructions": f"Do {id}.", "primitive": primitive,
            "config": {} if config is None else config, "inputs": list(inputs), **extra}


def fan_out_dict():
    """Nine steps, the same shape as the design example; 8 of 9 automatic."""
    return {
        "domain": "Packet review", "purpose": "Review every item in a packet and reply to each.",
        "input_requirements": [{"name": "packet", "description": "Several items", "required": True},
                               {"name": "brief", "description": "Shared brief", "required": True}],
        "deliverables": ["Review"], "completion_criteria": ["Every item is reviewed"],
        "steps": [
            _step("collect", "collect_inputs", ["input:packet", "input:brief"]),
            _step("split", "split_items", ["input:packet"], {"item_noun": "item", "max_items": 12}),
            _step("extract", "extract_fields", ["step:split"], FIELDS, for_each="split"),
            _step("route", "classify_route", ["step:split", "input:brief"], CATEGORIES, for_each="split"),
            _step("check", "generate_questions", ["step:extract"], {"max_questions": 2, "focus": "gaps"}, for_each="split"),
            _step("outreach", "draft_message", ["step:extract", "step:route"],
                  {"channel": "email", "audience": "reader", "external": True}, for_each="split"),
            _step("approval", "human_approval", ["step:outreach"], approval=APPROVAL),
            _step("summary", "summarize_report", ["step:route", "step:approval"], {"sections": ["Items"]}),
            _step("package", "deliver_package", ["step:summary", "step:approval"], {"title": "Review"}),
        ]}


def run(d):
    return validate_spec(WorkflowSpec.from_dict(d), ValidationContext(builtin_catalog(), lambda role: True))


def codes(report):
    return [e.split(":", 1)[0] for e in report.errors]


def step_of(d, sid):
    return next(s for s in d["steps"] if s["id"] == sid)


def test_fan_out_spec_is_valid_and_automation_unchanged():
    report = run(fan_out_dict())
    assert report.ok, report.errors
    assert report.metrics["total_steps"] == 9 and report.metrics["automatic_steps"] == 8
    assert report.metrics["automation_ratio"] == 0.89
    assert report.metrics["fan_out"] == {"split_step": "split",
                                         "per_item_steps": ["extract", "route", "check", "outreach"],
                                         "max_items": 12}


def test_fan_out_max_items_defaults_to_twenty():
    d = fan_out_dict()
    step_of(d, "split")["config"] = {"item_noun": "item"}
    assert run(d).metrics["fan_out"]["max_items"] == 20


def test_single_case_metrics_have_no_fan_out():
    d = fan_out_dict()
    d["steps"] = [s for s in d["steps"] if s["id"] != "split"]
    for s in d["steps"]:
        s.pop("for_each", None)
        s["inputs"] = ["input:packet" if r == "step:split" else r for r in s["inputs"]]
    report = run(d)
    assert report.ok, report.errors
    assert "fan_out" not in report.metrics
    assert "fan_out" not in workflow_metrics(WorkflowSpec.from_dict(d), builtin_catalog())


@pytest.mark.parametrize("target", ["nowhere", "collect", "outreach"])
def test_invalid_for_each(target):
    d = fan_out_dict()
    step_of(d, "extract")["for_each"] = target
    assert "invalid_for_each" in codes(run(d))
    assert any(e.startswith("invalid_for_each:extract:") for e in run(d).errors)


def test_multiple_splits():
    d = fan_out_dict()
    d["steps"].insert(2, _step("split_again", "split_items", ["input:packet"], {"item_noun": "item"}))
    assert "multiple_splits" in codes(run(d))


def test_segment_not_contiguous():
    d = fan_out_dict()
    step_of(d, "check").pop("for_each")
    assert "segment_not_contiguous" in codes(run(d))


def test_segment_must_start_right_after_the_split():
    d = fan_out_dict()
    step_of(d, "extract").pop("for_each")
    report = run(d)
    assert "segment_not_contiguous" in codes(report)


def test_split_without_items():
    d = fan_out_dict()
    for s in d["steps"]:
        s.pop("for_each", None)
    assert "split_without_items" in codes(run(d))


@pytest.mark.parametrize("make", [
    lambda: _step("gate", "human_approval", ["step:extract"], approval=APPROVAL, for_each="split"),
    lambda: _step("gate", "deliver_package", ["step:extract"], {"title": "x"}, for_each="split"),
    lambda: _step("gate", "split_items", ["input:packet"], {"item_noun": "x"}, for_each="split"),
])
def test_not_per_item(make):
    d = fan_out_dict()
    d["steps"].insert(3, make())
    assert any(e.startswith("not_per_item:gate:") for e in run(d).errors)


def test_segment_too_long():
    d = fan_out_dict()
    extra = [_step(f"more_{n}", "generate_questions", ["step:extract"], {"max_questions": 2, "focus": "gaps"}, for_each="split")
             for n in range(5)]
    d["steps"][3:3] = extra  # 4 + 5 = 9 per-item steps
    assert "segment_too_long" in codes(run(d))


@pytest.mark.parametrize("inputs", [["step:collect"], ["input:packet", "input:brief"], []])
def test_split_input_needs_exactly_one_input_ref(inputs):
    d = fan_out_dict()
    step_of(d, "split")["inputs"] = inputs
    assert "split_input" in codes(run(d))


def test_split_may_also_read_shared_context():
    d = fan_out_dict()
    step_of(d, "split")["inputs"] = ["input:packet", "step:collect"]
    assert run(d).ok


def test_external_per_item_draft_still_requires_later_approval():
    d = fan_out_dict()
    step_of(d, "approval")["inputs"] = ["step:route"]
    assert "external_draft_requires_approval" in codes(run(d))


def test_single_case_errors_unchanged_by_the_new_rules():
    d = copy.deepcopy(fan_out_dict())
    d["steps"] = [s for s in d["steps"] if s["id"] in ("collect", "package")]
    d["steps"][-1]["inputs"] = ["step:collect"]
    assert run(d).ok


@pytest.mark.parametrize("inputs", [["input:brief"], ["step:collect"], ["step:split", "input:packet"],
                                    ["input:packet"]])
def test_per_item_step_reads_its_own_item_not_the_packet(inputs):
    # A per-item step reading the whole packet would see every item: the cross-item mix-up the split exists to
    # prevent.
    d = fan_out_dict()
    step_of(d, "extract")["inputs"] = inputs
    assert any(e.startswith("per_item_input:extract:") for e in run(d).errors)


def test_per_item_step_may_read_an_earlier_per_item_step_and_shared_inputs():
    d = fan_out_dict()
    step_of(d, "check")["inputs"] = ["step:extract", "input:brief", "step:collect"]
    assert run(d).ok, run(d).errors
