import copy

import pytest

from daoharness.primitives import builtin_catalog
from daoharness.spec import WorkflowSpec
from daoharness.validator import (
    ValidationContext, ValidationFailed, ValidationReport, validate_spec, workflow_metrics)


def _step(id, primitive, inputs, config=None, **extra):
    return {"id": id, "role": f"{id} role", "instructions": f"Do {id}.", "primitive": primitive,
            "config": {} if config is None else config, "inputs": list(inputs), **extra}


def base_dict():
    return {
        "domain": "Request handling", "purpose": "Turn an incoming packet into an approved reply.",
        "input_requirements": [{"name": "packet", "description": "The incoming packet", "required": True}],
        "deliverables": ["Approved reply"], "completion_criteria": ["Reply approved"],
        "steps": [
            _step("collect", "collect_inputs", ["input:packet"]),
            _step("draft", "draft_message", ["step:collect"],
                  {"channel": "email", "audience": "Requester", "external": True}),
            _step("approve", "human_approval", ["step:draft"],
                  approval={"approver_role": "Lead", "what_is_approved": "Outbound reply"}),
            _step("pack", "deliver_package", ["step:approve"], {"title": "Package"}),
        ],
    }


def ctx(**kw):
    kw.setdefault("approver_resolves", lambda role: True)
    return ValidationContext(builtin_catalog(), **kw)


def run(d, context=None, **kw):
    return validate_spec(WorkflowSpec.from_dict(d), context or ctx(), **kw)


def codes(report):
    return {e.split(":", 1)[0] for e in report.errors}


def step_of(d, sid):
    return next(s for s in d["steps"] if s["id"] == sid)


def insert_before_pack(d, step):
    d["steps"].insert(len(d["steps"]) - 1, step)
    return d


# --- baseline ---------------------------------------------------------------------------------------------------

def test_minimal_spec_is_valid():
    report = run(base_dict())
    assert isinstance(report, ValidationReport)
    assert report.ok and report.errors == () and report.gaps == ()


def test_error_format_is_code_step_detail():
    d = base_dict()
    step_of(d, "draft")["inputs"] = ["input:nope"]
    report = run(d)
    assert not report.ok
    assert "unknown_input:draft:nope" in report.errors


def test_validation_failed_carries_errors():
    exc = ValidationFailed(["a:x:1", "b:y:2"])
    assert exc.errors == ("a:x:1", "b:y:2")
    assert str(exc) == "a:x:1; b:y:2"


# --- rule 1, 2 --------------------------------------------------------------------------------------------------

def test_rule1_unknown_input():
    d = base_dict()
    step_of(d, "collect")["inputs"] = ["input:packet", "input:other"]
    assert "unknown_input" in codes(run(d))


@pytest.mark.parametrize("ref", ["step:approve", "step:draft", "step:ghost"])
def test_rule2_forward_self_or_unknown_step(ref):
    d = base_dict()
    step_of(d, "draft")["inputs"] = ["step:collect", ref]
    assert "forward_or_unknown_step" in codes(run(d))


# --- rule 3 -----------------------------------------------------------------------------------------------------

def test_rule3_unknown_catalog_primitive():
    d = base_dict()
    step_of(d, "draft")["primitive"] = "made_up_step"
    assert "unknown_primitive" in codes(run(d))


# --- rule 4 -----------------------------------------------------------------------------------------------------

def test_rule4_config_schema_violation():
    d = base_dict()
    step_of(d, "draft")["config"] = {"channel": "pigeon", "audience": "Requester", "external": True}
    assert "invalid_config" in codes(run(d))


def test_rule4_unexpected_config_key():
    d = base_dict()
    step_of(d, "collect")["config"] = {"surprise": 1}
    assert "invalid_config" in codes(run(d))


DUPLICATE_CASES = {
    "extract_fields": {"fields": [{"name": "a", "description": "A", "required": True},
                                  {"name": "a", "description": "A2", "required": False}]},
    "checklist_match": {"items": [{"id": "x", "label": "X", "description": "X"},
                                  {"id": "x", "label": "X2", "description": "X2"}]},
    "screen_score": {"criteria": [{"id": "c", "label": "C", "description": "C", "weight": 2},
                                  {"id": "c", "label": "C2", "description": "C2", "weight": 1}], "scale_max": 5},
    "classify_route": {"categories": [{"id": "a", "label": "A", "description": "A"},
                                      {"id": "a", "label": "B", "description": "B"}]},
    "generic_model_step": {"output_fields": [{"name": "o", "description": "O"}, {"name": "o", "description": "O2"}]},
}


@pytest.mark.parametrize("primitive", sorted(DUPLICATE_CASES))
def test_rule4_duplicate_ids_in_config_lists(primitive):
    config = DUPLICATE_CASES[primitive]
    d = insert_before_pack(base_dict(), _step("extra", primitive, ["step:collect"], config))
    report = run(d)
    assert any(e.startswith("invalid_config:extra:") for e in report.errors), report.errors


@pytest.mark.parametrize("primitive", sorted(DUPLICATE_CASES))
def test_rule4_unique_ids_pass(primitive):
    config = copy.deepcopy(DUPLICATE_CASES[primitive])
    key = next(iter(k for k in config if isinstance(config[k], list)))
    sub = "name" if "name" in config[key][1] else "id"
    config[key][1][sub] = "zz"
    d = insert_before_pack(base_dict(), _step("extra", primitive, ["step:collect"], config))
    assert run(d).ok


@pytest.mark.parametrize("reserved", ["none_apply", "route"])
def test_rule4_reserved_category_ids(reserved):
    config = {"categories": [{"id": "a", "label": "A", "description": "A"},
                             {"id": reserved, "label": "R", "description": "R"}]}
    d = insert_before_pack(base_dict(), _step("extra", "classify_route", ["step:collect"], config))
    report = run(d)
    assert any(e.startswith("invalid_config:extra:") and reserved in e for e in report.errors), report.errors


# --- rule 5 -----------------------------------------------------------------------------------------------------

def test_rule5_human_step_without_approval():
    d = base_dict()
    step_of(d, "approve")["approval"] = None
    assert "approval_mismatch" in codes(run(d))


def test_rule5_non_human_step_with_approval():
    d = base_dict()
    step_of(d, "draft")["approval"] = {"approver_role": "Lead", "what_is_approved": "Draft"}
    assert "approval_mismatch" in codes(run(d))


def test_rule5_approver_not_member():
    seen = []
    report = run(base_dict(), ctx(approver_resolves=lambda role: seen.append(role) or False))
    assert "approver_not_member" in codes(report)
    assert seen == ["Lead"]


def test_rule5_approval_without_content():
    d = base_dict()
    step_of(d, "approve")["inputs"] = []
    assert "approval_without_content" in codes(run(d))


# --- rule 6 -----------------------------------------------------------------------------------------------------

def test_rule6_external_draft_not_approved():
    d = base_dict()
    step_of(d, "approve")["inputs"] = ["step:collect"]
    assert "external_draft_requires_approval" in codes(run(d))


def test_rule6_internal_draft_needs_no_approval():
    d = base_dict()
    step_of(d, "draft")["config"]["external"] = False
    step_of(d, "approve")["inputs"] = ["step:collect"]
    assert run(d).ok


def test_rule6_approval_before_draft_does_not_count():
    d = base_dict()
    d["steps"] = [
        _step("collect", "collect_inputs", ["input:packet"]),
        _step("approve", "human_approval", ["step:collect"],
              approval={"approver_role": "Lead", "what_is_approved": "Packet"}),
        _step("draft", "draft_message", ["step:approve"],
              {"channel": "email", "audience": "Requester", "external": True}),
        _step("pack", "deliver_package", ["step:draft"], {"title": "Package"}),
    ]
    assert "external_draft_requires_approval" in codes(run(d))


# --- rule 8 -----------------------------------------------------------------------------------------------------

def test_rule8_a_source_this_engine_cannot_reach_is_a_capability_gap():
    for source in ("gmail", "web_search"):
        d = base_dict()
        step_of(d, "draft")["required_sources"] = [source]
        report = run(d)
        assert report.ok
        assert list(report.gaps) == [{"step_id": "draft", "kind": "needs_capability", "detail": source}]


def test_rule8_unknown_source_is_capability_gap():
    d = base_dict()
    step_of(d, "draft")["required_sources"] = ["fax"]
    report = run(d)
    assert report.ok
    assert list(report.gaps) == [{"step_id": "draft", "kind": "needs_capability", "detail": "fax"}]


def test_rule8_connected_source_has_no_gap():
    d = base_dict()
    step_of(d, "draft")["required_sources"] = ["gmail"]
    assert run(d, ctx(connected_sources=frozenset({"gmail"}))).gaps == ()


# --- rule 9, 10 -------------------------------------------------------------------------------------------------

def test_rule9_last_step_must_deliver():
    d = base_dict()
    d["steps"].pop()
    assert "missing_deliverable" in codes(run(d))


@pytest.mark.parametrize("sid", ["collect", "approve", "pack"])
def test_rule10_input_bearing_primitives_need_inputs(sid):
    d = base_dict()
    step_of(d, sid)["inputs"] = []
    if sid == "approve":
        step_of(d, "pack")["inputs"] = ["step:draft"]
    assert any(e.startswith(f"missing_inputs:{sid}:") for e in run(d).errors)


# --- metrics ----------------------------------------------------------------------------------------------------

def test_metrics_for_minimal_spec():
    m = workflow_metrics(WorkflowSpec.from_dict(base_dict()), builtin_catalog())
    assert m == {"total_steps": 4, "automatic_steps": 3, "automation_ratio": 0.75}
    assert run(base_dict()).metrics == m


def test_metrics_count_human_steps_as_not_automatic():
    d = base_dict()
    d["steps"] = [
        _step("collect", "collect_inputs", ["input:packet"]),
        _step("a1", "human_approval", ["step:collect"], approval={"approver_role": "Lead", "what_is_approved": "A"}),
        _step("a2", "human_approval", ["step:a1"], approval={"approver_role": "Lead", "what_is_approved": "B"}),
        _step("note", "generic_model_step", ["step:a2"], {"output_fields": [{"name": "o", "description": "O"}]}),
        _step("pack", "deliver_package", ["step:note"], {"title": "Package"}),
    ]
    m = workflow_metrics(WorkflowSpec.from_dict(d), builtin_catalog())
    assert m["automatic_steps"] == 3 and m["automation_ratio"] == 0.6


# --- review fixes -----------------------------------------------------------------------------------------------

def test_route_with_16_categories_is_invalid_config_and_15_is_ok():
    def route(n):
        cats = [{"id": f"c{i}", "label": f"C{i}", "description": f"C{i}"} for i in range(n)]
        return insert_before_pack(base_dict(), _step("extra", "classify_route", ["step:collect"], {"categories": cats}))
    assert any(e.startswith("invalid_config:extra:") for e in run(route(16)).errors)
    assert run(route(15)).ok


def test_non_human_step_with_approval_gets_only_mismatch():
    d = base_dict()
    step_of(d, "draft")["approval"] = {"approver_role": "Stranger", "what_is_approved": "Draft"}
    report = run(d, ctx(approver_resolves=lambda role: role == "Lead"))
    assert [e for e in report.errors if e.split(":", 1)[0] in {"approval_mismatch", "approver_not_member"}] == \
        ["approval_mismatch:draft:approval not allowed"]


def test_repeated_sources_give_one_gap_each():
    d = base_dict()
    step_of(d, "draft")["required_sources"] = ["gmail", "fax", "gmail", "fax"]
    assert list(run(d).gaps) == [{"step_id": "draft", "kind": "needs_capability", "detail": "gmail"},
                                 {"step_id": "draft", "kind": "needs_capability", "detail": "fax"}]
