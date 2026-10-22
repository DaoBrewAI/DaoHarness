import copy

from daoharness.service import capability_result
from daoharness.spec import WorkflowSpec
from workflow_fakes import FakeModel, executor_for

DOCS = "Bank statements attached. Photo ID too."
RUN = "i.r1"


def _step(id, primitive, inputs, config=None, **extra):
    return {"id": id, "role": f"{id.title()} role", "instructions": f"Do the {id} step.", "primitive": primitive,
            "config": {} if config is None else config, "inputs": list(inputs), **extra}


def _spec(*steps, input_name="documents"):
    return {"domain": "Document review", "purpose": "Review supplied documents.",
            "input_requirements": [{"name": input_name, "description": "Text", "required": True}],
            "deliverables": ["Package"], "completion_criteria": ["Every field is extracted"], "steps": list(steps)}


FIELDS = {"fields": [{"name": "headline", "description": "First statement", "required": True}]}
AUTO = _spec(_step("collect", "collect_inputs", ["input:documents"]),
             _step("extract", "extract_fields", ["step:collect"], FIELDS),
             _step("summary", "summarize_report", ["step:extract"]),
             _step("package", "deliver_package", ["step:summary"]))


def _start(ex, spec_dict):
    return ex.start(org_id="o", run_id=RUN, instance_id="i", version=1, spec=WorkflowSpec.from_dict(spec_dict),
                    input_payload={"source_kind": "supplied_input", "content": {"documents": DOCS}})


def _rec(run, step_id):
    return next(s for s in run["steps"] if s["step_id"] == step_id)


def _run(tmp_path, spec_dict, model=None, **kw):
    model = model or FakeModel()
    ex, store, notifier = executor_for(tmp_path, model, **kw)
    _start(ex, spec_dict)
    return ex, store, model, ex.advance("o", RUN)


# --- sources, caps, versions --------------------------------------------------------------------------------------

def test_a_source_this_engine_cannot_reach_stops_the_step_as_needs_capability(tmp_path):
    spec = copy.deepcopy(AUTO)
    spec["steps"][1]["required_sources"] = ["gmail"]
    ex, store, model, run = _run(tmp_path, spec)
    assert run["state"] == "needs_attention" and run["code"] == "needs_capability"
    assert _rec(run, "extract")["error"] == {"code": "needs_capability", "detail": "gmail"} and model.calls == []
    assert _rec(run, "collect")["state"] == "completed"


def test_web_search_is_a_capability_this_engine_does_not_have(tmp_path):
    spec = copy.deepcopy(AUTO)
    spec["steps"][2]["required_sources"] = ["web_search"]
    _, _, model, run = _run(tmp_path, spec)
    assert run["code"] == "needs_capability" and _rec(run, "summary")["error"]["detail"] == "web_search"
    assert model.count("step", "summary") == 0


def test_a_gap_declared_for_a_step_stops_the_run_there(tmp_path):
    model = FakeModel()
    ex, _, _ = executor_for(tmp_path, model)
    ex.start(org_id="o", run_id=RUN, instance_id="i", version=1, spec=WorkflowSpec.from_dict(AUTO),
             input_payload={"source_kind": "supplied_input", "content": {"documents": DOCS}},
             blocked_steps={"summary": [{"kind": "needs_capability", "detail": "a calendar"}]})
    run = ex.advance("o", RUN)
    assert run["code"] == "needs_capability" and _rec(run, "summary")["error"]["detail"] == "a calendar"
    assert _rec(run, "extract")["state"] == "completed"


def test_a_gap_declared_for_the_delivery_does_not_stop_it(tmp_path):
    # Models list what happens after delivery (signing, sending) as a gap on the last step
    model = FakeModel()
    ex, _, _ = executor_for(tmp_path, model)
    ex.start(org_id="o", run_id=RUN, instance_id="i", version=1, spec=WorkflowSpec.from_dict(AUTO),
             input_payload={"source_kind": "supplied_input", "content": {"documents": DOCS}},
             blocked_steps={"package": [{"kind": "needs_capability", "detail": "e-signature of the lease"}]})
    run = ex.advance("o", RUN)
    assert run["state"] == "completed" and _rec(run, "package")["state"] == "completed"
    assert "Outside this workflow: e-signature of the lease" in capability_result(run)["limitations"]


def test_a_gap_declared_on_the_approval_still_lets_the_person_decide(tmp_path):
    spec = _spec(_step("collect", "collect_inputs", ["input:documents"]),
                 _step("extract", "extract_fields", ["step:collect"], FIELDS),
                 _step("approve", "human_approval", ["step:extract"],
                       approval={"approver_role": "Property Manager", "what_is_approved": "The screening"}),
                 _step("package", "deliver_package", ["step:approve"]))
    model = FakeModel()
    ex, _, _ = executor_for(tmp_path, model)
    ex.start(org_id="o", run_id=RUN, instance_id="i", version=1, spec=WorkflowSpec.from_dict(spec),
             input_payload={"source_kind": "supplied_input", "content": {"documents": DOCS}},
             blocked_steps={"approve": [{"kind": "needs_capability", "detail": "sending the lease to the tenant"}]})
    run = ex.advance("o", RUN)
    assert run["state"] == "waiting_approval" and _rec(run, "approve")["state"] == "waiting_approval"


def test_model_call_cap_pauses_needs_attention(tmp_path):
    _, _, model, run = _run(tmp_path, AUTO, cap=1)
    assert _rec(run, "extract")["state"] == "completed"
    assert _rec(run, "summary")["state"] == "needs_attention"
    assert run["state"] == "needs_attention" and run["code"] == "model_call_cap" and model.count("step") == 1


def test_decision_call_cap_pauses_needs_attention(tmp_path):
    items = {"items": [{"id": n, "label": n, "description": n} for n in ("a", "b", "c")]}
    spec = _spec(_step("assess", "checklist_match", ["input:documents"], items),
                 _step("route", "classify_route", ["input:documents"], {"categories": [
                     {"id": "a", "label": "A", "description": "A"}, {"id": "b", "label": "B", "description": "B"}]}),
                 _step("package", "deliver_package", ["step:assess", "step:route"]))
    model = FakeModel()
    ex, _, _ = executor_for(tmp_path, model)
    ex.decision_call_cap = 3
    _start(ex, spec)
    run = ex.advance("o", RUN)
    assert _rec(run, "assess")["state"] == "completed" and run["decision_calls"] == 3
    assert run["state"] == "needs_attention" and run["code"] == "decision_call_cap"


def test_builtin_primitive_version_is_not_checked(tmp_path):
    spec = copy.deepcopy(AUTO)
    spec["steps"][1]["primitive_version"] = 7
    assert _run(tmp_path, spec)[3]["state"] == "completed"
