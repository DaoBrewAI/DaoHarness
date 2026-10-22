import copy
import threading

import pytest

from daoharness.executor import (
    PAUSE_STATES, RUN_SCHEMA, STEP_RULES, ExecutorError, apply_guardrails, resolve_inputs)
from daoharness.primitives import builtin_catalog
from daoharness.spec import WorkflowSpec, digest
from workflow_fakes import FakeDecisionPort, FakeModel, executor_for

DOCS = "Bank statements attached. Photo ID too."


def _step(id, primitive, inputs, config=None, **extra):
    return {"id": id, "role": f"{id.title()} role", "instructions": f"Do the {id} step.", "primitive": primitive,
            "config": {} if config is None else config, "inputs": list(inputs), **extra}


def _spec(*steps):
    return {
        "domain": "Document review", "purpose": "Review supplied documents and package the result.",
        "input_requirements": [{"name": "documents", "description": "Document text", "required": True}],
        "deliverables": ["Review package"], "completion_criteria": ["Every field is extracted"],
        "steps": list(steps)}


AUTO = _spec(
    _step("collect", "collect_inputs", ["input:documents"]),
    _step("extract", "extract_fields", ["step:collect"],
          {"fields": [{"name": "headline", "description": "First statement", "required": True}]}),
    _step("summary", "summarize_report", ["step:extract"]),
    _step("package", "deliver_package", ["step:summary"], {"title": "Review package"}),
)

CHECKLIST = {"items": [{"id": "a", "label": "Item A", "description": "First item"},
                       {"id": "b", "label": "Item B", "description": "Second item"},
                       {"id": "x", "label": "Item X", "description": "Third item"}]}
ASSESS = _spec(
    _step("collect", "collect_inputs", ["input:documents"]),
    _step("assess", "checklist_match", ["input:documents"], CHECKLIST),
    _step("package", "deliver_package", ["step:assess"]),
)
CATEGORIES = [{"id": "a", "label": "A", "description": "Category A"},
              {"id": "b", "label": "B", "description": "Category B"}]
DRAFT = {"channel": "email", "audience": "reader", "external": True}


def _start(ex, spec_dict, run_id="i.r1", content=None):
    return ex.start(org_id="o", run_id=run_id, instance_id="i", version=1, spec=WorkflowSpec.from_dict(spec_dict),
                    input_payload={"source_kind": "supplied_input", "content": content or {"documents": DOCS}})


def _step_rec(run, step_id):
    return next(s for s in run["steps"] if s["step_id"] == step_id)


# --- core flow ---------------------------------------------------------------------------------------------------

def test_run_completes_and_outputs_flow_downstream(tmp_path):
    model = FakeModel()
    ex, store, _ = executor_for(tmp_path, model)
    spec = WorkflowSpec.from_dict(AUTO)
    ex.start(org_id="o", run_id="i.r1", instance_id="i", version=1, spec=spec,
             input_payload={"source_kind": "supplied_input", "content": {"documents": DOCS}})
    run = ex.advance("o", "i.r1")
    assert run["state"] == "completed" and [s["state"] for s in run["steps"]] == ["completed"] * 4
    summary_call = next(c for c in model.calls if c["input"]["step"]["id"] == "summary")
    assert "step:extract" in summary_call["input"]["inputs"]
    assert run["deliverable"]["items"][0]["step_id"] == "summary"
    assert run["model_calls"] == 2          # extract + summary; collect/package are deterministic
    assert store.read("o", "runs", "i.r1") == run


def test_run_record_shape(tmp_path):
    ex, _, _ = executor_for(tmp_path, FakeModel())
    started = _start(ex, AUTO)
    assert started["schema"] == RUN_SCHEMA and started["state"] == "running"
    assert started["spec_digest"] == WorkflowSpec.from_dict(AUTO).digest()
    assert started["spec"] == WorkflowSpec.from_dict(AUTO).to_dict()
    assert started["input"]["content"] == {"documents": DOCS}
    assert started["model_calls"] == started["decision_calls"] == started["human_interventions"] == 0
    assert started["code"] is None and started["missing_sources"] == [] and started["deliverable"] is None
    blank = started["steps"][1]
    assert blank["step_id"] == "extract" and blank["primitive"] == "extract_fields" and blank["role"] == "Extract role"
    assert blank["state"] == "pending" and blank["attempt"] == 0 and blank["output"] is None
    assert blank["checks"] == [] and blank["decisions"] == []
    assert blank["approval_id"] is None and blank["error"] is None and blank["started_at"] is None
    run = ex.advance("o", "i.r1")
    extract = _step_rec(run, "extract")
    assert extract["output_digest"] == digest(extract["output"])
    assert extract["receipt"]["provider"] == "fake" and extract["started_at"] and extract["finished_at"]
    assert run["completed_at"] and run["updated_at"]
    assert run["deliverable"] == _step_rec(run, "package")["output"]
    assert run["deliverable"]["title"] == "Review package"
    assert run["deliverable"]["items"][0]["role"] == "Summary role"


def test_model_step_call_contract(tmp_path):
    model = FakeModel()
    ex, _, _ = executor_for(tmp_path, model)
    _start(ex, AUTO)
    ex.advance("o", "i.r1")
    call = next(c for c in model.calls if c["input"]["step"]["id"] == "extract")
    assert call["operation"] == "step" and call["skill_id"] == "workflow-step.extract_fields"
    assert call["input"]["step"] == {"id": "extract", "role": "Extract role", "instructions": "Do the extract step.",
                                     "primitive": "extract_fields", "config": AUTO["steps"][1]["config"]}
    assert call["input"]["workflow"] == {"domain": "Document review", "purpose": AUTO["purpose"]}
    assert call["input"]["completion_criteria"] == AUTO["completion_criteria"]
    assert call["input"]["inputs"]["step:collect"] == {"packet": [{"ref": "input:documents", "text": DOCS}]}
    assert STEP_RULES in call["instructions"] and "Your role: Extract role." in call["instructions"]
    assert builtin_catalog().get("extract_fields").instructions in call["instructions"]
    assert call["evidence"] == []      # only this step's own input: refs are sent as evidence
    assert call["schema"] == builtin_catalog().get("extract_fields").output_schema_for(AUTO["steps"][1]["config"])


def test_guardrail_failure_pauses_needs_attention(tmp_path):
    model = FakeModel(step_overrides={"extract": lambda data: {"fields": [
        {"name": "headline", "found": True, "value": "", "source_quotes": []}]}})
    ex, _, _ = executor_for(tmp_path, model)
    _start(ex, AUTO)
    run = ex.advance("o", "i.r1")
    extract = _step_rec(run, "extract")
    assert extract["state"] == "needs_attention" and run["state"] == "needs_attention"
    assert run["code"] == "guardrail_failed"
    assert extract["error"] == {"code": "guardrail_failed", "checks": ["found_fields"]}
    assert [c["id"] for c in extract["checks"] if not c["passed"]] == ["found_fields", "citations"]
    assert [f["id"] for f in extract["flags"]] == ["citations"]     # advisory: kept as a flag, never the stop
    assert _step_rec(run, "summary")["state"] == "pending"
    assert model.count("step") == 1


def test_a_present_item_without_a_quote_is_flagged_not_stopped(tmp_path):
    port = FakeDecisionPort(answers={"a": {"evidence": ()}})
    ex, _, _ = executor_for(tmp_path, FakeModel(), decisions=port)
    _start(ex, ASSESS)
    run = ex.advance("o", "i.r1")
    assess = _step_rec(run, "assess")
    assert run["state"] == "completed" and assess["state"] == "completed"
    assert assess["flags"] == [{"id": "citations", "passed": False, "detail": "no quote for: a"}]


def test_schema_failure_stops_further_checks(tmp_path):
    model = FakeModel(step_overrides={"extract": lambda data: {"rows": []}})
    ex, _, _ = executor_for(tmp_path, model)
    _start(ex, AUTO)
    run = ex.advance("o", "i.r1")
    extract = _step_rec(run, "extract")
    assert [c["id"] for c in extract["checks"]] == ["schema"] and extract["checks"][0]["passed"] is False
    assert extract["error"]["checks"] == ["schema"]


def test_model_failure_marks_step_failed(tmp_path):
    model = FakeModel(fail_operations={"step"})
    ex, _, _ = executor_for(tmp_path, model)
    _start(ex, AUTO)
    run = ex.advance("o", "i.r1")
    assert run["state"] == "failed" and run["code"] == "provider_failed"
    assert _step_rec(run, "extract")["error"]["code"] == "provider_failed"
    assert run["model_calls"] == 1 and _step_rec(run, "collect")["state"] == "completed"


def test_resume_after_crash_uses_cache_and_skips_completed_steps(tmp_path):
    reference = executor_for(tmp_path / "ref", FakeModel())[0]
    _start(reference, AUTO)
    done = reference.advance("o", "i.r1")

    model_a = FakeModel()
    ex, store, _ = executor_for(tmp_path / "crash", model_a)
    _start(ex, AUTO)
    real_merge = ex._merge

    def crash(org_id, run, record, update):     # the process dies after summary's model call, before its merge
        if record["step_id"] == "summary":
            raise KeyboardInterrupt("simulated crash")
        return real_merge(org_id, run, record, update)
    ex._merge = crash
    with pytest.raises(KeyboardInterrupt):
        ex.advance("o", "i.r1")
    stored = store.read("o", "runs", "i.r1")
    assert _step_rec(stored, "extract")["state"] == "completed" and _step_rec(stored, "summary")["state"] == "running"
    assert stored["model_calls"] == 1 and model_a.count("step") == 2

    model_b = FakeModel()
    ex_b, _, _ = executor_for(tmp_path / "crash", model_b)
    run = ex_b.advance("o", "i.r1")
    assert model_b.count("step") == 0                       # summary is served from the cache, extract not re-run
    assert run["state"] == "completed" and run["model_calls"] == 2   # the summary call is counted once
    assert [s["output"] for s in run["steps"]] == [s["output"] for s in done["steps"]]
    assert run["deliverable"] == done["deliverable"]


def test_sensitive_data_masked_in_external_draft(tmp_path):
    spec = _spec(_step("collect", "collect_inputs", ["input:documents"]),
                 _step("draft", "draft_message", ["step:collect"], DRAFT),
                 _step("package", "deliver_package", ["step:draft"]))
    ex, _, _ = executor_for(tmp_path, FakeModel())
    _start(ex, spec)
    run = ex.advance("o", "i.r1")
    draft = _step_rec(run, "draft")
    assert "•••• 6789" in draft["output"]["body"] and "123-45-6789" not in draft["output"]["body"]
    ids = [c["id"] for c in draft["checks"]]
    assert "mask_sensitive" in ids and "draft_only" in ids
    assert all(c["passed"] for c in draft["checks"]) and run["state"] == "completed"
    assert "123-45-6789" not in str(run["deliverable"])


def test_start_is_idempotent(tmp_path):
    ex, store, _ = executor_for(tmp_path, FakeModel())
    first = _start(ex, AUTO)
    second = _start(ex, AUTO, content={"documents": "Different text."})
    assert second == first and store.read("o", "runs", "i.r1")["input"]["content"] == {"documents": DOCS}


def test_unknown_primitive_needs_capability(tmp_path):
    spec = _spec(_step("collect", "collect_inputs", ["input:documents"]),
                 _step("custom", "made_up_step", ["step:collect"]),
                 _step("package", "deliver_package", ["step:custom"]))
    ex, _, _ = executor_for(tmp_path, FakeModel())
    _start(ex, spec)
    run = ex.advance("o", "i.r1")
    assert run["state"] == "needs_attention" and run["code"] == "needs_capability"
    assert _step_rec(run, "custom")["state"] == "needs_attention"


def test_read_and_advance_unknown_run(tmp_path):
    ex, _, _ = executor_for(tmp_path, FakeModel())
    for call in (lambda: ex.read("o", "i.nope"), lambda: ex.advance("o", "i.nope")):
        with pytest.raises(ExecutorError) as info:
            call()
        assert info.value.code == "run_not_found" and info.value.status == 404


def test_advance_returns_paused_run_unchanged(tmp_path):
    model = FakeModel(fail_operations={"step"})
    ex, _, _ = executor_for(tmp_path, model)
    _start(ex, AUTO)
    failed = ex.advance("o", "i.r1")
    assert ex.advance("o", "i.r1") == failed and model.count("step") == 1
    assert "failed" in PAUSE_STATES and ex.read("o", "i.r1") == failed


# --- lock discipline ---------------------------------------------------------------------------------------------

def test_model_call_runs_outside_the_store_lock(tmp_path):
    acquired = []

    def probe(data):
        def grab():
            with store.locked("o"):
                acquired.append(True)
        t = threading.Thread(target=grab)
        t.start()
        t.join(timeout=5)
        return {"fields": [{"name": "headline", "found": True, "value": "Bank statements attached",
                            "source_quotes": ["Bank statements attached"]}]}

    ex, store, _ = executor_for(tmp_path, FakeModel(step_overrides={"extract": probe}))
    _start(ex, AUTO)
    assert ex.advance("o", "i.r1")["state"] == "completed"
    assert acquired == [True]


def test_result_dropped_when_run_leaves_running_during_call(tmp_path):
    def interfere(data):
        with store.locked("o"):
            run = store.read("o", "runs", "i.r1")
            run["state"] = "needs_attention"
            store.write("o", "runs", "i.r1", run)
        return {"fields": [{"name": "headline", "found": True, "value": "Bank", "source_quotes": ["Bank"]}]}

    model = FakeModel(step_overrides={"extract": interfere})
    ex, store, _ = executor_for(tmp_path, model)
    _start(ex, AUTO)
    run = ex.advance("o", "i.r1")
    assert run["state"] == "needs_attention"
    extract = _step_rec(run, "extract")
    assert extract["output"] is None and extract["state"] == "running" and run["model_calls"] == 0


def test_result_dropped_when_step_attempt_changes_during_call(tmp_path):
    calls = []

    def bump_once(data):
        calls.append(1)
        if len(calls) == 1:
            with store.locked("o"):
                run = store.read("o", "runs", "i.r1")
                run["steps"][1].update(state="pending", attempt=1)
                store.write("o", "runs", "i.r1", run)
        return {"fields": [{"name": "headline", "found": True, "value": f"v{len(calls)}",
                            "source_quotes": ["Bank statements attached"]}]}

    ex, store, _ = executor_for(tmp_path, FakeModel(step_overrides={"extract": bump_once}))
    _start(ex, AUTO)
    run = ex.advance("o", "i.r1")
    extract = _step_rec(run, "extract")
    assert run["state"] == "completed" and len(calls) == 2
    assert extract["attempt"] == 1 and extract["output"]["fields"][0]["value"] == "v2"
    assert run["model_calls"] == 2       # extract (attempt 1) + summary; the dropped call was never merged


# --- decisions ---------------------------------------------------------------------------------------------------

def test_a_plugged_in_decider_answers_decision_steps(tmp_path):
    port = FakeDecisionPort(answers={"b": {"choice": "missing"}, "x": {"choice": "unclear"}})
    model = FakeModel()
    ex, _, _ = executor_for(tmp_path, model, decisions=port)
    _start(ex, ASSESS)
    run = ex.advance("o", "i.r1")
    assess = _step_rec(run, "assess")
    assert run["state"] == "completed" and assess["state"] == "completed"
    assert [(i["id"], i["status"]) for i in assess["output"]["items"]] == [
        ("a", "present"), ("b", "missing"), ("x", "unclear")]
    assert [d["item_id"] for d in assess["decisions"]] == ["a", "b", "x"]
    assert all(d["rung"] == "provider" and d["accepted"] and d["provider"] == "fake-decider"
               and d["model"] == "fake-decider-1" and d["confidence"] == 0.95 for d in assess["decisions"])
    assert run["decision_calls"] == 3 and run["model_calls"] == 0
    assert model.count("decide") == 0 and len(port.calls) == 1
    assert port.calls[0]["rubric"] == builtin_catalog().get("checklist_match").rubric


def test_the_model_decides_every_item_of_a_step_in_one_call(tmp_path):
    many = {"items": [{"id": f"item_{n}", "label": f"Item {n}", "description": "An item"} for n in range(20)]}
    model = FakeModel()
    ex, _, _ = executor_for(tmp_path, model, decisions=False)
    _start(ex, _spec(_step("collect", "collect_inputs", ["input:documents"]),
                     _step("assess", "checklist_match", ["input:documents"], many),
                     _step("package", "deliver_package", ["step:assess"])))
    run = ex.advance("o", "i.r1")
    assert run["state"] == "completed" and model.count("decide") == 1
    assert run["model_calls"] == 1 and run["decision_calls"] == 20
    assert {d["rung"] for d in _step_rec(run, "assess")["decisions"]} == {"model"}


def test_an_answer_below_the_threshold_goes_to_a_person(tmp_path):
    port = FakeDecisionPort(answers={"x": {"confidence": 0.4}})
    model = FakeModel()
    ex, store, notifier = executor_for(tmp_path, model, decisions=port)
    _start(ex, ASSESS)
    run = ex.advance("o", "i.r1")
    assert run["state"] == "waiting_approval" and model.count("decide") == 0
    [review] = store.list("o", "approvals")
    assert review["kind"] == "decision_review" and [i["id"] for i in review["items"]] == ["x"]
    assert notifier.events == [("o", review["approval_id"])]
    ex.decide("o", review["approval_id"], actor="o", decision="choose", content_digest=review["content_digest"],
              choices={"x": "missing"})
    run = ex.advance("o", "i.r1")
    assess = _step_rec(run, "assess")
    assert run["state"] == "completed" and {i["id"]: i["status"] for i in assess["output"]["items"]}["x"] == "missing"
    assert [(d["rung"], d["accepted"]) for d in assess["decisions"] if d["item_id"] == "x"] == [
        ("provider", False), ("human", True)]


def test_a_decider_that_cannot_answer_sends_every_item_to_a_person(tmp_path):
    ex, store, _ = executor_for(tmp_path, FakeModel(), decisions=FakeDecisionPort(unavailable=True))
    _start(ex, ASSESS)
    run = ex.advance("o", "i.r1")
    assert run["state"] == "waiting_approval"
    [review] = store.list("o", "approvals")
    assert [i["id"] for i in review["items"]] == ["a", "b", "x"]
    assert all(d["error"] == "decider_unreachable" for d in _step_rec(run, "assess")["decisions"])


def _route(tmp_path, confidence, *, threshold=None, engine_threshold=None):
    config = {"categories": CATEGORIES}
    if threshold is not None:
        config["decision_threshold"] = threshold
    spec = _spec(_step("collect", "collect_inputs", ["input:documents"]),
                 _step("route", "classify_route", ["input:documents"], config),
                 _step("package", "deliver_package", ["step:route"]))
    port = FakeDecisionPort(answers={"route": {"choice": "a", "confidence": confidence}})
    ex, _, _ = executor_for(tmp_path, FakeModel(), decisions=port)
    if engine_threshold is not None:
        ex.decision_threshold = engine_threshold
    _start(ex, spec)
    run = ex.advance("o", "i.r1")
    return [d["accepted"] for d in _step_rec(run, "route")["decisions"]], run["state"]


def test_the_threshold_is_0_75_unless_the_step_or_the_engine_sets_another(tmp_path):
    assert _route(tmp_path / "a", 0.75) == ([True], "completed")
    assert _route(tmp_path / "b", 0.74) == ([False], "waiting_approval")
    assert _route(tmp_path / "c", 0.85, threshold=0.9) == ([False], "waiting_approval")
    assert _route(tmp_path / "d", 0.85, engine_threshold=0.9) == ([False], "waiting_approval")
    assert _route(tmp_path / "e", 0.92, threshold=0.9, engine_threshold=0.95) == ([True], "completed")


def test_decision_resume_uses_cache_without_new_calls(tmp_path):
    ex, store, _ = executor_for(tmp_path, FakeModel(), decisions=FakeDecisionPort(answers={"x": {"choice": "missing"}}))
    _start(ex, ASSESS)
    done = ex.advance("o", "i.r1")
    crashed = copy.deepcopy(done)
    crashed.update(state="running", deliverable=None, completed_at=None)
    for rec in crashed["steps"][1:]:
        rec.update(state="pending", output=None, output_digest=None, checks=[], decisions=[], finished_at=None)
    store.write("o", "runs", "i.r1", crashed)

    port_b, model_b = FakeDecisionPort(), FakeModel()
    ex_b, _, _ = executor_for(tmp_path, model_b, decisions=port_b)
    run = ex_b.advance("o", "i.r1")
    assert port_b.calls == [] and model_b.calls == []
    assert run["state"] == "completed"
    assert _step_rec(run, "assess")["output"] == _step_rec(done, "assess")["output"]
    assert _step_rec(run, "assess")["decisions"] == _step_rec(done, "assess")["decisions"]
    assert run["decision_calls"] == done["decision_calls"]


# --- draft rule checks -------------------------------------------------------------------------------------------

def _rule_spec(rules):
    return _spec(_step("collect", "collect_inputs", ["input:documents"]),
                 _step("draft", "draft_message", ["step:collect"], {**DRAFT, "rules": rules}),
                 _step("package", "deliver_package", ["step:draft"]))


def test_draft_rule_violation_needs_attention(tmp_path):
    port = FakeDecisionPort(answers={"rule_1": {"choice": "no", "confidence": 0.95}})
    ex, _, _ = executor_for(tmp_path, FakeModel(), decisions=port)
    _start(ex, _rule_spec(["Mentions a reply-by date"]))
    run = ex.advance("o", "i.r1")
    draft = _step_rec(run, "draft")
    assert draft["state"] == "needs_attention" and run["code"] == "guardrail_failed"
    failed = [c for c in draft["checks"] if not c["passed"]]
    assert [c["id"] for c in failed] == ["rule_compliance"] and "Mentions a reply-by date" in failed[0]["detail"]
    assert draft["error"]["checks"] == ["rule_compliance"]
    assert [(d["item_id"], d["rung"], d["choice"]) for d in draft["decisions"]] == [("rule_1", "provider", "no")]
    assert run["decision_calls"] == 1


def _reviewed_rule_spec(rules):
    return _spec(_step("collect", "collect_inputs", ["input:documents"]),
                 _step("draft", "draft_message", ["step:collect"], {**DRAFT, "rules": rules}),
                 _step("approval", "human_approval", ["step:draft"],
                       approval={"approver_role": "Reviewer", "what_is_approved": "Outbound email"}),
                 _step("package", "deliver_package", ["step:approval"]))


def test_draft_rule_violation_before_an_approval_goes_to_the_approver_flagged(tmp_path):
    # A draft that a person approves next is not stopped by a broken rule: the approver sees the draft with the
    # broken rule flagged and decides.
    port = FakeDecisionPort(answers={"rule_1": {"choice": "no", "confidence": 0.95}})
    ex, store, notifier = executor_for(tmp_path, FakeModel(), decisions=port)
    _start(ex, _reviewed_rule_spec(["Mentions a reply-by date", "Is polite"]))
    run = ex.advance("o", "i.r1")
    draft = _step_rec(run, "draft")
    assert run["state"] == "waiting_approval" and draft["state"] == "completed" and draft["error"] is None
    rule = next(c for c in draft["checks"] if c["id"] == "rule_compliance")
    assert rule["passed"] is False and rule["detail"] == "violated: Mentions a reply-by date"
    assert draft["flags"] == [rule]
    [approval] = [a for a in store.list("o", "approvals") if a["status"] == "pending"]
    assert approval["flags"] == [{"step_id": "draft", "role": "Draft role", "check": "rule_compliance",
                                  "detail": "violated: Mentions a reply-by date"}]
    assert "flags" not in approval["content"]           # the flag informs the approver; the digest binds the draft
    assert notifier.events == [("o", approval["approval_id"])]


def test_a_draft_read_again_after_its_approval_still_goes_to_the_approver_flagged(tmp_path):
    # The package reads step:draft next to step:approval. It runs after the approval and depends on it, so it reads
    # the approver's verdict (resolve_inputs), and the broken rule goes to the approver as a flag.
    port = FakeDecisionPort(answers={"rule_1": {"choice": "no", "confidence": 0.95}})
    ex, store, _ = executor_for(tmp_path, FakeModel(), decisions=port)
    spec = _reviewed_rule_spec(["Mentions a reply-by date"])
    spec["steps"][-1]["inputs"] = ["step:approval", "step:draft"]
    _start(ex, spec)
    run = ex.advance("o", "i.r1")
    draft = _step_rec(run, "draft")
    assert run["state"] == "waiting_approval" and draft["state"] == "completed" and draft["error"] is None
    [approval] = store.list("o", "approvals")
    assert [f["detail"] for f in approval["flags"]] == ["violated: Mentions a reply-by date"]


def test_a_draft_that_also_feeds_a_step_outside_its_approval_still_stops_on_a_broken_rule(tmp_path):
    # A step that reads step:draft without depending on the approval would never see an approver's fix, so the
    # broken rule still stops the run.
    port = FakeDecisionPort(answers={"rule_1": {"choice": "no", "confidence": 0.95}})
    ex, store, _ = executor_for(tmp_path, FakeModel(), decisions=port)
    spec = _reviewed_rule_spec(["Mentions a reply-by date"])
    steps = spec["steps"]
    steps.insert(2, _step("notes", "summarize_report", ["step:draft"], {"sections": ["Notes"]}))
    steps[-1]["inputs"] = ["step:approval", "step:notes"]
    _start(ex, spec)
    run = ex.advance("o", "i.r1")
    draft = _step_rec(run, "draft")
    assert draft["state"] == "needs_attention" and draft["error"]["checks"] == ["rule_compliance"]
    assert draft["flags"] == [] and store.list("o", "approvals") == []


def test_unsettled_draft_rule_before_an_approval_is_flagged_too(tmp_path):
    port = FakeDecisionPort(answers={"rule_1": {"confidence": 0.5}})
    ex, store, _ = executor_for(tmp_path, FakeModel(), decisions=port)
    _start(ex, _reviewed_rule_spec(["Mentions a reply-by date"]))
    run = ex.advance("o", "i.r1")
    assert run["state"] == "waiting_approval" and _step_rec(run, "draft")["state"] == "completed"
    [approval] = store.list("o", "approvals")
    assert [f["detail"] for f in approval["flags"]] == ["unsettled: Mentions a reply-by date"]


def test_clean_draft_before_an_approval_has_no_flags(tmp_path):
    ex, store, _ = executor_for(tmp_path, FakeModel(), decisions=FakeDecisionPort())
    _start(ex, _reviewed_rule_spec(["Is polite"]))
    run = ex.advance("o", "i.r1")
    assert _step_rec(run, "draft")["flags"] == []
    [approval] = store.list("o", "approvals")
    assert approval["flags"] == []


def test_draft_rules_pass_and_see_only_masked_text(tmp_path):
    port = FakeDecisionPort()
    ex, _, _ = executor_for(tmp_path, FakeModel(), decisions=port)
    _start(ex, _rule_spec(["Is polite", "Asks for the remaining items"]))
    run = ex.advance("o", "i.r1")
    draft = _step_rec(run, "draft")
    assert run["state"] == "completed"
    assert next(c for c in draft["checks"] if c["id"] == "rule_compliance")["passed"] is True
    contexts = [i.context for call in port.calls for i in call["items"]]
    assert len(contexts) == 2 and all("•••• 6789" in c and "123-45-6789" not in c for c in contexts)
    assert run["decision_calls"] == 2


def test_draft_rule_unsettled_fails_without_inbox_item(tmp_path):
    port = FakeDecisionPort(answers={"rule_1": {"confidence": 0.5}})
    ex, store, notifier = executor_for(tmp_path, FakeModel(), decisions=port)
    _start(ex, _rule_spec(["Mentions a reply-by date"]))
    run = ex.advance("o", "i.r1")
    assert run["state"] == "needs_attention" and _step_rec(run, "draft")["error"]["checks"] == ["rule_compliance"]
    assert store.list("o", "approvals") == [] and notifier.events == []
    assert run["model_calls"] == 1     # the draft itself; the decider answered the rule


def test_draft_rule_outcomes_are_cached(tmp_path):
    ex, store, _ = executor_for(tmp_path, FakeModel())
    _start(ex, _rule_spec(["Is polite"]))
    done = ex.advance("o", "i.r1")
    crashed = copy.deepcopy(done)
    crashed.update(state="running", deliverable=None, completed_at=None)
    for rec in crashed["steps"][1:]:
        rec.update(state="pending", output=None, checks=[], decisions=[])
    store.write("o", "runs", "i.r1", crashed)
    port_b, model_b = FakeDecisionPort(), FakeModel()
    ex_b, _, _ = executor_for(tmp_path, model_b, decisions=port_b)
    run = ex_b.advance("o", "i.r1")
    assert run["state"] == "completed" and port_b.calls == [] and model_b.calls == []
    assert _step_rec(run, "draft")["decisions"] == _step_rec(done, "draft")["decisions"]


# --- helpers -----------------------------------------------------------------------------------------------------

def test_resolve_inputs_reads_run_input_and_step_outputs():
    spec = WorkflowSpec.from_dict(AUTO)
    run = {"input": {"content": {"documents": "D"}},
           "steps": [{"step_id": "collect", "output": {"packet": []}}, {"step_id": "extract", "output": None}]}
    assert resolve_inputs(run, spec, spec.step("collect")) == {"input:documents": "D"}
    assert resolve_inputs(run, spec, spec.step("extract")) == {"step:collect": {"packet": []}}
    run["input"]["content"] = {}
    assert resolve_inputs(run, spec, spec.step("collect")) == {"input:documents": ""}


def test_apply_guardrails_orders_checks_for_external_draft():
    spec = WorkflowSpec.from_dict(_rule_spec([]))
    step = spec.step("draft")
    primitive = builtin_catalog().get("draft_message")
    output = {"subject": "Hi", "body": "Number 123-45-6789", "notes": ""}
    masked, checks = apply_guardrails(primitive, step, output, {"step:collect": {"packet": []}}, spec)
    assert [c.id for c in checks] == ["schema", "required_fields", "mask_sensitive", "draft_only"]
    assert masked["body"] == "Number •••• 6789" and output["body"] == "Number 123-45-6789"


def test_summary_citations_may_name_steps_upstream_of_its_inputs():
    # The summary may cite the draft that reached it through the approval step.
    spec = WorkflowSpec.from_dict({
        "domain": "D", "purpose": "P", "deliverables": ["X"], "completion_criteria": ["C1"],
        "input_requirements": [{"name": "documents", "description": "d", "required": True}],
        "steps": [
            {"id": "collect", "role": "r", "instructions": "i", "primitive": "collect_inputs",
             "inputs": ["input:documents"]},
            {"id": "draft", "role": "r", "instructions": "i", "primitive": "draft_message",
             "config": {"channel": "email", "audience": "a", "external": True}, "inputs": ["step:collect"]},
            {"id": "approval", "role": "r", "instructions": "i", "primitive": "human_approval", "inputs": ["step:draft"],
             "approval": {"approver_role": "Lead", "what_is_approved": "Email"}},
            {"id": "summary", "role": "r", "instructions": "i", "primitive": "summarize_report",
             "config": {"sections": []}, "inputs": ["step:approval"]},
            {"id": "package", "role": "r", "instructions": "i", "primitive": "deliver_package",
             "config": {"title": "T"}, "inputs": ["step:summary"]}]})
    output = {"title": "t", "summary": "s", "criteria": [{"criterion": "C1", "met": True, "evidence": "e"}],
              "sections": [{"title": "x", "body": "y", "source_steps": ["step:draft", "step:approval", "step:package"]}]}
    fixed, checks = apply_guardrails(builtin_catalog().get("summarize_report"), spec.step("summary"), output,
                                     {"step:approval": {"content": {}}}, spec)
    assert fixed["sections"][0]["source_steps"] == ["draft", "approval"]
    assert all(c.passed for c in checks)
