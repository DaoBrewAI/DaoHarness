"""Single-flight, caps, cache keys, unexpected errors and crash windows."""
import logging
import threading

import pytest

from daoharness.executor import ExecutorError
from daoharness.spec import WorkflowSpec
from industry_fixtures import ACCOUNTING
from workflow_fakes import FakeDecisionPort, FakeModel, executor_for

DOCS = "Bank statements attached. Photo ID too."
RUN = "i.r1"
GOOD = {"fields": [{"name": "headline", "found": True, "value": "Bank statements attached",
                    "source_quotes": ["Bank statements attached"]}]}


def _step(id, primitive, inputs, config=None, **extra):
    return {"id": id, "role": f"{id.title()} role", "instructions": f"Do the {id} step.", "primitive": primitive,
            "config": {} if config is None else config, "inputs": list(inputs), **extra}


def _spec(*steps, inputs=("documents",)):
    return {"domain": "Document review", "purpose": "Review supplied documents.",
            "input_requirements": [{"name": n, "description": "Text", "required": True} for n in inputs],
            "deliverables": ["Package"], "completion_criteria": ["Every field is extracted"], "steps": list(steps)}


FIELDS = {"fields": [{"name": "headline", "description": "First statement", "required": True}]}
ITEM_X = {"items": [{"id": "x", "label": "X", "description": "An item"}]}
AUTO = _spec(_step("collect", "collect_inputs", ["input:documents"]),
             _step("extract", "extract_fields", ["step:collect"], FIELDS),
             _step("summary", "summarize_report", ["step:extract"]),
             _step("package", "deliver_package", ["step:summary"]))


def _start(ex, spec, run_id=RUN, content=None):
    return ex.start(org_id="o", run_id=run_id, instance_id="i", version=1, spec=WorkflowSpec.from_dict(spec),
                    input_payload={"source_kind": "supplied_input", "content": content or {"documents": DOCS}})


def _rec(run, sid):
    return next(s for s in run["steps"] if s["step_id"] == sid)


def _crash_on_merge(ex, step_id):
    real = ex._merge

    def crash(org_id, run, record, update):
        if record["step_id"] == step_id:
            raise KeyboardInterrupt("simulated crash")
        return real(org_id, run, record, update)
    ex._merge = crash


# 1. single flight ------------------------------------------------------------------------------------------------

def test_concurrent_advance_is_single_flight(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def extract(data):
        entered.set()
        assert release.wait(5)
        return GOOD

    model = FakeModel(step_overrides={"extract": extract})
    ex, store, _ = executor_for(tmp_path, model)
    _start(ex, AUTO)
    results = []
    worker = threading.Thread(target=lambda: results.append(ex.advance("o", RUN)))
    worker.start()
    assert entered.wait(5)
    second = ex.advance("o", RUN)                       # returns at once, executes nothing
    assert second["state"] == "running" and _rec(second, "extract")["state"] == "running"
    release.set()
    worker.join(10)
    run = store.read("o", "runs", RUN)
    assert results and results[0]["state"] == "completed" and run["state"] == "completed"
    assert model.count("step", "extract") == 1 and run["model_calls"] == model.count("step") == 2


def test_single_flight_is_released_after_errors(tmp_path):
    model = FakeModel(fail_operations={"step"})
    ex, _, _ = executor_for(tmp_path, model)
    _start(ex, AUTO)
    assert ex.advance("o", RUN)["state"] == "failed"
    model.fail_operations.clear()
    ex.retry("o", RUN)
    assert ex.advance("o", RUN)["state"] == "completed"
    _start(ex, AUTO, run_id="i.r2")
    real = ex._execute_step
    ex._execute_step = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt("crash"))
    with pytest.raises(KeyboardInterrupt):
        ex.advance("o", "i.r2")
    ex._execute_step = real
    assert ex.advance("o", "i.r2")["state"] == "completed"   # the same process can resume after a crash


# 2. model_call_cap also bounds decision steps --------------------------------------------------------------------

def test_model_cap_stops_decision_step_before_the_router(tmp_path):
    spec = _spec(_step("collect", "collect_inputs", ["input:documents"]),
                 _step("extract", "extract_fields", ["step:collect"], FIELDS),
                 _step("assess", "checklist_match", ["input:documents"], ITEM_X),
                 _step("package", "deliver_package", ["step:extract", "step:assess"]))
    port, model = FakeDecisionPort(answers={"x": {"confidence": 0.4}}), FakeModel()
    ex, _, _ = executor_for(tmp_path, model, cap=1, decisions=port)
    _start(ex, spec)
    run = ex.advance("o", RUN)
    assert run["state"] == "needs_attention" and run["code"] == "model_call_cap"
    assert _rec(run, "assess")["state"] == "needs_attention"
    assert model.count("decide") == 0 and port.calls == [] and run["model_calls"] == 1


def test_model_cap_stops_draft_rule_checks(tmp_path):
    spec = _spec(_step("collect", "collect_inputs", ["input:documents"]),
                 _step("draft", "draft_message", ["step:collect"],
                       {"channel": "email", "audience": "reader", "external": True, "rules": ["Is polite"]}),
                 _step("package", "deliver_package", ["step:draft"]))
    port, model = FakeDecisionPort(), FakeModel()
    ex, _, _ = executor_for(tmp_path, model, cap=1, decisions=port)
    _start(ex, spec)
    run = ex.advance("o", RUN)
    assert run["state"] == "needs_attention" and run["code"] == "model_call_cap"
    assert port.calls == [] and model.count("step") == 1 and run["model_calls"] == 1


# 3. model-step evidence and cache key ----------------------------------------------------------------------------

TWO_INPUTS = _spec(_step("extract", "extract_fields", ["input:documents"], FIELDS),
                   _step("package", "deliver_package", ["step:extract"]), inputs=("documents", "notes"))


def test_model_step_sends_only_referenced_inputs_as_evidence(tmp_path):
    model = FakeModel()
    ex, _, _ = executor_for(tmp_path, model)
    _start(ex, TWO_INPUTS, content={"documents": DOCS, "notes": "Private notes."})
    ex.advance("o", RUN)
    assert model.calls[0]["evidence"] == [{"source_id": "input:documents", "text": DOCS}]


def test_model_step_cache_key_tracks_the_evidence_sent(tmp_path):
    model = FakeModel()
    ex, _, _ = executor_for(tmp_path, model)
    _start(ex, TWO_INPUTS, "i.r1", {"documents": DOCS, "notes": "Run one notes."})
    _start(ex, TWO_INPUTS, "i.r2", {"documents": DOCS, "notes": "Different run two notes."})
    _start(ex, TWO_INPUTS, "i.r3", {"documents": "Other documents entirely.", "notes": "Run one notes."})
    for run_id in ("i.r1", "i.r2", "i.r3"):
        assert ex.advance("o", run_id)["state"] == "completed"
    # r2 sends exactly what r1 sent (unreferenced notes are never sent), so it reuses r1's answer; r3 does not
    assert model.count("step") == 2
    assert [c["evidence"][0]["text"] for c in model.calls] == [DOCS, "Other documents entirely."]


# 4. unexpected exceptions ----------------------------------------------------------------------------------------

def test_unexpected_step_exception_fails_the_run_and_retry_works(tmp_path, caplog):
    calls = []

    def boom(data):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("provider timeout with secret detail")
        return GOOD

    ex, store, _ = executor_for(tmp_path, FakeModel(step_overrides={"extract": boom}))
    _start(ex, AUTO)
    with caplog.at_level(logging.WARNING):
        run = ex.advance("o", RUN)
    assert run["state"] == "failed" and run["code"] == "step_error"
    assert _rec(run, "extract")["state"] == "failed" and _rec(run, "extract")["error"] == {"code": "step_error"}
    assert "secret detail" not in str(store.read("o", "runs", RUN)) and "secret detail" not in caplog.text
    ex.retry("o", RUN)
    assert ex.advance("o", RUN)["state"] == "completed"


def test_model_error_keeps_its_own_code(tmp_path):
    ex, _, _ = executor_for(tmp_path, FakeModel(fail_operations={"step"}))
    _start(ex, AUTO)
    assert ex.advance("o", RUN)["code"] == "provider_failed"


# 6a. crash after a call but before the merge still counts the call -----------------------------------------------

def test_crash_after_model_call_counts_the_call_once(tmp_path):
    model = FakeModel()
    ex, store, _ = executor_for(tmp_path, model)
    _start(ex, AUTO)
    _crash_on_merge(ex, "extract")
    with pytest.raises(KeyboardInterrupt):
        ex.advance("o", RUN)
    ex2, _, _ = executor_for(tmp_path, model)
    run = ex2.advance("o", RUN)
    assert model.count("step") == 2 and run["model_calls"] == 2 and run["state"] == "completed"


def test_crash_after_decision_call_counts_the_calls_once(tmp_path):
    spec = _spec(_step("assess", "checklist_match", ["input:documents"], ITEM_X),
                 _step("package", "deliver_package", ["step:assess"]))
    model = FakeModel()
    ex, _, _ = executor_for(tmp_path, model, decisions=False)
    _start(ex, spec)
    _crash_on_merge(ex, "assess")
    with pytest.raises(KeyboardInterrupt):
        ex.advance("o", RUN)
    ex2, _, _ = executor_for(tmp_path, model, decisions=False)
    run = ex2.advance("o", RUN)
    assert model.count("decide") == 1
    assert run["decision_calls"] == 1 and run["model_calls"] == 1 and run["state"] == "completed"


def test_cache_reuse_after_counting_adds_nothing(tmp_path):
    model = FakeModel()
    ex, _, _ = executor_for(tmp_path, model)
    _start(ex, AUTO, "i.r1")
    _start(ex, AUTO, "i.r2")
    assert ex.advance("o", "i.r1")["model_calls"] == 2
    assert ex.advance("o", "i.r2")["model_calls"] == 0      # identical request: served from cache, counted by r1
    assert model.count("step") == 2


# 6b. retry of a later step is clamped to the first unfinished step -----------------------------------------------

def test_retry_later_step_is_clamped_to_first_unfinished_step(tmp_path):
    model = FakeModel(step_overrides={"extract": lambda d: {"fields": [
        {"name": "headline", "found": True, "value": "", "source_quotes": []}]}})
    ex, _, _ = executor_for(tmp_path, model)
    _start(ex, AUTO)
    assert ex.advance("o", RUN)["state"] == "needs_attention"
    model.step_overrides.pop("extract")
    retried = ex.retry("o", RUN, step_id="summary")
    assert _rec(retried, "extract")["attempt"] == 1 and _rec(retried, "summary")["attempt"] == 0
    run = ex.advance("o", RUN)
    assert run["state"] == "completed" and model.count("step", "extract") == 2


# 6c. decide crash window -----------------------------------------------------------------------------------------

def _waiting(tmp_path, notifier_error=False):
    ex, store, notifier = executor_for(tmp_path, FakeModel())
    if notifier_error:
        def fail(org_id, approval):
            raise RuntimeError("inbox down with approval content")
        notifier.approval_requested = fail
    ex.start(org_id="o", run_id=RUN, instance_id="i", version=1, spec=WorkflowSpec.from_dict(ACCOUNTING["spec"]),
             input_payload={"source_kind": "generated_sample", "content": ACCOUNTING["sample"]})
    run = ex.advance("o", RUN)
    return ex, store, run


def _crash_run_write(store):
    real = store.write

    def write(org, coll, key, value):
        if coll == "runs":
            raise KeyboardInterrupt("crash")
        return real(org, coll, key, value)
    store.write = write
    return real


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_decide_replays_run_transition_after_crash(tmp_path, decision):
    ex, store, _ = _waiting(tmp_path)
    [approval] = [a for a in store.list("o", "approvals") if a["status"] == "pending"]
    real = _crash_run_write(store)
    with pytest.raises(KeyboardInterrupt):
        ex.decide("o", approval["approval_id"], actor="o", decision=decision,
                  content_digest=approval["content_digest"])
    store.write = real
    assert store.read("o", "runs", RUN)["state"] == "waiting_approval"
    run = ex.decide("o", approval["approval_id"], actor="o", decision=decision,
                    content_digest=approval["content_digest"])
    assert run["human_interventions"] == 1
    if decision == "approve":
        assert run["state"] == "running" and _rec(run, "approval")["state"] == "completed"
        assert ex.advance("o", RUN)["state"] == "completed"
    else:
        assert run["state"] == "rejected"
    # once the run moved on, the same call is closed again
    with pytest.raises(ExecutorError) as info:
        ex.decide("o", approval["approval_id"], actor="o", decision=decision,
                  content_digest=approval["content_digest"])
    assert info.value.code == "approval_closed"


def test_decide_replay_requires_the_same_decision(tmp_path):
    ex, store, _ = _waiting(tmp_path)
    [approval] = [a for a in store.list("o", "approvals") if a["status"] == "pending"]
    real = _crash_run_write(store)
    with pytest.raises(KeyboardInterrupt):
        ex.decide("o", approval["approval_id"], actor="o", decision="approve",
                  content_digest=approval["content_digest"])
    store.write = real
    with pytest.raises(ExecutorError) as info:
        ex.decide("o", approval["approval_id"], actor="o", decision="reject",
                  content_digest=approval["content_digest"])
    assert info.value.code == "approval_closed"


# 6d. notifier failures never block the approval ------------------------------------------------------------------

def test_notifier_failure_does_not_block_approval(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        ex, store, run = _waiting(tmp_path, notifier_error=True)
    assert run["state"] == "waiting_approval"
    [approval] = [a for a in store.list("o", "approvals") if a["status"] == "pending"]
    assert _rec(run, "approval")["approval_id"] == approval["approval_id"]
    assert "approval content" not in caplog.text and "notif" in caplog.text.lower()
