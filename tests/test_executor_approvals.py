import copy

import pytest

from daoharness.executor import ExecutorError, readable_inputs
from daoharness.ports import ModelError
from daoharness.spec import WorkflowSpec, digest
from industry_fixtures import ACCOUNTING
from workflow_fakes import FakeDecisionPort, FakeModel, executor_for

RUN = "i.r1"


def _start(ex, spec_dict=None, content=None):
    spec = WorkflowSpec.from_dict(spec_dict or ACCOUNTING["spec"])
    return ex.start(org_id="o", run_id=RUN, instance_id="i", version=1, spec=spec,
                    input_payload={"source_kind": "generated_sample", "content": content or ACCOUNTING["sample"]})


def _rec(run, step_id):
    return next(s for s in run["steps"] if s["step_id"] == step_id)


def _pending(store):
    return [a for a in store.list("o", "approvals") if a["status"] == "pending"]


def _waiting(tmp_path, model=None, **kw):
    model = model or FakeModel()
    ex, store, notifier = executor_for(tmp_path, model, **kw)
    _start(ex)
    run = ex.advance("o", RUN)
    return ex, store, notifier, model, run


def _expect(code, status, fn):
    with pytest.raises(ExecutorError) as info:
        fn()
    assert (info.value.code, info.value.status) == (code, status)


# --- human approval ----------------------------------------------------------------------------------------------

def test_run_pauses_at_approval_and_notifies(tmp_path):
    ex, store, notifier, _, run = _waiting(tmp_path)
    assert run["state"] == "waiting_approval" and run["code"] is None
    [approval] = _pending(store)
    assert approval["assignee"] == "o" and approval["kind"] == "approval"
    assert set(approval["content"]) == {"step:outreach"}
    body = approval["content"]["step:outreach"]["body"]
    assert "•••• 6789" in body and "123-45-6789" not in body
    assert approval["content_digest"] == digest(approval["content"])
    assert approval["run_id"] == RUN and approval["instance_id"] == "i" and approval["version"] == 1
    assert approval["step_id"] == "approval" and approval["approver_role"] == "Engagement partner"
    assert approval["what_is_approved"] == "Outbound email"
    assert approval["workflow_domain"] == "Accounting client intake" and approval["created_at"]
    assert approval["decision"] is None and approval["decided_by"] is None and approval["edited_content"] is None
    assert len(approval["approval_id"]) == 32
    assert notifier.events == [("o", approval["approval_id"])]
    assert _rec(run, "approval")["state"] == "waiting_approval"
    assert _rec(run, "approval")["approval_id"] == approval["approval_id"]
    assert _rec(run, "summary")["state"] == "pending" and _rec(run, "package")["state"] == "pending"
    assert ex.advance("o", RUN) == run      # paused: nothing more runs


def test_regular_approvals_carry_kind_approval(tmp_path):
    _, store, _, _, _ = _waiting(tmp_path)
    assert [a["kind"] for a in store.list("o", "approvals")] == ["approval"]


def test_approve_continues_to_deliverable(tmp_path):
    ex, store, _, _, _ = _waiting(tmp_path)
    [approval] = _pending(store)
    decided = ex.decide("o", approval["approval_id"], actor="o", decision="approve",
                        content_digest=approval["content_digest"], note="Looks good")
    assert decided["state"] == "running" and decided["code"] is None
    step = _rec(decided, "approval")
    assert step["state"] == "completed" and step["finished_at"]
    assert step["checks"] == [{"id": "human_approval", "passed": True, "detail": "approve by o"}]
    assert step["output"] == {"decision": "approved", "approver": "o", "approved_content_digest":
                              approval["content_digest"], "edited": False, "content": approval["content"],
                              "note": "Looks good"}
    assert step["output_digest"] == digest(step["output"])
    run = ex.advance("o", RUN)
    assert run["state"] == "completed"
    assert "approval" in [i["step_id"] for i in run["deliverable"]["items"]]
    assert run["human_interventions"] == 1
    stored = store.read("o", "approvals", approval["approval_id"])
    assert stored["status"] == "approved" and stored["decision"] == "approve" and stored["decided_by"] == "o"
    assert stored["decided_at"] and stored["note"] == "Looks good"


def test_edit_then_approve_uses_edited_content_downstream(tmp_path):
    ex, store, _, _, _ = _waiting(tmp_path)
    [approval] = _pending(store)
    edited = copy.deepcopy(approval["content"])
    edited["step:outreach"]["body"] = "Please send the bank statements. Ref 987-65-4321."
    ex.decide("o", approval["approval_id"], actor="o", decision="edit",
              content_digest=approval["content_digest"], edited_content=edited)
    run = ex.advance("o", RUN)
    assert run["state"] == "completed" and run["human_interventions"] == 2
    item = next(i for i in run["deliverable"]["items"] if i["step_id"] == "approval")
    assert item["content"]["edited"] is True
    assert item["content"]["content"]["step:outreach"]["body"] == "Please send the bank statements. Ref •••• 4321."
    assert "987-65-4321" not in str(run) and "987-65-4321" not in str(store.list("o", "approvals"))
    step = _rec(run, "approval")
    assert step["output"]["approved_content_digest"] == digest(step["output"]["content"])
    assert step["checks"][0]["detail"] == "edit by o"


def _reading_the_draft(summary_reads_it=False):
    spec = copy.deepcopy(ACCOUNTING["spec"])
    steps = {step["id"]: step for step in spec["steps"]}
    steps["package"]["inputs"] = ["step:summary", "step:approval", "step:outreach"]
    if summary_reads_it:
        steps["summary"]["inputs"].append("step:outreach")
    return spec


def test_a_later_step_reading_the_draft_directly_gets_the_approved_edit(tmp_path):
    # An approver's edit replaces the draft for every step that runs after the approval and depends on it, also when
    # that step reads step:outreach itself rather than the approval's output.
    model = FakeModel()
    ex, store, _ = executor_for(tmp_path, model)
    _start(ex, _reading_the_draft(summary_reads_it=True))
    ex.advance("o", RUN)
    [approval] = _pending(store)
    original = copy.deepcopy(approval["content"]["step:outreach"])
    edited = copy.deepcopy(approval["content"])
    edited["step:outreach"]["body"] = "Please send the bank statements. Ref 987-65-4321."
    ex.decide("o", approval["approval_id"], actor="o", decision="edit",
              content_digest=approval["content_digest"], edited_content=edited)
    run = ex.advance("o", RUN)
    assert run["state"] == "completed"
    package = next(i for i in run["deliverable"]["items"] if i["step_id"] == "outreach")["content"]
    assert package["body"] == "Please send the bank statements. Ref •••• 4321."
    assert _rec(run, "outreach")["output"] == original          # the draft's own record is untouched
    # summary runs after the approval but does not depend on it: it reads the draft as drafted
    [summary] = [c for c in model.calls if c["operation"] == "step" and c["input"]["step"]["id"] == "summary"]
    assert summary["input"]["inputs"]["step:outreach"] == original


def test_a_later_step_reading_an_approved_draft_gets_it_unchanged(tmp_path):
    ex, store, _ = executor_for(tmp_path, FakeModel())
    _start(ex, _reading_the_draft())
    ex.advance("o", RUN)
    [approval] = _pending(store)
    ex.decide("o", approval["approval_id"], actor="o", decision="approve", content_digest=approval["content_digest"])
    run = ex.advance("o", RUN)
    package = next(i for i in run["deliverable"]["items"] if i["step_id"] == "outreach")["content"]
    assert package == approval["content"]["step:outreach"] == _rec(run, "outreach")["output"]


def test_reject_ends_run(tmp_path):
    ex, store, _, _, _ = _waiting(tmp_path)
    [approval] = _pending(store)
    run = ex.decide("o", approval["approval_id"], actor="o", decision="reject",
                    content_digest=approval["content_digest"], note="Wrong tone")
    assert run["state"] == "rejected" and run["code"] == "approval_rejected" and run["completed_at"]
    assert _rec(run, "approval")["state"] == "rejected" and run["human_interventions"] == 1
    assert store.read("o", "approvals", approval["approval_id"])["status"] == "rejected"
    assert ex.advance("o", RUN) == run
    _expect("run_not_retryable", 409, lambda: ex.retry("o", RUN))


def test_wrong_digest_is_stale_and_wrong_actor_forbidden(tmp_path):
    ex, store, _, _, _ = _waiting(tmp_path)
    [approval] = _pending(store)
    aid, good = approval["approval_id"], approval["content_digest"]
    _expect("approval_stale", 409,
            lambda: ex.decide("o", aid, actor="o", decision="approve", content_digest="sha256:" + "0" * 64))
    _expect("approval_forbidden", 403,
            lambda: ex.decide("o", aid, actor="someone-else", decision="approve", content_digest=good))
    _expect("approval_not_found", 404,
            lambda: ex.decide("o", "f" * 32, actor="o", decision="approve", content_digest=good))
    _expect("approval_not_found", 404,
            lambda: ex.decide("o", "../x", actor="o", decision="approve", content_digest=good))
    _expect("invalid_decision", 400,
            lambda: ex.decide("o", aid, actor="o", decision="maybe", content_digest=good))
    assert store.read("o", "approvals", aid)["status"] == "pending"


def test_edit_requires_same_keys(tmp_path):
    ex, store, _, _, _ = _waiting(tmp_path)
    [approval] = _pending(store)
    aid, good = approval["approval_id"], approval["content_digest"]
    for bad in (None, "text", {}, {"step:outreach": {}, "extra": 1}, {"other": {}}):
        _expect("invalid_edit", 400, lambda bad=bad: ex.decide(
            "o", aid, actor="o", decision="edit", content_digest=good, edited_content=bad))
    assert ex.read("o", RUN)["human_interventions"] == 0


def test_approval_decision_is_not_repeatable(tmp_path):
    ex, store, _, _, _ = _waiting(tmp_path)
    [approval] = _pending(store)
    ex.decide("o", approval["approval_id"], actor="o", decision="approve", content_digest=approval["content_digest"])
    _expect("approval_closed", 409, lambda: ex.decide(
        "o", approval["approval_id"], actor="o", decision="approve", content_digest=approval["content_digest"]))


def test_approval_is_stale_when_run_moved_on(tmp_path):
    ex, store, _, _, _ = _waiting(tmp_path)
    [approval] = _pending(store)
    run = store.read("o", "runs", RUN)
    run["state"] = "needs_attention"
    store.write("o", "runs", RUN, run)
    _expect("approval_stale", 409, lambda: ex.decide(
        "o", approval["approval_id"], actor="o", decision="approve", content_digest=approval["content_digest"]))


def test_approver_not_member_needs_attention(tmp_path):
    ex, store, notifier = executor_for(tmp_path, FakeModel())
    ex.membership = type("Nobody", (), {"resolve": lambda self, org, role: None})()
    _start(ex)
    run = ex.advance("o", RUN)
    assert run["state"] == "needs_attention" and run["code"] == "approver_not_member"
    assert store.list("o", "approvals") == [] and notifier.events == []


def test_crash_after_approval_write_reuses_it_without_renotifying(tmp_path):
    ex, store, notifier, _, run = _waiting(tmp_path)
    [approval] = _pending(store)
    crashed = copy.deepcopy(run)
    crashed["state"] = "running"
    _rec(crashed, "approval").update(state="running", approval_id=None)
    store.write("o", "runs", RUN, crashed)
    again = ex.advance("o", RUN)
    assert again["state"] == "waiting_approval"
    assert _rec(again, "approval")["approval_id"] == approval["approval_id"]
    assert notifier.events == [("o", approval["approval_id"])] and len(store.list("o", "approvals")) == 1


# --- retry -------------------------------------------------------------------------------------------------------

def test_retry_upstream_supersedes_pending_approval_and_requires_reapproval(tmp_path):
    ex, store, notifier, model, _ = _waiting(tmp_path)
    [old] = _pending(store)
    model.step_overrides["outreach"] = lambda data: {"subject": "Revised", "body": "Revised body text.", "notes": ""}
    retried = ex.retry("o", RUN, step_id="outreach")
    assert retried["state"] == "running" and retried["code"] is None and retried["human_interventions"] == 1
    assert _rec(retried, "outreach")["attempt"] == 1 and _rec(retried, "outreach")["state"] == "pending"
    assert _rec(retried, "approval")["attempt"] == 0 and _rec(retried, "approval")["approval_id"] is None
    assert store.read("o", "approvals", old["approval_id"])["status"] == "superseded"
    run = ex.advance("o", RUN)
    assert run["state"] == "waiting_approval"
    [new] = _pending(store)
    assert new["approval_id"] != old["approval_id"] and new["content"]["step:outreach"]["body"] == "Revised body text."
    assert notifier.events == [("o", old["approval_id"]), ("o", new["approval_id"])]
    _expect("approval_closed", 409, lambda: ex.decide(
        "o", old["approval_id"], actor="o", decision="approve", content_digest=old["content_digest"]))
    ex.decide("o", new["approval_id"], actor="o", decision="approve", content_digest=new["content_digest"])
    assert ex.advance("o", RUN)["state"] == "completed"


def test_retry_with_unchanged_content_opens_a_fresh_approval(tmp_path):
    ex, store, notifier, _, _ = _waiting(tmp_path)
    [old] = _pending(store)
    ex.retry("o", RUN, step_id="approval")
    run = ex.advance("o", RUN)
    assert run["state"] == "waiting_approval"
    [new] = _pending(store)
    assert new["content_digest"] == old["content_digest"] and new["approval_id"] != old["approval_id"]
    assert store.read("o", "approvals", old["approval_id"])["status"] == "superseded"
    assert _rec(run, "approval")["approval_id"] == new["approval_id"] and len(notifier.events) == 2
    ex.decide("o", new["approval_id"], actor="o", decision="approve", content_digest=new["content_digest"])
    assert ex.advance("o", RUN)["state"] == "completed"


def test_retry_failed_step_reruns_only_that_step(tmp_path):
    calls = []

    def flaky(data):
        calls.append(1)
        if len(calls) == 1:
            raise ModelError("provider_failed")
        return {"fields": [{"name": f["name"], "found": True, "value": "Northwind LLC",
                            "source_quotes": ["Prior-year return for Northwind LLC attached"]}
                           for f in data["step"]["config"]["fields"]]}

    model = FakeModel(step_overrides={"extract": flaky})
    ex, store, _ = executor_for(tmp_path, model)
    _start(ex)
    failed = ex.advance("o", RUN)
    assert failed["state"] == "failed" and failed["code"] == "provider_failed"
    collect_before = copy.deepcopy(_rec(failed, "collect"))
    retried = ex.retry("o", RUN)
    assert _rec(retried, "extract")["attempt"] == 1 and _rec(retried, "collect") == collect_before
    assert retried["missing_sources"] == []
    run = ex.advance("o", RUN)
    assert run["state"] == "waiting_approval" and len(calls) == 2
    [approval] = _pending(store)
    ex.decide("o", approval["approval_id"], actor="o", decision="approve", content_digest=approval["content_digest"])
    run = ex.advance("o", RUN)
    assert run["state"] == "completed" and _rec(run, "collect") == collect_before
    assert run["human_interventions"] == 2      # the retry and the approval


def test_every_retry_counts_as_an_intervention(tmp_path):
    ex, store, notifier, model, _ = _waiting(tmp_path)
    counts = []
    for _ in range(3):
        counts.append(ex.retry("o", RUN, step_id="outreach")["human_interventions"])
        assert ex.advance("o", RUN)["state"] == "waiting_approval"
    assert counts == [1, 2, 3]
    assert _rec(ex.read("o", RUN), "outreach")["attempt"] == 3


def test_retry_rules(tmp_path):
    ex, _, _ = executor_for(tmp_path, FakeModel())
    _expect("run_not_found", 404, lambda: ex.retry("o", RUN))
    _start(ex)
    _expect("run_not_retryable", 409, lambda: ex.retry("o", RUN))          # running
    ex.advance("o", RUN)
    _expect("step_not_found", 404, lambda: ex.retry("o", RUN, step_id="nope"))
    run = ex.retry("o", RUN)                                                 # default: first non-completed
    assert _rec(run, "approval")["attempt"] == 1 and _rec(run, "outreach")["attempt"] == 0


# --- decision review ---------------------------------------------------------------------------------------------

def test_readable_inputs_shows_what_a_person_needs_to_decide():
    # The reviewer reads the inputs as titled sections, without field names, blank values or internal markers.
    record = "Client record: Avery Rowan, a self-employed designer, asks for tax preparation."
    inputs = {
        "step:collect_client_intake": {"packet": [{"ref": "input:client_documents", "text": record}]},
        "step:extract_identity": {"fields": [
            {"name": "legal_name", "value": "Avery Rowan", "found": True, "source_quotes": [record]},
            {"name": "owner_name", "value": "", "found": False, "source_quotes": []},
        ]},
        "input:notes": "  ",
    }
    assert readable_inputs(inputs) == (
        "From step: collect client intake\n"
        "client documents:\n" + record + "\n\n"
        "From step: extract identity\n"
        "legal name: Avery Rowan\n"
        "owner name: not found")


def test_readable_inputs_renders_other_shapes_and_keeps_unseen_quotes():
    inputs = {
        "input:request": "Please review the contract.",
        "step:score": {"score": 3, "flags": ["late", "unsigned"], "detail": {"reason": "No signature page"}},
        "step:extract": {"fields": [{"name": "party", "value": "Acme", "source_quotes": ["Party: Acme Ltd"]}]},
    }
    assert readable_inputs(inputs) == (
        "Input: request\nPlease review the contract.\n\n"
        "From step: score\nscore: 3\nflags:\n  - late\n  - unsigned\ndetail:\n  reason: No signature page\n\n"
        "From step: extract\nparty: Acme\n  source quote: Party: Acme Ltd")


UNVERIFIED = {"evidence_quotes": ["this sentence is not in the documents at all"]}


def _review(tmp_path, port_answers=None, decide_overrides=None, spec_dict=None):
    port = FakeDecisionPort(answers=port_answers or {"bank_statements": {"confidence": 0.4}})
    model = FakeModel(decide_overrides=decide_overrides or {"bank_statements": UNVERIFIED})
    ex, store, notifier = executor_for(tmp_path, model, decisions=port)
    _start(ex, spec_dict)
    run = ex.advance("o", RUN)
    return ex, store, notifier, run


def test_unsettled_decisions_create_decision_review_and_choose_completes_step(tmp_path):
    ex, store, notifier, run = _review(tmp_path)
    assert run["state"] == "waiting_approval"
    assess = _rec(run, "assess")
    assert assess["state"] == "waiting_approval" and assess["approval_id"]
    [review] = _pending(store)
    assert review["kind"] == "decision_review" and review["assignee"] == "o"
    assert review["approver_role"] == "decision reviewer"
    assert review["what_is_approved"] == "1 decision(s) the model was not sure of"
    [item] = review["items"]
    assert review["content"] == {"items": review["items"]} and review["content_digest"] == digest(review["content"])
    assert item["id"] == "bank_statements" and item["options"] == ["present", "missing", "unclear"]
    assert item["question"].startswith("Do the inputs show this item?")
    excerpt = item["context_excerpt"]
    assert 0 < len(excerpt) <= 1500 and "[[next input]]" not in excerpt
    assert ACCOUNTING["sample"]["documents"][:200] in excerpt
    assert [(c["rung"], c["provider"], c["confidence"]) for c in item["candidates"]] == [
        ("provider", "fake-decider", 0.4)]
    assert set(item["candidates"][0]) == {"rung", "provider", "choice", "confidence"}
    assert notifier.events == [("o", review["approval_id"])]

    decided = ex.decide("o", review["approval_id"], actor="o", decision="choose",
                        content_digest=review["content_digest"], choices={"bank_statements": "missing"})
    assess = _rec(decided, "assess")
    assert assess["state"] == "completed" and decided["state"] == "running"
    statuses = {i["id"]: i["status"] for i in assess["output"]["items"]}
    assert statuses["bank_statements"] == "missing"
    assert {k: v for k, v in statuses.items() if k != "bank_statements"} == {
        "prior_year_return": "present", "photo_id": "present", "engagement_letter": "present",
        "payroll_reports": "present", "prior_year_financials": "present"}
    assert next(i for i in assess["output"]["items"] if i["id"] == "bank_statements")["note"] == "human"
    assert "pending_outcomes" not in assess
    human = [d for d in assess["decisions"] if d["rung"] == "human"]
    assert human == [{"item_id": "bank_statements", "rung": "human", "provider": "human", "model": None,
                      "choice": "missing", "confidence": None, "accepted": True}]
    assert decided["human_interventions"] == 1
    stored = store.read("o", "approvals", review["approval_id"])
    assert stored["status"] == "approved" and stored["decision"] == "choose"
    assert stored["choices"] == {"bank_statements": "missing"}
    assert ex.advance("o", RUN)["state"] == "waiting_approval"     # continues to the outbound-email approval


def test_choose_rejects_partial_or_invalid_choices_and_approve_on_decision_review(tmp_path):
    ex, store, _, run = _review(tmp_path, port_answers={"bank_statements": {"confidence": 0.4},
                                                        "photo_id": {"confidence": 0.4}},
                                decide_overrides={"bank_statements": UNVERIFIED, "photo_id": UNVERIFIED})
    [review] = _pending(store)
    assert [i["id"] for i in review["items"]] == ["bank_statements", "photo_id"]
    aid, good = review["approval_id"], review["content_digest"]
    for bad in (None, {}, {"bank_statements": "missing"}, {"bank_statements": "missing", "photo_id": "bogus"},
                {"bank_statements": "missing", "photo_id": "present", "extra": "present"}):
        _expect("invalid_choices", 400, lambda bad=bad: ex.decide(
            "o", aid, actor="o", decision="choose", content_digest=good, choices=bad))
    for decision in ("approve", "edit"):
        _expect("invalid_decision", 400, lambda decision=decision: ex.decide(
            "o", aid, actor="o", decision=decision, content_digest=good, edited_content=review["content"]))
    assert store.read("o", "approvals", aid)["status"] == "pending"
    rejected = ex.decide("o", aid, actor="o", decision="reject", content_digest=good)
    assert rejected["state"] == "rejected" and _rec(rejected, "assess")["state"] == "rejected"


def test_choose_is_invalid_on_regular_approval(tmp_path):
    ex, store, _, _, _ = _waiting(tmp_path)
    [approval] = _pending(store)
    _expect("invalid_decision", 400, lambda: ex.decide(
        "o", approval["approval_id"], actor="o", decision="choose", content_digest=approval["content_digest"],
        choices={}))


def test_choose_that_fails_guardrail_needs_attention(tmp_path):
    spec = {**ACCOUNTING["spec"], "steps": ACCOUNTING["spec"]["steps"][:4] + [
        {"id": "package", "role": "Coordinator", "instructions": "Package.", "primitive": "deliver_package",
         "config": {}, "inputs": ["step:route"]}]}
    ex, store, _, run = _review(tmp_path, port_answers={"route": {"confidence": 0.3}},
                                decide_overrides={"route": UNVERIFIED}, spec_dict=spec)
    [review] = _pending(store)
    assert review["items"][0]["options"][-1] == "none_apply"
    run = ex.decide("o", review["approval_id"], actor="o", decision="choose",
                    content_digest=review["content_digest"], choices={"route": "none_apply"})
    route = _rec(run, "route")
    assert route["state"] == "needs_attention" and run["state"] == "needs_attention"
    assert run["code"] == "guardrail_failed" and route["error"]["checks"] == ["category"]
    assert route["output"]["decided_by"] == "human" and run["human_interventions"] == 1


def test_decision_review_without_reviewer_needs_attention(tmp_path):
    port = FakeDecisionPort(answers={"bank_statements": {"confidence": 0.4}})
    ex, store, _ = executor_for(tmp_path, FakeModel(decide_overrides={"bank_statements": UNVERIFIED}),
                                decisions=port)
    ex.membership = type("Nobody", (), {"resolve": lambda self, org, role: None})()
    _start(ex)
    run = ex.advance("o", RUN)
    assert run["state"] == "needs_attention" and run["code"] == "approver_not_member"
    assert store.list("o", "approvals") == []


def test_decision_review_survives_resume_without_new_calls(tmp_path):
    ex, store, notifier, run = _review(tmp_path)
    [review] = _pending(store)
    crashed = copy.deepcopy(run)
    crashed["state"] = "running"
    _rec(crashed, "assess").update(state="running", approval_id=None)
    store.write("o", "runs", RUN, crashed)
    port_b, model_b = FakeDecisionPort(), FakeModel()
    ex_b, _, _ = executor_for(tmp_path, model_b, decisions=port_b)
    again = ex_b.advance("o", RUN)
    assert port_b.calls == [] and model_b.calls == []
    assert _rec(again, "assess")["approval_id"] == review["approval_id"] and len(_pending(store)) == 1


def test_one_unreadable_model_answer_asks_a_person_about_that_item_only(tmp_path):
    model = FakeModel(decide_overrides={"bank_statements": {"choice": "no idea"},
                                        "photo_id": {"choice": "Missing", "confidence": "92%"}})
    ex, store, _ = executor_for(tmp_path, model, decisions=False)
    _start(ex)
    run = ex.advance("o", RUN)
    assert run["state"] == "waiting_approval" and model.count("decide") == 1
    [review] = _pending(store)
    assert review["kind"] == "decision_review" and [i["id"] for i in review["items"]] == ["bank_statements"]
    assess = _rec(run, "assess")
    by_item = {d["item_id"]: d for d in assess["decisions"]}
    assert by_item["bank_statements"]["error"] == "invalid_decision_output"
    assert not by_item["bank_statements"]["accepted"]
    assert by_item["photo_id"]["accepted"] and by_item["photo_id"]["choice"] == "missing"
    assert by_item["photo_id"]["confidence"] == 0.92
    assert by_item["photo_id"]["normalized"] == ["choice 'Missing' read as 'missing'",
                                                 "confidence '92%' read as 0.92"]
    assert sum(1 for d in assess["decisions"] if d["accepted"]) == 5
    run = ex.decide("o", review["approval_id"], actor="o", decision="choose",
                    content_digest=review["content_digest"], choices={"bank_statements": "present"})
    statuses = {i["id"]: i["status"] for i in _rec(run, "assess")["output"]["items"]}
    assert statuses["bank_statements"] == "present" and statuses["photo_id"] == "missing"
