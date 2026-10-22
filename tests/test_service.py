"""WorkflowService: build, save, run, decide; every action applied once; runs that outlive their process."""
import copy
import dataclasses
import json
import re

import pytest

from daoharness.ports import ModelError
from daoharness.service import DYNAMIC_SCHEMA, WorkflowServiceError
from industry_fixtures import ACCOUNTING
from workflow_fakes import FakeDecisionPort, FakeModel, as_model_payload, service_for

ORG = "o"
HEX32 = re.compile(r"[0-9a-f]{32}")
SAMPLE = ACCOUNTING["sample"]


def _payload(spec=None, **kw):
    return as_model_payload(copy.deepcopy(spec or ACCOUNTING["spec"]), **kw)


def _expect(status, code, fn):
    with pytest.raises(WorkflowServiceError) as info:
        fn()
    assert (info.value.status, info.value.code) == (status, code), info.value.message
    return info.value


def _created(tmp_path, model=None, spec=None, **kw):
    model = model or FakeModel()
    model.compile_outputs.append(_payload(spec))
    svc = service_for(tmp_path, model, **kw)
    return svc, model, svc.create(ORG, "c1", ACCOUNTING["description"])


def _confirmed(tmp_path, **kw):
    svc, model, created = _created(tmp_path, **kw)
    iid = created["instance"]["instance_id"]
    svc.confirm(ORG, iid, "k1", created["draft_id"])
    return svc, model, iid


def _waiting(tmp_path, **kw):
    svc, model, iid = _confirmed(tmp_path, **kw)
    return svc, model, iid, svc.run(ORG, iid, "r1", {"version": 1, "content": SAMPLE})


def _pending(svc):
    return svc.approvals(ORG, ORG)["approvals"]


def _approve_all(svc, iid, run):
    while run["status"] == "waiting_approval":
        [approval] = _pending(svc)
        if approval["kind"] == "decision_review":
            choices = {i["id"]: i["options"][0] for i in approval["items"]}
            run = svc.decide(ORG, ORG, approval["approval_id"], f"d-{approval['approval_id']}", "choose",
                             approval["content_digest"], choices=choices)
        else:
            run = svc.decide(ORG, ORG, approval["approval_id"], f"d-{approval['approval_id']}", "approve",
                             approval["content_digest"])
    return run


# --- build and save ----------------------------------------------------------------------------------------------

def test_create_builds_a_draft_from_the_description(tmp_path):
    svc, model, created = _created(tmp_path)
    assert created["schema"] == DYNAMIC_SCHEMA and created["status"] == "draft" and created["version"] == 1
    assert HEX32.fullmatch(created["instance"]["instance_id"]) and HEX32.fullmatch(created["draft_id"])
    assert created["need"] == ACCOUNTING["description"] and created["action_id"] == "c1"
    workflow = created["workflow"]
    assert set(workflow) == {"spec", "metrics", "gaps", "questions", "diff"}
    assert workflow["spec"]["domain"] == ACCOUNTING["spec"]["domain"] and workflow["gaps"] == []
    assert workflow["metrics"] == {"total_steps": 9, "automatic_steps": 8, "automation_ratio": 0.89}
    assert set(created["authoring_receipt"]) >= {"run_id", "provider", "model", "output_digest"}
    assert model.count("compile") == 1


def test_create_is_applied_once_and_an_action_id_cannot_be_reused_for_another_request(tmp_path):
    svc, model, created = _created(tmp_path)
    assert svc.create(ORG, "c1", ACCOUNTING["description"]) == created and model.count("compile") == 1
    _expect(409, "action_conflict", lambda: svc.create(ORG, "c1", "Another job entirely."))
    assert len(svc.catalog(ORG)["creations"]) == 1


@pytest.mark.parametrize("need, source", [("", None), ("   ", None), ("x" * 16001, None), (3, None),
                                          ("A job.", "email")])
def test_create_checks_its_input(tmp_path, need, source):
    svc = service_for(tmp_path, FakeModel())
    _expect(400, "invalid_dynamic_input", lambda: svc.create(ORG, "c1", need, source))


@pytest.mark.parametrize("action_id", ["", "Upper", "a b", "x" * 129, None])
def test_action_ids_are_checked(tmp_path, action_id):
    svc = service_for(tmp_path, FakeModel())
    _expect(400, "invalid_dynamic_input", lambda: svc.create(ORG, action_id, "A job."))


def test_a_build_that_stays_invalid_is_422_and_the_failure_is_kept_for_the_action(tmp_path):
    bad = _payload()
    bad["spec"]["steps"] = bad["spec"]["steps"][:-1]
    model = FakeModel(compile_outputs=[bad, copy.deepcopy(bad)])
    svc = service_for(tmp_path, model)
    err = _expect(422, "compile_invalid", lambda: svc.create(ORG, "c1", ACCOUNTING["description"]))
    assert "missing_deliverable" in err.message
    _expect(422, "compile_invalid", lambda: svc.create(ORG, "c1", ACCOUNTING["description"]))
    assert model.count("compile") == 2 and svc.catalog(ORG)["creations"] == []


def test_an_unavailable_model_is_503(tmp_path):
    svc = service_for(tmp_path, FakeModel(fail_operations={"compile"}))
    err = _expect(503, "model_unavailable", lambda: svc.create(ORG, "c1", ACCOUNTING["description"]))
    assert "provider_failed" in err.message


def test_confirm_saves_the_draft_once(tmp_path):
    svc, model, created = _created(tmp_path)
    iid = created["instance"]["instance_id"]
    _expect(404, "draft_not_found", lambda: svc.confirm(ORG, iid, "k0", "0" * 32))
    saved = svc.confirm(ORG, iid, "k1", created["draft_id"])
    assert saved["status"] == "confirmed" and saved["workflow"]["spec"] == created["workflow"]["spec"]
    assert svc.confirm(ORG, iid, "k1", created["draft_id"]) == saved
    _expect(409, "already_saved", lambda: svc.confirm(ORG, iid, "k2", created["draft_id"]))
    _expect(404, "workflow_not_found", lambda: svc.confirm(ORG, "f" * 32, "k3", created["draft_id"]))
    detail = svc.detail(ORG, iid)
    assert detail["readiness"] == {"status": "configured"} and detail["definition"]["action"]["enabled"]
    assert detail["family"] is None and detail["runs"] == []


def test_a_draft_cannot_run(tmp_path):
    svc, _, created = _created(tmp_path)
    iid = created["instance"]["instance_id"]
    assert svc.detail(ORG, iid)["readiness"] == {"status": "needs_input"}
    _expect(409, "workflow_not_ready", lambda: svc.run(ORG, iid, "r1", {"version": 1, "content": SAMPLE}))


# --- run ---------------------------------------------------------------------------------------------------------

def test_a_run_goes_through_its_approval_to_a_result(tmp_path):
    svc, model, iid, run = _waiting(tmp_path)
    assert run["status"] == "waiting_approval" and run["action_id"] == "r1" and run["version"] == 1
    assert run["input"] == {"source_kind": "supplied_input", "content": SAMPLE}
    [approval] = _pending(svc)
    assert approval["kind"] == "approval" and approval["approver_role"] == "Engagement partner"
    assert run["workflow_run"]["pending_approval_ids"] == [approval["approval_id"]]
    done = svc.decide(ORG, ORG, approval["approval_id"], "d1", "approve", approval["content_digest"])
    assert done["status"] == "complete" and done["workflow_run"]["state"] == "completed"
    result = done["result"]
    assert result["title"] and result["summary"] and result["tables"][0]["title"] == "Step record"
    assert "Drafts were not sent." in result["limitations"]
    assert done["workflow_run"]["metrics"]["human_interventions"] == 1
    assert done["usage"]["calls"] == done["workflow_run"]["metrics"]["model_calls"] > 0
    assert svc.detail(ORG, iid)["creation"]["run_actions"] == ["r1"]
    [summary] = svc.detail(ORG, iid)["runs"]
    assert summary["action_id"] == "r1" and summary["state"] == "completed"


def test_a_run_is_started_once(tmp_path):
    svc, model, iid, run = _waiting(tmp_path)
    calls = len(model.calls)
    assert svc.run(ORG, iid, "r1", {"version": 1, "content": SAMPLE}) == run and len(model.calls) == calls
    assert svc.run_view(ORG, iid, "r1") == run
    _expect(404, "run_not_found", lambda: svc.run_view(ORG, iid, "nope"))


@pytest.mark.parametrize("given", [
    {}, {"version": 1}, "text", {"version": 1, "content": {"documents": "D"}},
    {"version": 1, "content": {**SAMPLE, "extra": "x"}}, {"version": 1, "content": {**SAMPLE, "profile": 3}},
    {"version": 1, "content": {**SAMPLE, "profile": "x" * (64 * 1024 + 1)}},
    {"version": 1, "content": SAMPLE, "sample_id": "0" * 32}])
def test_run_input_is_checked(tmp_path, given):
    svc, model, iid = _confirmed(tmp_path)
    _expect(400, "invalid_dynamic_input", lambda: svc.run(ORG, iid, "r1", given))
    assert svc.detail(ORG, iid)["runs"] == []


def test_rejecting_ends_the_run(tmp_path):
    svc, _, iid, run = _waiting(tmp_path)
    [approval] = _pending(svc)
    done = svc.decide(ORG, ORG, approval["approval_id"], "d1", "reject", approval["content_digest"])
    assert done["status"] == "rejected" and "result" not in done and done["workflow_run"]["next_action"] is None
    assert _pending(svc) == []


def test_decide_checks_its_input(tmp_path):
    svc, _, iid, run = _waiting(tmp_path)
    [approval] = _pending(svc)
    aid, good = approval["approval_id"], approval["content_digest"]
    _expect(404, "approval_not_found", lambda: svc.decide(ORG, ORG, "0" * 32, "d1", "approve", good))
    _expect(409, "approval_stale", lambda: svc.decide(ORG, ORG, aid, "d2", "approve", "sha256:" + "0" * 64))
    _expect(400, "invalid_dynamic_input", lambda: svc.decide(ORG, ORG, aid, "d3", "approve", good, note="x" * 4001))
    _expect(403, "approval_forbidden", lambda: svc.decide(ORG, "someone else", aid, "d4", "approve", good))
    assert svc.approvals(ORG, "someone else") == {"approvals": []}
    done = svc.decide(ORG, ORG, aid, "d5", "approve", good)
    assert svc.decide(ORG, ORG, aid, "d5", "approve", good) == done      # the same action again
    _expect(409, "approval_closed", lambda: svc.decide(ORG, ORG, aid, "d6", "approve", good))


def test_an_edit_is_what_later_steps_read(tmp_path):
    svc, _, iid, run = _waiting(tmp_path)
    [approval] = _pending(svc)
    edited = copy.deepcopy(approval["content"])
    edited["step:outreach"]["body"] = "Please send the bank statements by Friday."
    done = svc.decide(ORG, ORG, approval["approval_id"], "d1", "edit", approval["content_digest"],
                      edited_content=edited)
    assert done["status"] == "complete" and done["workflow_run"]["metrics"]["human_interventions"] == 2
    package = done["workflow_run"]["deliverable"]
    approved = next(i for i in package["items"] if i["step_id"] == "approval")["content"]
    assert approved["content"]["step:outreach"]["body"] == "Please send the bank statements by Friday."


# --- decisions -----------------------------------------------------------------------------------------------------

def test_a_decision_below_the_threshold_goes_to_a_person_then_the_run_goes_on(tmp_path):
    port = FakeDecisionPort(answers={"photo_id": {"confidence": 0.3}})
    svc, model, iid, run = _waiting(tmp_path, decisions=port)
    [review] = _pending(svc)
    assert review["kind"] == "decision_review" and [i["id"] for i in review["items"]] == ["photo_id"]
    assert review["what_is_approved"] == "1 decision(s) the model was not sure of"
    assert model.count("decide") == 0
    run = svc.decide(ORG, ORG, review["approval_id"], "d1", "choose", review["content_digest"],
                     choices={"photo_id": "missing"})
    assert run["status"] == "waiting_approval" and _pending(svc)[0]["kind"] == "approval"
    assess = next(s for s in run["workflow_run"]["steps"] if s["step_id"] == "assess")
    assert {i["id"]: i["status"] for i in assess["output"]["items"]}["photo_id"] == "missing"


def test_without_a_decider_the_model_decides(tmp_path):
    svc, model, iid, run = _waiting(tmp_path, decisions=False)
    assert run["status"] == "waiting_approval" and model.count("decide") == 2     # the checklist and the route
    assess = next(s for s in run["workflow_run"]["steps"] if s["step_id"] == "assess")
    assert {d["rung"] for d in assess["decisions"]} == {"model"}


def test_the_threshold_can_be_set_for_the_service(tmp_path):
    port = FakeDecisionPort(answers={"photo_id": {"confidence": 0.8}})
    svc, _, iid, run = _waiting(tmp_path / "a", decisions=port, decision_threshold=0.85)
    assert _pending(svc)[0]["kind"] == "decision_review"
    svc, _, iid, run = _waiting(tmp_path / "b", decisions=FakeDecisionPort(answers={"photo_id": {"confidence": 0.8}}))
    assert _pending(svc)[0]["kind"] == "approval"


# --- usage ---------------------------------------------------------------------------------------------------------

class TokenModel(FakeModel):
    """Reports 100 input and 20 output tokens for every call."""

    def generate(self, **kw):
        content, receipt = super().generate(**kw)
        return content, dataclasses.replace(receipt, input_tokens=100, output_tokens=20, tokens_estimated=False)


def test_usage_adds_up_the_tokens_the_receipts_report(tmp_path):
    svc, model, iid, run = _waiting(tmp_path, model=TokenModel())
    done = _approve_all(svc, iid, run)
    usage = done["usage"]
    calls = done["workflow_run"]["metrics"]["model_calls"]
    assert usage == {"calls": calls, "input_tokens": 100 * calls, "output_tokens": 20 * calls, "estimated": False}


def test_usage_counts_the_decision_calls_too(tmp_path):
    svc, model, iid, run = _waiting(tmp_path, model=TokenModel(), decisions=False)
    done = _approve_all(svc, iid, run)
    calls = done["workflow_run"]["metrics"]["model_calls"]
    assert model.count("decide") == 2 and calls == model.count("step") + model.count("decide")
    assert done["usage"] == {"calls": calls, "input_tokens": 100 * calls, "output_tokens": 20 * calls,
                             "estimated": False}


def test_usage_says_estimated_when_a_provider_reports_no_tokens(tmp_path):
    svc, _, iid, run = _waiting(tmp_path)
    usage = _approve_all(svc, iid, run)["usage"]
    assert usage["estimated"] is True and usage["input_tokens"] == usage["output_tokens"] == 0


# --- interrupted runs, retry -----------------------------------------------------------------------------------------

def test_a_run_left_running_by_a_stopped_process_reads_interrupted_and_continues(tmp_path):
    dropped = []
    svc, model, iid = _confirmed(tmp_path, submit=dropped.append)
    run = svc.run(ORG, iid, "r1", {"version": 1, "content": SAMPLE})
    assert run["status"] == "running" and len(dropped) == 1      # handed to a job that never ran
    _expect(409, "run_not_retryable", lambda: svc.resume_run(ORG, iid, "r1"))
    restarted = service_for(tmp_path, model)
    view = restarted.run_view(ORG, iid, "r1")
    assert view["status"] == "interrupted" and view["workflow_run"]["next_action"] == "resume"
    assert "Continue it" in view["message"] and restarted.detail(ORG, iid)["runs"][0]["state"] == "interrupted"
    resumed = restarted.resume_run(ORG, iid, "r1")
    assert resumed["status"] == "waiting_approval"


def test_a_failed_step_can_be_retried(tmp_path):
    calls = []

    def flaky(data):
        calls.append(1)
        if len(calls) == 1:
            raise ModelError("provider_failed")
        return {"questions": [{"question": "Which bank statements are missing?", "reason": "checklist"}]}

    svc, model, iid, run = _waiting(tmp_path, model=FakeModel(step_overrides={"questions": flaky}))
    assert run["status"] == "failed" and run["code"] == "provider_failed"
    assert run["workflow_run"]["retryable"] and run["workflow_run"]["next_action"] == "retry"
    run = svc.retry(ORG, iid, "r1")
    assert run["status"] == "waiting_approval" and run["workflow_run"]["metrics"]["human_interventions"] == 1
    _expect(404, "step_not_found", lambda: svc.retry(ORG, iid, "r1", step_id=3))


def test_a_run_that_cannot_advance_reads_interrupted(tmp_path, monkeypatch):
    svc, model, iid = _confirmed(tmp_path, submit=lambda job: None)
    svc.run(ORG, iid, "r1", {"version": 1, "content": SAMPLE})

    def broken(*args, **kwargs):
        raise RuntimeError("a bug in a merge")

    monkeypatch.setattr(svc.executor, "_merge", broken)
    assert svc.drive(ORG, f"{iid}.r1") == "failed"
    assert svc.run_view(ORG, iid, "r1")["status"] == "interrupted"
    monkeypatch.undo()
    svc.submit = lambda job: job()
    assert svc.resume_run(ORG, iid, "r1")["status"] == "waiting_approval"


# --- what this engine cannot do ---------------------------------------------------------------------------------------

def test_a_source_the_engine_cannot_reach_is_a_gap_and_the_run_stops_there(tmp_path):
    spec = copy.deepcopy(ACCOUNTING["spec"])
    spec["steps"][4]["required_sources"] = ["web_search"]
    svc, model, created = _created(tmp_path, spec=spec)
    assert created["workflow"]["gaps"] == [{"step_id": "questions", "kind": "needs_capability",
                                            "detail": "web_search"}]
    iid = created["instance"]["instance_id"]
    svc.confirm(ORG, iid, "k1", created["draft_id"])
    readiness = svc.detail(ORG, iid)["readiness"]
    assert readiness["status"] == "needs_capability" and readiness["missing"] == ["web_search"]
    run = svc.run(ORG, iid, "r1", {"version": 1, "content": SAMPLE})
    assert run["status"] == "needs_attention" and run["code"] == "needs_capability"
    assert model.count("step", "questions") == 0


def test_the_catalog_lists_each_workflow_with_its_runs(tmp_path):
    svc, _, iid, run = _waiting(tmp_path)
    [entry] = svc.catalog(ORG)["creations"]
    assert entry["creation"]["instance"]["instance_id"] == iid and entry["creation"]["status"] == "confirmed"
    assert [r["state"] for r in entry["runs"]] == ["waiting_approval"]
    assert entry["runs"][0]["stopped_at"] == {"step_id": "approval", "role": "Engagement partner"}
    _expect(404, "workflow_not_found", lambda: svc.detail(ORG, "not an id"))


# --- the delivered folder ----------------------------------------------------------------------------------------

def test_a_completed_run_leaves_its_result_and_package_in_a_folder(tmp_path):
    svc, _, iid, run = _waiting(tmp_path, deliveries=tmp_path / "deliveries")
    assert "delivered_to" not in run
    done = _approve_all(svc, iid, run)
    [folder] = list((tmp_path / "deliveries" / "accounting-client-intake").iterdir())
    assert folder.name.startswith("20260101-0000-") and len(folder.name) == len("20260101-0000-") + 8
    assert done["delivered_to"] == str(folder)
    result = (folder / "result.md").read_text(encoding="utf-8")
    assert result.startswith(f"# {done['result']['title']}") and "## Limitations" in result
    assert "- Drafts were not sent." in result
    package = json.loads((folder / "package.json").read_text(encoding="utf-8"))
    assert package == done["workflow_run"]["deliverable"]
    assert sorted(p.name for p in folder.iterdir()) == ["package.json", "result.md"]


def test_a_rejected_run_leaves_no_folder_and_a_disk_error_does_not_fail_the_run(tmp_path):
    svc, _, iid, run = _waiting(tmp_path, deliveries=tmp_path / "deliveries")
    [approval] = _pending(svc)
    svc.decide(ORG, ORG, approval["approval_id"], "d1", "reject", approval["content_digest"])
    assert not (tmp_path / "deliveries").exists()
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the folder should be")
    svc2, _, iid2, run2 = _waiting(tmp_path / "two", deliveries=blocked)
    done = _approve_all(svc2, iid2, run2)
    assert done["status"] == "complete" and "delivered_to" not in done
