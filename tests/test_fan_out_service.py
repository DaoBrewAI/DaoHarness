"""Service: the run view, the batch approval, item retries and the deliverable for a per-item run."""
import copy

import pytest

from daoharness.service import WorkflowServiceError
from industry_fixtures import ACCOUNTING, fan_out_fake_outputs, recruiting_fan_out_sample, recruiting_fan_out_spec
from workflow_fakes import FakeModel, as_model_payload, service_for

ORG = "o"
NAMES = ["Priya Shah", "Marco Silva", "Li Wei"]


def _expect(status, code, fn):
    with pytest.raises(WorkflowServiceError) as info:
        fn()
    assert (info.value.status, info.value.code) == (status, code), info.value.message
    return info.value


def _confirmed(tmp_path, spec=None, model=None, **kw):
    model = model or FakeModel(step_overrides=fan_out_fake_outputs())
    model.compile_outputs.append(as_model_payload(copy.deepcopy(spec or recruiting_fan_out_spec())))
    svc = service_for(tmp_path, model, **kw)
    created = svc.create(ORG, "c1", "For each applicant in the packet, screen them and draft a reply.")
    iid = created["instance"]["instance_id"]
    svc.confirm(ORG, iid, "k1", created["draft_id"])
    return svc, model, iid


def _waiting(tmp_path, **kw):
    svc, model, iid = _confirmed(tmp_path, **kw)
    view = svc.run(ORG, iid, "r1", {"version": 1, "content": recruiting_fan_out_sample()})
    return svc, model, iid, view


def _step(view, step_id):
    return next(s for s in view["workflow_run"]["steps"] if s["step_id"] == step_id)


def _approval(svc):
    [approval] = svc.approvals(ORG, ORG)["approvals"]
    return approval


def _keys(view):
    return {i["label"]: i["key"] for i in view["workflow_run"]["items"]}


def test_a_per_item_run_lists_its_items_and_asks_for_one_batch_approval(tmp_path):
    svc, model, iid, view = _waiting(tmp_path)
    run = view["workflow_run"]
    assert view["status"] == "waiting_approval"
    assert [(i["label"], i["status"], i["reason"]) for i in run["items"]] == [(n, "active", None) for n in NAMES]
    assert all(set(i) == {"key", "label", "status", "reason"} for i in run["items"])
    extract = _step(view, "extract")
    assert [i["label"] for i in extract["items"]] == NAMES
    assert all("pending_outcomes" not in i and "review" not in i for i in extract["items"])
    assert "items" not in _step(view, "split") and "items" not in _step(view, "approval")
    approval = _approval(svc)
    assert approval["itemized"] is True and approval["item_keys"] == list(_keys(view).values())
    assert approval["item_refs"] == ["step:outreach"]
    assert svc.detail(ORG, iid)["runs"][0]["items"] == 3


def test_a_single_case_run_has_no_items(tmp_path):
    svc, model, iid = _confirmed(tmp_path, spec=ACCOUNTING["spec"], model=FakeModel())
    view = svc.run(ORG, iid, "r1", {"version": 1, "content": ACCOUNTING["sample"]})
    run = view["workflow_run"]
    assert "items" not in run and all("items" not in s for s in run["steps"])
    assert not {"itemized", "item_keys", "item_refs"} & set(_approval(svc))
    assert svc.detail(ORG, iid)["runs"][0]["items"] is None


def test_each_item_is_approved_edited_or_rejected_and_the_package_says_so(tmp_path):
    svc, model, iid, view = _waiting(tmp_path)
    keys = _keys(view)
    approval = _approval(svc)
    edited = {"step:outreach": {"subject": "Hi", "body": "Hello Marco, thanks for applying.", "notes": ""}}
    done = svc.decide(ORG, ORG, approval["approval_id"], "d1", "review_items", approval["content_digest"],
                      item_decisions={keys["Priya Shah"]: {"decision": "approve"},
                                      keys["Marco Silva"]: {"decision": "edit", "edited_content": edited},
                                      keys["Li Wei"]: {"decision": "reject"}})
    assert done["status"] == "complete"
    assert done["workflow_run"]["metrics"]["human_interventions"] == 2      # the batch and one edited item
    verdicts = next(i for i in done["workflow_run"]["deliverable"]["items"] if i["step_id"] == "approval")["content"]
    by_label = {i["label"]: i for i in verdicts["items"]}
    assert by_label["Priya Shah"]["decision"] == "approved" and not by_label["Priya Shah"]["edited"]
    assert by_label["Marco Silva"]["edited"] and by_label["Marco Silva"]["content"] == edited
    assert by_label["Li Wei"]["decision"] == "rejected"


def test_rejecting_every_item_rejects_the_run(tmp_path):
    svc, model, iid, view = _waiting(tmp_path)
    approval = _approval(svc)
    done = svc.decide(ORG, ORG, approval["approval_id"], "d1", "review_items", approval["content_digest"],
                      item_decisions={k: {"decision": "reject"} for k in approval["item_keys"]})
    assert done["status"] == "rejected"


def test_one_item_can_be_run_again(tmp_path):
    svc, model, iid, view = _waiting(tmp_path)
    keys = _keys(view)
    before = model.count("step", "extract")
    view = svc.retry(ORG, iid, "r1", step_id="extract", item_key=keys["Marco Silva"])
    assert view["status"] == "waiting_approval" and model.count("step", "extract") == before + 1
    assert view["workflow_run"]["metrics"]["human_interventions"] == 1
    _expect(404, "item_not_found", lambda: svc.retry(ORG, iid, "r1", step_id="extract", item_key="it_0000000000"))
    _expect(404, "item_not_found", lambda: svc.retry(ORG, iid, "r1", step_id="extract", item_key=3))
