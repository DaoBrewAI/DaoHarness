"""The same service code, two industries, end to end with fakes. Industry content lives only in the fixtures."""
import copy

import pytest

from daoharness.guardrails import flatten_text, mask_text
from daoharness.primitives import builtin_catalog
from industry_fixtures import ACCOUNTING, RECRUITING
from workflow_fakes import FakeModel, as_model_payload, service_for

ORG = "o"
INDUSTRIES = pytest.mark.parametrize("fixture", [ACCOUNTING, RECRUITING], ids=["accounting", "recruiting"])
ASSESSMENT = 2  # the only step whose primitive differs between the two industries
RAW_IDENTIFIER = "123-45-6789"  # the fake draft_message output carries it; the approver must only see it masked


def _decision_count(step):
    config = step["config"]
    if step["primitive"] == "classify_route":
        return 1
    return len(config.get("items") or config.get("criteria"))


@INDUSTRIES
def test_industry_runs_end_to_end_on_same_code(tmp_path, fixture):
    spec = copy.deepcopy(fixture["spec"])
    model = FakeModel(compile_outputs=[as_model_payload(spec)])
    svc = service_for(tmp_path, model)

    created = svc.create(ORG, "c1", fixture["description"])
    assert created["status"] == "draft" and created["workflow"]["gaps"] == []
    assert created["workflow"]["metrics"]["automation_ratio"] >= 0.7
    iid = created["instance"]["instance_id"]

    assert svc.confirm(ORG, iid, "k1", created["draft_id"])["status"] == "confirmed"
    run = svc.run(ORG, iid, "r1", {"version": 1, "content": fixture["sample"]})
    assert run["status"] == "waiting_approval"

    [approval] = svc.approvals(ORG, ORG)["approvals"]
    assert approval["kind"] == "approval" and approval["step_id"] == "approval"
    assert approval["workflow_domain"] == spec["domain"]
    draft = approval["content"]["step:outreach"]
    assert draft["subject"] and draft["body"]
    text = " ".join(flatten_text(approval["content"]))
    assert RAW_IDENTIFIER not in text and "•••• 6789" in text
    assert mask_text(approval["content"])[1] == 0  # nothing left that the masking patterns would catch

    done = svc.decide(ORG, ORG, approval["approval_id"], "d1", "approve", approval["content_digest"])
    assert done["status"] == "complete"
    wr = done["workflow_run"]
    assert wr["state"] == "completed" and wr["pending_approval_ids"] == []

    catalog = builtin_catalog()
    spec_steps = {s["id"]: s for s in spec["steps"]}
    records = {s["step_id"]: s for s in wr["steps"]}
    assert list(records) == [s["id"] for s in spec["steps"]]
    assert all(r["state"] == "completed" for r in records.values())
    for step_id, record in records.items():
        executor = catalog.get(record["primitive"]).executor
        if executor in ("model", "decision"):
            assert record["checks"], step_id
            assert all(c["passed"] for c in record["checks"]), (step_id, record["checks"])
        if executor == "decision":
            decisions = record["decisions"]
            assert len(decisions) == _decision_count(spec_steps[step_id]), step_id
            assert all(d["rung"] == "provider" and d["accepted"] is True and d["provider"] == "fake-decider"
                       and d["model"] and d["confidence"] is not None for d in decisions), (step_id, decisions)
        else:
            assert record["decisions"] == [], step_id
    decision_total = sum(_decision_count(spec_steps[i]) for i, r in records.items()
                         if catalog.get(r["primitive"]).executor == "decision")
    assert wr["metrics"]["decision_calls"] == decision_total

    approved = records["approval"]["output"]
    assert approved["decision"] == "approved" and approved["approver"] == ORG
    assert approved["approved_content_digest"] == approval["content_digest"]
    assert RAW_IDENTIFIER not in " ".join(flatten_text(approved["content"]))

    deliverable = wr["deliverable"]
    assert deliverable["title"] == spec_steps["package"]["config"]["title"] and deliverable["items"]
    result = done["result"]
    assert result["title"] and result["summary"]
    assert [c["criterion"] for c in result["criteria"]] == spec["completion_criteria"]
    assert "Drafts were not sent." in result["limitations"]


def test_both_industries_use_identical_primitive_set_except_assessment(tmp_path):
    compiled = []
    for name, fixture in (("a", ACCOUNTING), ("r", RECRUITING)):
        model = FakeModel(compile_outputs=[as_model_payload(copy.deepcopy(fixture["spec"]))])
        created = service_for(tmp_path / name, model).create(ORG, "c1", fixture["description"])
        compiled.append([s["primitive"] for s in created["workflow"]["spec"]["steps"]])
    accounting, recruiting = compiled
    assert len(accounting) == len(recruiting) == 9
    assert accounting[ASSESSMENT] == "checklist_match" and recruiting[ASSESSMENT] == "screen_score"
    assert ([p for i, p in enumerate(accounting) if i != ASSESSMENT]
            == [p for i, p in enumerate(recruiting) if i != ASSESSMENT])
    catalog = builtin_catalog()
    assert all(catalog.get(p) is not None for p in accounting + recruiting)


# --- per-item fan-out beside single-case ---------------------------------------------------------------------------

def test_fan_out_end_to_end(tmp_path):
    from industry_fixtures import fan_out_fake_outputs, recruiting_fan_out_sample, recruiting_fan_out_spec
    from test_executor_fan_out import CRITERIA, ContextPort

    # 1-2. build, save, run on a packet of three applicants
    spec = recruiting_fan_out_spec()
    model = FakeModel(compile_outputs=[as_model_payload(copy.deepcopy(spec))], step_overrides=fan_out_fake_outputs())
    svc = service_for(tmp_path / "fan", model, decisions=ContextPort(low=("Marco Silva",)))
    created = svc.create(ORG, "c1", "For each applicant in the packet, screen them and draft a reply.")
    assert created["workflow"]["metrics"]["fan_out"]["per_item_steps"] == ["extract", "assess", "route", "outreach"]
    iid = created["instance"]["instance_id"]
    svc.confirm(ORG, iid, "k1", created["draft_id"])
    view = svc.run(ORG, iid, "r1", {"version": 1, "content": recruiting_fan_out_sample()})
    keys = {i["label"]: i["key"] for i in view["workflow_run"]["items"]}
    assert list(keys) == ["Priya Shah", "Marco Silva", "Li Wei"]

    # 3. one low-confidence item: one review for the step
    [review] = svc.approvals(ORG, ORG)["approvals"]
    assert review["kind"] == "decision_review" and review["step_id"] == "assess"
    assert {i["item_key"] for i in review["items"]} == {keys["Marco Silva"]}
    view = svc.decide(ORG, ORG, review["approval_id"], "d1", "choose", review["content_digest"],
                      choices={i["id"]: "4" for i in review["items"]})

    # 4. one batched approval: approve two, edit one
    [approval] = svc.approvals(ORG, ORG)["approvals"]
    assert approval["kind"] == "approval" and approval["itemized"] is True
    assert approval["item_keys"] == list(keys.values())
    edited = {"step:outreach": {"subject": "Next steps", "body": "Hello Li Wei, let us talk next week.", "notes": ""}}
    done = svc.decide(ORG, ORG, approval["approval_id"], "d2", "review_items", approval["content_digest"],
                      item_decisions={keys["Priya Shah"]: {"decision": "approve"},
                                      keys["Marco Silva"]: {"decision": "approve"},
                                      keys["Li Wei"]: {"decision": "edit", "edited_content": edited}})
    assert done["status"] == "complete"

    # 5. the deliverable holds three items; each item's fields quote only its own text
    wr = done["workflow_run"]
    approved = next(i for i in wr["deliverable"]["items"] if i["step_id"] == "approval")["content"]
    assert [(i["label"], i["decision"], i["edited"]) for i in approved["items"]] == [
        ("Priya Shah", "approved", False), ("Marco Silva", "approved", False), ("Li Wei", "approved", True)]
    split = next(s for s in wr["steps"] if s["step_id"] == "split")["output"]["items"]
    texts = {i["key"]: i["text"] for i in split}
    extract = next(s for s in wr["steps"] if s["step_id"] == "extract")
    for item in extract["items"]:
        for field in item["output"]["fields"]:
            for quote in field["source_quotes"]:
                assert quote in texts[item["key"]]
                assert all(quote not in t for k, t in texts.items() if k != item["key"])
    # 6. Marco's four criteria were decided by a person, the rest by the decider
    assess = next(s for s in wr["steps"] if s["step_id"] == "assess")
    human = [d for i in assess["items"] for d in i["decisions"] if d["rung"] == "human"]
    assert len(human) == len(CRITERIA) and wr["metrics"]["human_interventions"] == 3

    # 7. the single-case workflow on the same code keeps its shape
    single = FakeModel(compile_outputs=[as_model_payload(copy.deepcopy(ACCOUNTING["spec"]))])
    svc2 = service_for(tmp_path / "single", single)
    created = svc2.create(ORG, "c1", ACCOUNTING["description"])
    assert "fan_out" not in created["workflow"]["metrics"]
    iid = created["instance"]["instance_id"]
    svc2.confirm(ORG, iid, "k1", created["draft_id"])
    svc2.run(ORG, iid, "r1", {"version": 1, "content": ACCOUNTING["sample"]})
    [approval] = svc2.approvals(ORG, ORG)["approvals"]
    assert not {"itemized", "item_keys", "item_refs"} & set(approval)
    done = svc2.decide(ORG, ORG, approval["approval_id"], "d1", "approve", approval["content_digest"])
    wr = done["workflow_run"]
    assert done["status"] == "complete" and "items" not in wr
    assert all("items" not in s for s in wr["steps"])
    assert [t["title"] for t in done["result"]["tables"]] == ["Step record"]
