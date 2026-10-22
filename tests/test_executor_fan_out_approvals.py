"""Executor: one batched approval with a verdict per item; retry or exclude a single item."""
import copy

import pytest

from daoharness.decision import DecisionAnswer
from daoharness.executor import ExecutorError
from daoharness.spec import WorkflowSpec, digest
from industry_fixtures import ACCOUNTING, fan_out_fake_outputs, recruiting_fan_out_sample, recruiting_fan_out_spec
from workflow_fakes import FakeModel, executor_for

RUN = "i.r1"


def _model(**overrides):
    return FakeModel(step_overrides={**fan_out_fake_outputs(), **overrides})


def _start(ex, spec=None):
    return ex.start(org_id="o", run_id=RUN, instance_id="i", version=1,
                    spec=WorkflowSpec.from_dict(spec or recruiting_fan_out_spec()),
                    input_payload={"source_kind": "supplied_input", "content": recruiting_fan_out_sample()})


def _rec(run, step_id):
    return next(s for s in run["steps"] if s["step_id"] == step_id)


def _keys(run):
    return {i["label"]: i["key"] for i in run["items"]}


def _waiting(tmp_path, model=None, **kw):
    ex, store, notifier = executor_for(tmp_path, model or _model(), **kw)
    _start(ex)
    run = ex.advance("o", RUN)
    approval = store.read("o", "approvals", _rec(run, "approval")["approval_id"])
    return ex, store, notifier, run, approval


def _review(ex, approval, item_decisions, **kw):
    return ex.decide("o", approval["approval_id"], actor="o", decision="review_items",
                     content_digest=approval["content_digest"], item_decisions=item_decisions, **kw)


EDITED = {"subject": "Next steps", "body": "Hello Marco, ref 123-45-6789, see you soon.", "notes": ""}


# --- itemized approval ---------------------------------------------------------------------------------------------

def test_post_segment_approval_is_itemized_single_inbox_item(tmp_path):
    ex, store, notifier, run, approval = _waiting(tmp_path)
    assert approval["kind"] == "approval" and approval["itemized"] is True
    assert approval["item_keys"] == [i["key"] for i in run["items"]]
    assert approval["item_refs"] == ["step:outreach"]
    assert approval["content_digest"] == digest(approval["content"])
    assert len(notifier.events) == 1 and len(store.list("o", "approvals")) == 1


def test_single_case_approval_is_not_itemized(tmp_path):
    ex, store, _ = executor_for(tmp_path, FakeModel())
    ex.start(org_id="o", run_id=RUN, instance_id="i", version=1, spec=WorkflowSpec.from_dict(ACCOUNTING["spec"]),
             input_payload={"source_kind": "supplied_input", "content": ACCOUNTING["sample"]})
    run = ex.advance("o", RUN)
    approval = store.read("o", "approvals", _rec(run, "approval")["approval_id"])
    assert not {"itemized", "item_keys", "item_refs"} & set(approval)


def test_approve_all_gives_itemized_output(tmp_path):
    ex, store, _, run, approval = _waiting(tmp_path)
    run = ex.decide("o", approval["approval_id"], actor="o", decision="approve",
                    content_digest=approval["content_digest"])
    output = _rec(run, "approval")["output"]
    assert output["decision"] == "approved" and output["approver"] == "o"
    assert [(i["label"], i["decision"], i["edited"]) for i in output["items"]] == [
        ("Priya Shah", "approved", False), ("Marco Silva", "approved", False), ("Li Wei", "approved", False)]
    assert output["items"][0]["content"]["step:outreach"]["subject"] == "Next steps for Priya Shah"
    assert output["approved_content_digest"] == digest(output["items"])
    assert run["human_interventions"] == 1 and run["state"] == "running"
    assert ex.advance("o", RUN)["state"] == "completed"


def test_review_items_mixed_approve_edit_reject(tmp_path):
    ex, store, _, run, approval = _waiting(tmp_path)
    keys = _keys(run)
    run = _review(ex, approval, {
        keys["Priya Shah"]: {"decision": "approve"},
        keys["Marco Silva"]: {"decision": "edit", "edited_content": {"step:outreach": EDITED}},
        keys["Li Wei"]: {"decision": "reject"}}, note="two of three")
    output = _rec(run, "approval")["output"]
    by_label = {i["label"]: i for i in output["items"]}
    assert (by_label["Priya Shah"]["decision"], by_label["Priya Shah"]["edited"]) == ("approved", False)
    assert (by_label["Marco Silva"]["decision"], by_label["Marco Silva"]["edited"]) == ("approved", True)
    assert by_label["Marco Silva"]["content"]["step:outreach"]["body"] == "Hello Marco, ref •••• 6789, see you soon."
    assert by_label["Li Wei"]["decision"] == "rejected"
    assert by_label["Li Wei"]["content"]["step:outreach"]["subject"] == "Next steps for Li Wei"  # kept, marked
    assert output["note"] == "two of three" and output["approved_content_digest"] == digest(output["items"])
    assert run["human_interventions"] == 2 and run["state"] == "running"
    stored = store.read("o", "approvals", approval["approval_id"])
    assert stored["status"] == "approved" and stored["decision"] == "review_items"
    assert stored["item_decisions"][keys["Marco Silva"]]["edited_content"]["step:outreach"]["body"].endswith(
        "•••• 6789, see you soon.")
    run = ex.advance("o", RUN)
    assert run["state"] == "completed"
    package = run["deliverable"]["items"]
    approved = next(i for i in package if i["step_id"] == "approval")["content"]
    assert [i["decision"] for i in approved["items"]] == ["approved", "approved", "rejected"]


def test_review_items_all_rejected_rejects_run(tmp_path):
    ex, store, _, run, approval = _waiting(tmp_path)
    run = _review(ex, approval, {k: {"decision": "reject"} for k in approval["item_keys"]})
    assert run["state"] == "rejected" and run["code"] == "approval_rejected"
    assert _rec(run, "approval")["state"] == "rejected"
    assert store.read("o", "approvals", approval["approval_id"])["status"] == "rejected"


@pytest.mark.parametrize("bad", [
    lambda keys: {keys[0]: {"decision": "approve"}, keys[1]: {"decision": "approve"}},           # a key missing
    lambda keys: {**{k: {"decision": "approve"} for k in keys}, "it_0000000000": {"decision": "approve"}},
    lambda keys: {k: {"decision": "maybe"} for k in keys},
    lambda keys: {k: "approve" for k in keys},
    lambda keys: {**{k: {"decision": "approve"} for k in keys}, keys[0]: {"decision": "edit"}},
    lambda keys: {**{k: {"decision": "approve"} for k in keys},
                  keys[0]: {"decision": "edit", "edited_content": {"step:other": EDITED}}},
    lambda keys: None,
])
def test_review_items_must_cover_every_key(tmp_path, bad):
    ex, store, _, run, approval = _waiting(tmp_path)
    with pytest.raises(ExecutorError) as info:
        _review(ex, approval, bad(approval["item_keys"]))
    assert (info.value.code, info.value.status) == ("invalid_item_decisions", 400)
    assert store.read("o", "approvals", approval["approval_id"])["status"] == "pending"


def test_review_items_edit_is_masked(tmp_path):
    ex, store, _, run, approval = _waiting(tmp_path)
    keys = approval["item_keys"]
    run = _review(ex, approval, {**{k: {"decision": "approve"} for k in keys},
                                 keys[0]: {"decision": "edit", "edited_content": {"step:outreach": EDITED}}})
    content = _rec(run, "approval")["output"]["items"][0]["content"]
    assert "123-45-6789" not in str(content) and "•••• 6789" in content["step:outreach"]["body"]


def test_review_items_replay_after_crash(tmp_path):
    ex, store, _, run, approval = _waiting(tmp_path)
    keys = approval["item_keys"]
    decisions = {**{k: {"decision": "approve"} for k in keys}, keys[1]: {"decision": "reject"}}
    real = store.write

    def crash(org, coll, key, value):
        if coll == "runs":
            raise KeyboardInterrupt("crash")
        return real(org, coll, key, value)
    store.write = crash
    with pytest.raises(KeyboardInterrupt):
        _review(ex, approval, decisions)
    store.write = real
    assert store.read("o", "runs", RUN)["state"] == "waiting_approval"
    # the replay applies the stored verdicts, whatever the second call carries
    run = _review(ex, approval, {k: {"decision": "approve"} for k in keys})
    assert [i["decision"] for i in _rec(run, "approval")["output"]["items"]] == ["approved", "rejected", "approved"]
    assert run["human_interventions"] == 1


def test_review_items_not_allowed_on_single_case_approval(tmp_path):
    ex, store, _ = executor_for(tmp_path, FakeModel())
    ex.start(org_id="o", run_id=RUN, instance_id="i", version=1, spec=WorkflowSpec.from_dict(ACCOUNTING["spec"]),
             input_payload={"source_kind": "supplied_input", "content": ACCOUNTING["sample"]})
    run = ex.advance("o", RUN)
    approval = store.read("o", "approvals", _rec(run, "approval")["approval_id"])
    with pytest.raises(ExecutorError) as info:
        _review(ex, approval, {})
    assert info.value.code == "invalid_decision"


def test_whole_batch_edit_not_allowed_on_itemized_approval(tmp_path):
    ex, store, _, run, approval = _waiting(tmp_path)
    with pytest.raises(ExecutorError) as info:
        ex.decide("o", approval["approval_id"], actor="o", decision="edit",
                  content_digest=approval["content_digest"], edited_content=copy.deepcopy(approval["content"]))
    assert info.value.code == "invalid_decision"


def test_stale_digest_rejected(tmp_path):
    ex, store, _, run, approval = _waiting(tmp_path)
    with pytest.raises(ExecutorError) as info:
        ex.decide("o", approval["approval_id"], actor="o", decision="review_items", content_digest="sha256:" + "0" * 64,
                  item_decisions={k: {"decision": "approve"} for k in approval["item_keys"]})
    assert info.value.code == "approval_stale"


class RulePort:
    """Answers "no" to every draft rule whose draft names Marco Silva; everything else as the default fake."""

    def decide(self, items, *, rubric, org_id):
        return [DecisionAnswer(id=i.id, choice="no" if i.kind == "yes_no" and "Marco Silva" in i.context
                               else i.options[0], confidence=0.95, provider="rule-decider", model="rules-1")
                for i in items]


def test_rule_flags_are_collected_per_item(tmp_path):
    spec = recruiting_fan_out_spec()
    spec["steps"][5]["config"]["rules"] = ["Thank the applicant."]
    ex, store, _ = executor_for(tmp_path, _model(), decisions=RulePort())
    _start(ex, spec)
    run = ex.advance("o", RUN)
    keys = _keys(run)
    assert run["state"] == "waiting_approval"
    marco = next(i for i in _rec(run, "outreach")["items"] if i["label"] == "Marco Silva")
    assert [f["id"] for f in marco["flags"]] == ["rule_compliance"]
    approval = store.read("o", "approvals", _rec(run, "approval")["approval_id"])
    assert [(f["step_id"], f["check"], f["item_key"], f["item_label"]) for f in approval["flags"]] == [
        ("outreach", "rule_compliance", keys["Marco Silva"], "Marco Silva")]


def reply_spec():
    """A shape a compiler model produces: the draft reads extract and screen, the approval reviews screen and draft,
    the summary reads the approval, and the package reads the draft directly next to the approval."""
    spec = recruiting_fan_out_spec()
    steps = {step["id"]: step for step in spec["steps"]}
    steps["outreach"]["inputs"] = ["step:extract", "step:assess"]
    steps["approval"]["inputs"] = ["step:assess", "step:outreach"]
    steps["summary"]["inputs"] = ["step:extract", "step:assess", "step:approval"]
    steps["package"]["inputs"] = ["step:assess", "step:outreach", "step:approval", "step:summary"]
    return spec


def test_a_later_step_reading_per_item_drafts_gets_each_approved_verdict(tmp_path):
    ex, store, _ = executor_for(tmp_path, _model())
    _start(ex, reply_spec())
    run = ex.advance("o", RUN)
    approval = store.read("o", "approvals", _rec(run, "approval")["approval_id"])
    keys = _keys(run)
    marco = copy.deepcopy(_item(approval, keys["Marco Silva"]))
    marco["step:outreach"] = EDITED
    run = _review(ex, approval, {keys["Priya Shah"]: {"decision": "approve"},
                                 keys["Marco Silva"]: {"decision": "edit", "edited_content": marco},
                                 keys["Li Wei"]: {"decision": "reject"}})
    run = ex.advance("o", RUN)
    assert run["state"] == "completed"
    drafts = next(i for i in run["deliverable"]["items"] if i["step_id"] == "outreach")["content"]
    by_label = {i["label"]: i for i in drafts["items"]}
    assert by_label["Marco Silva"]["output"]["body"] == "Hello Marco, ref •••• 6789, see you soon."
    assert by_label["Priya Shah"]["output"]["subject"] == "Next steps for Priya Shah"
    assert "rejected" not in by_label["Priya Shah"] and "rejected" not in by_label["Marco Silva"]
    assert by_label["Li Wei"]["rejected"] is True
    assert by_label["Li Wei"]["output"]["subject"] == "Next steps for Li Wei"      # kept, marked
    original = next(i for i in _rec(run, "outreach")["items"] if i["label"] == "Marco Silva")
    assert original["output"]["body"] != EDITED["body"]                            # the draft record is untouched


class UnsettledRulePort:
    """Not sure enough of the draft rules for Marco Silva (below the 0.75 threshold)."""

    def decide(self, items, *, rubric, org_id):
        marco = [i.kind == "yes_no" and "Marco Silva" in i.context for i in items]
        return [DecisionAnswer(id=i.id, choice="no" if unsure else i.options[0], confidence=0.6 if unsure else 0.95,
                               provider="rule-decider", model="rules-1")
                for i, unsure in zip(items, marco)]


@pytest.mark.parametrize("port, detail", [(RulePort(), "violated"), (UnsettledRulePort(), "unsettled")])
def test_reply_graph_flags_a_draft_rule_to_the_approver_and_does_not_stop(tmp_path, port, detail):
    spec = reply_spec()
    spec["steps"][5]["config"]["rules"] = ["Thank the applicant.", "Clearly communicates status or next steps."]
    ex, store, _ = executor_for(tmp_path, _model(), decisions=port)
    _start(ex, spec)
    run = ex.advance("o", RUN)
    keys = _keys(run)
    assert run["state"] == "waiting_approval" and run["code"] is None
    outreach = _rec(run, "outreach")
    assert outreach["state"] == "completed" and all(i["state"] == "completed" for i in outreach["items"])
    marco = next(i for i in outreach["items"] if i["label"] == "Marco Silva")
    assert [f["id"] for f in marco["flags"]] == ["rule_compliance"] and marco["flags"][0]["detail"].startswith(detail)
    approval = store.read("o", "approvals", _rec(run, "approval")["approval_id"])
    assert [(f["check"], f["item_label"]) for f in approval["flags"]] == [("rule_compliance", "Marco Silva")]
    assert approval["item_keys"] == [keys["Priya Shah"], keys["Marco Silva"], keys["Li Wei"]]


def test_a_per_item_draft_read_outside_its_approval_still_stops_on_a_broken_rule(tmp_path):
    spec = reply_spec()
    steps = {step["id"]: step for step in spec["steps"]}
    steps["outreach"]["config"]["rules"] = ["Thank the applicant."]
    steps["summary"]["inputs"] = ["step:extract", "step:assess", "step:outreach"]   # bypasses the approval
    ex, store, _ = executor_for(tmp_path, _model(), decisions=RulePort())
    _start(ex, spec)
    run = ex.advance("o", RUN)
    assert run["state"] == "needs_attention"
    marco = next(i for i in _rec(run, "outreach")["items"] if i["label"] == "Marco Silva")
    assert marco["state"] == "needs_attention" and marco["error"]["checks"] == ["rule_compliance"]
    assert store.list("o", "approvals") == []


def _item(approval, key):
    return {ref: copy.deepcopy(next(i for i in approval["content"][ref]["items"] if i["key"] == key)["output"])
            for ref in approval["item_refs"]}


# --- retry one item, exclude one item ------------------------------------------------------------------------------

class Hollow:
    """An extract override that claims a field with no value and no quote for the named people, until fixed."""

    def __init__(self, *names):
        self.names, self.base = set(names), fan_out_fake_outputs()["extract"]

    def __call__(self, input_data):
        out = self.base(input_data)
        if input_data["inputs"]["step:split"]["label"] in self.names:
            out["fields"][0].update(value="", source_quotes=[])
        return out


def _extract_calls(model, name=None):
    return [c for c in model.calls if c["operation"] == "step" and c["input"]["step"]["id"] == "extract"
            and (name is None or c["input"]["inputs"]["step:split"]["label"] == name)]


def _paused_on_extract(tmp_path, *names):
    leak = Hollow(*names)
    model = _model(extract=leak)
    ex, store, _ = executor_for(tmp_path, model)
    _start(ex)
    run = ex.advance("o", RUN)
    assert run["code"] == "items_need_attention"
    return ex, store, model, leak, run


def test_retry_one_item_keeps_other_items_cache(tmp_path):
    ex, store, model, leak, run = _paused_on_extract(tmp_path, "Priya Shah")
    keys = _keys(run)
    leak.names.clear()
    run = ex.retry("o", RUN, "extract", item_key=keys["Priya Shah"])
    assert run["state"] == "running" and run["human_interventions"] == 1
    record = _rec(run, "extract")
    assert record["state"] == "pending" and record["attempt"] == 0
    priya = next(i for i in record["items"] if i["key"] == keys["Priya Shah"])
    assert priya["state"] == "pending" and priya["attempt"] == 1
    assert [i["state"] for i in record["items"] if i["key"] != keys["Priya Shah"]] == ["completed", "completed"]
    assert next(i for i in run["items"] if i["key"] == keys["Priya Shah"])["retried"] == 1
    run = ex.advance("o", RUN)
    assert run["state"] == "waiting_approval"
    assert len(_extract_calls(model)) == 4 and len(_extract_calls(model, "Priya Shah")) == 2


def test_every_item_retry_counts_as_an_intervention(tmp_path):
    ex, store, model, leak, run = _paused_on_extract(tmp_path, "Priya Shah")
    priya_key = _keys(run)["Priya Shah"]
    for expected in (1, 2, 3):
        ex.retry("o", RUN, "extract", item_key=priya_key)
        run = ex.advance("o", RUN)
        assert run["human_interventions"] == expected
    leak.names.clear()
    ex.retry("o", RUN, "extract", item_key=priya_key)
    run = ex.advance("o", RUN)
    assert run["state"] == "waiting_approval" and run["human_interventions"] == 4
    assert next(i for i in run["items"] if i["key"] == priya_key)["retried"] == 4


def test_retry_item_resets_post_segment_steps(tmp_path):
    ex, store, _, run, approval = _waiting(tmp_path)
    decisions_port = ex.decisions.provider
    keys = _keys(run)
    before = len(decisions_port.calls)
    run = ex.retry("o", RUN, "assess", item_key=keys["Marco Silva"])
    assert store.read("o", "approvals", approval["approval_id"])["status"] == "superseded"
    for step_id in ("route", "outreach"):
        states = {i["label"]: i["state"] for i in _rec(run, step_id)["items"]}
        assert states == {"Priya Shah": "completed", "Marco Silva": "pending", "Li Wei": "completed"}
        assert _rec(run, step_id)["state"] == "pending"
    assert _rec(run, "approval")["state"] == "pending" and _rec(run, "approval")["approval_id"] is None
    assert _rec(run, "extract")["state"] == "completed"
    run = ex.advance("o", RUN)
    assert run["state"] == "waiting_approval"
    assert len(decisions_port.calls) == before + 1  # only Marco's assess ran again (route came from the cache)
    pending = [a for a in store.list("o", "approvals") if a["status"] == "pending"]
    assert len(pending) == 1 and pending[0]["approval_id"] != approval["approval_id"]


def test_retry_item_supersedes_pending_review(tmp_path):
    from test_executor_fan_out import ContextPort, _review_model

    ex, store, _ = executor_for(tmp_path, _review_model(), decisions=ContextPort(low=("Marco Silva",)))
    _start(ex)
    run = ex.advance("o", RUN)
    keys = _keys(run)
    old = store.read("o", "approvals", _rec(run, "assess")["approval_id"])
    run = ex.retry("o", RUN, "assess", item_key=keys["Priya Shah"])
    assert store.read("o", "approvals", old["approval_id"])["status"] == "superseded"
    marco = next(i for i in _rec(run, "assess")["items"] if i["key"] == keys["Marco Silva"])
    assert marco["state"] == "waiting_review"
    run = ex.advance("o", RUN)
    assert run["state"] == "waiting_approval"
    new = store.read("o", "approvals", _rec(run, "assess")["approval_id"])
    assert new["status"] == "pending" and new["approval_id"] != old["approval_id"]
    assert {i["item_key"] for i in new["items"]} == {keys["Marco Silva"]}


def test_retry_item_errors(tmp_path):
    ex, store, model, leak, run = _paused_on_extract(tmp_path, "Priya Shah")
    keys = _keys(run)
    for step_id, key in (("extract", "it_0000000000"), ("split", keys["Priya Shah"]),
                         ("approval", "it_0000000000")):
        with pytest.raises(ExecutorError) as info:
            ex.retry("o", RUN, step_id, item_key=key)
        assert (info.value.code, info.value.status) == ("item_not_found", 404)


def test_exclude_item_continues_run_and_lists_excluded(tmp_path):
    ex, store, model, leak, run = _paused_on_extract(tmp_path, "Priya Shah")
    keys = _keys(run)
    run = ex.exclude_item("o", RUN, keys["Priya Shah"], actor="o", note="wrong role")
    assert run["state"] == "running" and run["human_interventions"] == 1
    priya = next(i for i in run["items"] if i["key"] == keys["Priya Shah"])
    assert (priya["status"], priya["reason"], priya["excluded_by"]) == ("excluded", "wrong role", "o")
    run = ex.advance("o", RUN)
    assert run["state"] == "waiting_approval"
    extract = _rec(run, "extract")
    assert extract["state"] == "completed"
    assert [i["label"] for i in extract["output"]["items"]] == ["Marco Silva", "Li Wei"]
    assert extract["output"]["excluded"] == [{"key": keys["Priya Shah"], "label": "Priya Shah",
                                              "reason": "wrong role"}]
    assert [i["label"] for i in _rec(run, "outreach")["items"]] == ["Marco Silva", "Li Wei"]
    approval = store.read("o", "approvals", _rec(run, "approval")["approval_id"])
    assert approval["item_keys"] == [keys["Marco Silva"], keys["Li Wei"]]
    assert len(_extract_calls(model)) == 3


def test_exclude_reaggregates_earlier_per_item_steps(tmp_path):
    def leaky_outreach(input_data):
        out = fan_out_fake_outputs()["outreach"](input_data)
        fields = {f["name"]: f["value"] for f in input_data["inputs"]["step:extract"]["fields"]}
        if fields["full_name"] == "Li Wei":
            out["body"] = ""  # a required field left empty
        return out
    ex, store, _ = executor_for(tmp_path, _model(outreach=leaky_outreach))
    _start(ex)
    run = ex.advance("o", RUN)
    keys = _keys(run)
    assert _rec(run, "outreach")["state"] == "needs_attention"
    run = ex.exclude_item("o", RUN, keys["Li Wei"], actor="o", note=None)
    for step_id in ("extract", "assess", "route"):
        output = _rec(run, step_id)["output"]
        assert [i["label"] for i in output["items"]] == ["Priya Shah", "Marco Silva"]
        assert output["excluded"] == [{"key": keys["Li Wei"], "label": "Li Wei", "reason": "excluded"}]
    assert ex.advance("o", RUN)["state"] == "waiting_approval"


def test_exclude_last_item_needs_attention(tmp_path):
    ex, store, model, leak, run = _paused_on_extract(tmp_path, "Priya Shah", "Marco Silva", "Li Wei")
    keys = _keys(run)
    for name in ("Priya Shah", "Marco Silva"):
        ex.exclude_item("o", RUN, keys[name], actor="o", note=None)
        run = ex.advance("o", RUN)
        assert run["code"] == "items_need_attention"
    ex.exclude_item("o", RUN, keys["Li Wei"], actor="o", note=None)
    run = ex.advance("o", RUN)
    assert run["state"] == "needs_attention" and run["code"] == "no_items_left"
    assert _rec(run, "extract")["error"] == {"code": "no_items_left"}
    assert len(_extract_calls(model)) == 3


def test_exclude_rejected_when_not_paused(tmp_path):
    ex, store, _, run, approval = _waiting(tmp_path)
    keys = _keys(run)
    with pytest.raises(ExecutorError) as info:
        ex.exclude_item("o", RUN, keys["Priya Shah"], actor="o", note=None)
    assert (info.value.code, info.value.status) == ("item_not_excludable", 409)

    ex, store, model, leak, run = _paused_on_extract(tmp_path / "b", "Priya Shah")
    keys = _keys(run)
    with pytest.raises(ExecutorError) as info:
        ex.exclude_item("o", RUN, keys["Marco Silva"], actor="o", note=None)  # completed there
    assert info.value.code == "item_not_excludable"
    with pytest.raises(ExecutorError) as info:
        ex.exclude_item("o", RUN, "it_0000000000", actor="o", note=None)
    assert (info.value.code, info.value.status) == ("item_not_found", 404)


def test_retry_without_item_key_unchanged(tmp_path):
    ex, store, model, leak, run = _paused_on_extract(tmp_path, "Priya Shah")
    leak.names.clear()
    run = ex.retry("o", RUN, "extract")
    record = _rec(run, "extract")
    assert record["attempt"] == 1 and record["items"] == [] and record["state"] == "pending"
    run = ex.advance("o", RUN)
    assert run["state"] == "waiting_approval" and len(_extract_calls(model)) == 6


def test_whole_step_retry_keeps_item_attempts_of_later_steps(tmp_path):
    ex, store, model, leak, run = _paused_on_extract(tmp_path, "Priya Shah")
    keys = _keys(run)
    leak.names.clear()
    ex.retry("o", RUN, "extract", item_key=keys["Priya Shah"])
    ex.advance("o", RUN)
    run = ex.retry("o", RUN, "split")  # re-split: the same items, the retried item keeps its newer attempt
    priya = next(i for i in _rec(run, "extract")["items"] if i["key"] == keys["Priya Shah"])
    assert priya["attempt"] == 1 and priya["state"] == "pending"
    calls = len(_extract_calls(model))
    assert ex.advance("o", RUN)["state"] == "waiting_approval"
    assert len(_extract_calls(model)) == calls  # every item, Priya included, came from the cache
