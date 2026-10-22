"""Executor: a split step writes the roster and per-item steps run once per item, breadth-first."""
import copy
import threading
import time

import pytest

import daoharness.executor as executor_module
from daoharness.decision import DecisionAnswer
from daoharness.executor import DEFAULT_PARALLEL_ITEMS, WorkflowExecutor
from daoharness.items import item_key
from daoharness.ports import ModelError
from daoharness.spec import WorkflowSpec
from industry_fixtures import (
    ACCOUNTING, FAN_OUT_APPLICANTS, fan_out_fake_outputs, recruiting_fan_out_sample, recruiting_fan_out_spec)
from workflow_fakes import FakeModel, executor_for

NAMES = [a[0] for a in FAN_OUT_APPLICANTS]


def _model(**overrides):
    return FakeModel(step_overrides={**fan_out_fake_outputs(), **overrides})


def _start(ex, spec=None, content=None, run_id="i.r1"):
    return ex.start(org_id="o", run_id=run_id, instance_id="i", version=1,
                    spec=WorkflowSpec.from_dict(spec or recruiting_fan_out_spec()),
                    input_payload={"source_kind": "supplied_input",
                                   "content": content or recruiting_fan_out_sample()})


def _rec(run, step_id):
    return next(s for s in run["steps"] if s["step_id"] == step_id)


def _label(call):
    inputs = call["input"]["inputs"]
    if "step:split" in inputs:
        return inputs["step:split"]["label"]
    fields = {f["name"]: f["value"] for f in inputs["step:extract"]["fields"]}
    return fields["full_name"]


def _calls(model, step_id, name=None):
    return [c for c in model.calls if c["operation"] == "step" and c["input"]["step"]["id"] == step_id
            and (name is None or _label(c) == name)]


def _keys(run):
    return {i["label"]: i["key"] for i in run["items"]}


def _flaky(times, exc=lambda: ModelError("provider_failed")):
    """An extract override that raises for Marco Silva the first `times` calls, then answers."""
    base = fan_out_fake_outputs()["extract"]
    seen = {"n": 0}

    def extract(input_data):
        if input_data["inputs"]["step:split"]["label"] == "Marco Silva" and seen["n"] < times:
            seen["n"] += 1
            raise exc()
        return base(input_data)
    return extract


def _hollow_extract(input_data):
    """For Priya Shah, claims a field it found nothing for: no value and no quote."""
    out = fan_out_fake_outputs()["extract"](input_data)
    if input_data["inputs"]["step:split"]["label"] == "Priya Shah":
        out["fields"][0] = {"name": "full_name", "found": True, "value": "", "source_quotes": []}
    return out


# --- roster and breadth-first segment ------------------------------------------------------------------------------

def test_split_writes_roster_and_segment_runs_once_per_item(tmp_path):
    model = _model()
    ex, store, _ = executor_for(tmp_path, model)
    _start(ex)
    run = ex.advance("o", "i.r1")
    assert run["state"] == "waiting_approval" and _rec(run, "approval")["state"] == "waiting_approval"
    assert [(i["label"], i["status"]) for i in run["items"]] == [(n, "active") for n in NAMES]
    assert all(set(i) == {"key", "label", "status", "reason", "excluded_by"} for i in run["items"])
    split = _rec(run, "split")["output"]["items"]
    assert [item_key(i["text"]) for i in split] == [i["key"] for i in run["items"]]
    extract_calls = _calls(model, "extract")
    assert [_label(c) for c in extract_calls] == NAMES
    for call, name in zip(extract_calls, NAMES):
        text = call["input"]["inputs"]["step:split"]["text"]
        assert name in text and all(other not in text for other in NAMES if other != name)
    for step_id in ("extract", "assess", "route", "outreach"):
        record = _rec(run, step_id)
        assert record["state"] == "completed"
        assert [i["state"] for i in record["items"]] == ["completed"] * 3
        assert [i["key"] for i in record["output"]["items"]] == [i["key"] for i in run["items"]]
        assert record["output"]["excluded"] == []
    extracted = [i["output"]["fields"][0]["value"] for i in _rec(run, "extract")["output"]["items"]]
    assert extracted == NAMES
    # breadth-first: every extract call happens before the first outreach call
    order = [c["input"]["step"]["id"] for c in model.calls if c["operation"] == "step"]
    assert max(i for i, s in enumerate(order) if s == "extract") < min(i for i, s in enumerate(order) if s == "outreach")


def test_post_segment_step_receives_aggregate(tmp_path):
    ex, store, notifier = executor_for(tmp_path, _model())
    _start(ex)
    run = ex.advance("o", "i.r1")
    approval = store.read("o", "approvals", _rec(run, "approval")["approval_id"])
    content = approval["content"]["step:outreach"]
    assert [i["label"] for i in content["items"]] == NAMES and content["excluded"] == []
    assert all(i["state"] == "completed" and i["output"]["subject"].endswith(i["label"]) for i in content["items"])


def test_separator_mode_splits_without_a_model_call(tmp_path):
    spec = recruiting_fan_out_spec()
    spec["steps"][1]["config"]["separator"] = "\n\n"
    model = _model()
    ex, _store, _ = executor_for(tmp_path, model)
    sample = recruiting_fan_out_sample()
    sample["packet"] = sample["packet"].split("\n\n", 1)[1]  # no cover note
    _start(ex, spec, sample)
    run = ex.advance("o", "i.r1")
    assert model.count("step", "split") == 0
    assert [i["label"] for i in run["items"]] == [f"Applicant: {n}" for n in NAMES]
    assert run["state"] == "waiting_approval"
    boundaries = next(c for c in _rec(run, "split")["checks"] if c["id"] == "item_boundaries")
    assert boundaries == {"id": "item_boundaries", "passed": True, "detail": ""}


def test_separator_missing_from_the_packet_falls_back_to_the_model_split(tmp_path):
    # A description of the items written as the separator never occurs in the packet, which would make the whole
    # packet one item. The split falls back to the item starts and says so.
    spec = recruiting_fan_out_spec()
    spec["steps"][1]["config"]["separator"] = "individual applicant resume or profile"
    model = _model()
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex, spec)
    run = ex.advance("o", "i.r1")
    assert model.count("step", "split") == 1
    assert [i["label"] for i in run["items"]] == NAMES
    assert run["state"] == "waiting_approval"
    record = _rec(run, "split")
    assert record["state"] == "completed"
    boundaries = next(c for c in record["checks"] if c["id"] == "item_boundaries")
    assert boundaries == {"id": "item_boundaries", "passed": True,
                          "detail": 'separator "individual applicant resume or profile" not found in the input; '
                                    "split at the item starts instead"}
    assert [i["label"] for i in record["output"]["items"]] == NAMES
    # the model is not handed the separator that failed; it finds the item starts on its own
    assert all("separator" not in c["input"]["step"]["config"] for c in _calls(model, "split"))


def test_separator_fallback_is_cached_like_a_model_split(tmp_path):
    spec = recruiting_fan_out_spec()
    spec["steps"][1]["config"]["separator"] = "=== next ==="
    model = _model()
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex, spec)
    ex.advance("o", "i.r1")
    _start(ex, spec, run_id="i.r2")
    run = ex.advance("o", "i.r2")
    assert model.count("step", "split") == 1
    assert [i["label"] for i in run["items"]] == NAMES
    boundaries = next(c for c in _rec(run, "split")["checks"] if c["id"] == "item_boundaries")
    assert boundaries["detail"] == 'separator "=== next ===" not found in the input; split at the item starts instead'


def test_separator_fallback_says_so_when_the_model_call_fails(tmp_path):
    def fail(_input):
        raise ModelError("provider_failed")
    spec = recruiting_fan_out_spec()
    spec["steps"][1]["config"]["separator"] = "=== next ==="
    ex, _store, _ = executor_for(tmp_path, _model(split=fail))
    _start(ex, spec)
    run = ex.advance("o", "i.r1")
    record = _rec(run, "split")
    assert record["error"] == {"code": "provider_failed",
                               "detail": 'separator "=== next ===" not found in the input; split at the item '
                                         "starts instead"}


def test_separator_with_one_genuine_item_uses_the_model_split(tmp_path):
    # One piece from a separator usually means the separator is wrong, so even a genuine single item goes to the
    # model split (one model call) and the record says why.
    spec = recruiting_fan_out_spec()
    spec["steps"][1]["config"]["separator"] = "---"
    model = _model()
    ex, _store, _ = executor_for(tmp_path, model)
    content = recruiting_fan_out_sample()
    content["packet"] = content["packet"].split("\n\n")[1] + "\n---\n"
    _start(ex, spec, content)
    run = ex.advance("o", "i.r1")
    assert model.count("step", "split") == 1
    assert [i["label"] for i in run["items"]] == [NAMES[0]]
    boundaries = next(c for c in _rec(run, "split")["checks"] if c["id"] == "item_boundaries")
    assert boundaries == {"id": "item_boundaries", "passed": True,
                          "detail": 'separator "---" cuts the input into fewer than 2 items; split at the item '
                                    "starts instead"}


def test_separator_that_cuts_one_piece_falls_back_to_the_model_split(tmp_path):
    # A separator present only once at the very start (e.g. a header rule) still leaves one item holding everything.
    spec = recruiting_fan_out_spec()
    spec["steps"][1]["config"]["separator"] = "=== packet ==="
    model = _model()
    ex, _store, _ = executor_for(tmp_path, model)
    content = recruiting_fan_out_sample()
    content["packet"] = "=== packet ===\n" + content["packet"]
    _start(ex, spec, content)
    run = ex.advance("o", "i.r1")
    assert model.count("step", "split") == 1
    assert [i["label"] for i in run["items"]] == NAMES
    boundaries = next(c for c in _rec(run, "split")["checks"] if c["id"] == "item_boundaries")
    assert boundaries["detail"] == ('separator "=== packet ===" cuts the input into fewer than 2 items; split at the '
                                    "item starts instead")


def test_separator_fallback_keeps_boundary_problems_beside_the_note(tmp_path):
    spec = recruiting_fan_out_spec()
    spec["steps"][1]["config"]["separator"] = "=== next ==="
    model = _model(split=lambda _i: {"items": [{"label": "x", "start_quote": "a quote that is nowhere at all"}]})
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex, spec)
    run = ex.advance("o", "i.r1")
    record = _rec(run, "split")
    assert record["state"] == "needs_attention"
    boundaries = next(c for c in record["checks"] if c["id"] == "item_boundaries")
    assert boundaries["passed"] is False
    assert boundaries["detail"] == ('separator "=== next ===" not found in the input; split at the item starts '
                                    "instead; start_quote 1 not found")


def test_a_failed_check_stops_only_its_item(tmp_path):
    model = _model(extract=_hollow_extract)
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex)
    run = ex.advance("o", "i.r1")
    keys = _keys(run)
    record = _rec(run, "extract")
    states = {i["key"]: i["state"] for i in record["items"]}
    assert states == {keys["Priya Shah"]: "needs_attention", keys["Marco Silva"]: "completed",
                      keys["Li Wei"]: "completed"}
    priya = next(i for i in record["items"] if i["key"] == keys["Priya Shah"])
    assert priya["error"] == {"code": "guardrail_failed", "checks": ["found_fields"]}
    assert record["state"] == "needs_attention"
    assert record["error"] == {"code": "items_need_attention", "items": [keys["Priya Shah"]]}
    assert run["state"] == "needs_attention" and run["code"] == "items_need_attention"
    assert record["output"] is None


def test_completed_items_are_not_rerun_on_readvance(tmp_path):
    model = _model(extract=_hollow_extract)
    ex, store, _ = executor_for(tmp_path, model)
    _start(ex)
    ex.advance("o", "i.r1")
    assert len(_calls(model, "extract")) == 3
    with store.locked("o"):  # what a resume does: the step runs again on the same attempt
        run = store.read("o", "runs", "i.r1")
        run["state"] = "running"
        _rec(run, "extract")["state"] = "pending"
        store.write("o", "runs", "i.r1", run)
    run = ex.advance("o", "i.r1")
    assert len(_calls(model, "extract")) == 3
    assert run["code"] == "items_need_attention"


def test_resplit_same_packet_hits_item_cache(tmp_path):
    model = _model()
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex)
    first = ex.advance("o", "i.r1")
    calls = {s: len(_calls(model, s)) for s in ("split", "extract", "outreach")}
    decisions = model.count("decide")
    ex.retry("o", "i.r1", "split")
    run = ex.advance("o", "i.r1")
    assert run["state"] == "waiting_approval"
    assert len(_calls(model, "split")) == calls["split"] + 1
    assert len(_calls(model, "extract")) == calls["extract"] and len(_calls(model, "outreach")) == calls["outreach"]
    assert model.count("decide") == decisions
    assert run["items"] == first["items"]
    assert run["model_calls"] == first["model_calls"] + 1


def test_single_case_run_record_shape_unchanged(tmp_path):
    ex, _store, _ = executor_for(tmp_path, FakeModel())
    ex.start(org_id="o", run_id="i.r1", instance_id="i", version=1,
             spec=WorkflowSpec.from_dict(copy.deepcopy(ACCOUNTING["spec"])),
             input_payload={"source_kind": "supplied_input", "content": ACCOUNTING["sample"]})
    run = ex.advance("o", "i.r1")
    assert run["state"] == "waiting_approval"
    assert "items" not in run and all("items" not in s for s in run["steps"])


# --- item failures: one automatic retry for a failed call, none for a failed check -----------------------------

def test_failed_item_model_call_is_retried_once_automatically(tmp_path):
    model = _model(extract=_flaky(1))
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex)
    run = ex.advance("o", "i.r1")
    assert run["state"] == "waiting_approval"
    record = _rec(run, "extract")
    marco = next(i for i in record["items"] if i["label"] == "Marco Silva")
    assert marco["state"] == "completed" and marco["auto_retried"] is True and marco["attempt"] == 1
    assert all("auto_retried" not in i for i in record["items"] if i["label"] != "Marco Silva")
    assert len(_calls(model, "extract", "Marco Silva")) == 2
    step_calls = sum(1 for c in model.calls if c["operation"] in ("step", "decide"))
    assert run["model_calls"] == step_calls  # the failed call is counted too
    assert run["human_interventions"] == 0


def test_item_failing_twice_pauses_step(tmp_path):
    model = _model(extract=_flaky(5))
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex)
    run = ex.advance("o", "i.r1")
    keys = _keys(run)
    record = _rec(run, "extract")
    marco = next(i for i in record["items"] if i["label"] == "Marco Silva")
    assert marco["state"] == "failed" and marco["error"] == {"code": "provider_failed"} and marco["auto_retried"]
    assert record["error"] == {"code": "items_need_attention", "items": [keys["Marco Silva"]]}
    assert len(_calls(model, "extract", "Marco Silva")) == 2
    assert _rec(run, "assess")["state"] == "pending" and model.count("decide") == 0


def test_failed_check_is_not_retried_automatically(tmp_path):
    model = _model(extract=_hollow_extract)
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex)
    ex.advance("o", "i.r1")
    assert len(_calls(model, "extract", "Priya Shah")) == 1


def test_item_exception_fails_only_that_item(tmp_path):
    model = _model(extract=_flaky(5, exc=lambda: RuntimeError("boom")))
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex)
    run = ex.advance("o", "i.r1")
    record = _rec(run, "extract")
    states = {i["label"]: (i["state"], i["error"]) for i in record["items"]}
    assert states == {"Priya Shah": ("completed", None), "Marco Silva": ("failed", {"code": "step_error"}),
                      "Li Wei": ("completed", None)}
    assert run["code"] == "items_need_attention"


def test_single_case_model_failure_not_retried(tmp_path):
    def broken(_input_data):
        raise ModelError("provider_failed")
    model = FakeModel(step_overrides={"extract": broken})
    ex, _store, _ = executor_for(tmp_path, model)
    ex.start(org_id="o", run_id="i.r1", instance_id="i", version=1,
             spec=WorkflowSpec.from_dict(copy.deepcopy(ACCOUNTING["spec"])),
             input_payload={"source_kind": "supplied_input", "content": ACCOUNTING["sample"]})
    run = ex.advance("o", "i.r1")
    assert model.count("step", "extract") == 1
    assert run["state"] == "failed" and _rec(run, "extract")["error"] == {"code": "provider_failed"}


# --- parallel items, scaled caps, pre-flight budget, aggregate size ------------------------------------------------

FIVE = ("Ada Park", "Ben Ode", "Cy Lee", "Dee Roe", "Eli Fox")


def _five_sample():
    blocks = [f"Applicant: {n}\nExperience: {i + 2} years\nCurrent title: Engineer {i}" for i, n in enumerate(FIVE)]
    return {"packet": "\n\n".join(blocks), "role_brief": "Senior engineer."}


def _concurrency_probe(delays=None):
    base = fan_out_fake_outputs()["extract"]
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    def extract(input_data):
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep((delays or {}).get(input_data["inputs"]["step:split"]["label"], 0.05))
        with guard:
            state["active"] -= 1
        return base(input_data)
    return extract, state


def test_default_parallelism_is_one_item_at_a_time(tmp_path):
    assert DEFAULT_PARALLEL_ITEMS == 1  # measured: concurrent calls finish one after another
    extract, state = _concurrency_probe()
    ex, _store, _ = executor_for(tmp_path, _model(extract=extract))
    assert ex.max_parallel_items == 1
    _start(ex, content=_five_sample())
    ex.advance("o", "i.r1")
    assert state["max"] == 1


def test_items_run_in_parallel_bounded(tmp_path):
    extract, state = _concurrency_probe()
    ex, _store, _ = executor_for(tmp_path, _model(extract=extract))
    ex.max_parallel_items = 3
    _start(ex, content=_five_sample())
    run = ex.advance("o", "i.r1")
    assert state["max"] == 3
    assert run["state"] == "waiting_approval"


def test_parallel_merge_is_deterministic_in_roster_order(tmp_path):
    extract, _state = _concurrency_probe({"Ada Park": 0.2, "Ben Ode": 0.0, "Cy Lee": 0.1, "Dee Roe": 0.0})
    ex, _store, _ = executor_for(tmp_path, _model(extract=extract))
    ex.max_parallel_items = 3
    _start(ex, content=_five_sample())
    run = ex.advance("o", "i.r1")
    assert [i["label"] for i in _rec(run, "extract")["items"]] == list(FIVE)
    assert [i["label"] for i in _rec(run, "extract")["output"]["items"]] == list(FIVE)


def test_scaled_model_cap_allows_n_items_and_pauses_beyond(tmp_path, monkeypatch):
    # split 1 + per item (extract 1 + outreach 1; the decision port settles assess and route) = 7 calls; the
    # worst case the pre-flight counts is 14 (assess and route could each need a model call).
    model = _model()
    ex, _store, _ = executor_for(tmp_path, model, cap=2)
    ex.per_item_model_calls = 4  # 2 + 4 x 3 = 14
    _start(ex)
    run = ex.advance("o", "i.r1")
    assert run["state"] == "waiting_approval" and run["model_calls"] == 7

    # A cap can still stop the run partway when the estimate is short (automatic retries are not in it).
    monkeypatch.setattr(WorkflowExecutor, "_worst_case_calls", lambda self, org_id, step, catalog, depth=0: 0)
    model = _model()
    ex, _store, _ = executor_for(tmp_path / "b", model, cap=2)
    ex.per_item_model_calls = 1  # 2 + 1 x 3 = 5: split 1 + extract 3 + one outreach
    _start(ex)
    run = ex.advance("o", "i.r1")
    record = _rec(run, "outreach")
    assert record["state"] == "needs_attention" and record["error"] == {"code": "model_call_cap", "cap": 5}
    assert [i["state"] for i in record["items"]] == ["completed", "pending", "pending"]
    assert run["model_calls"] == 5 and run["code"] == "model_call_cap"


def test_scaled_caps_respect_the_ceiling(tmp_path):
    ex, _store, _ = executor_for(tmp_path, _model(), cap=2)
    ex.per_item_model_calls = 100
    ex.model_call_ceiling = 4
    assert ex._caps({"items": [{"status": "active"}] * 3}) == (4, 400 + 60 * 3)
    assert ex._caps({"items": [{"status": "active"}, {"status": "excluded"}]})[0] == 4
    assert ex._caps({}) == (2, 400)


def test_preflight_budget_pauses_before_item_spend(tmp_path):
    # Worst case per item: extract 1 + assess 1 + route 1 + outreach 1 = 4; x3 + summary 1 + split 1 spent = 14.
    model = _model()
    ex, _store, _ = executor_for(tmp_path, model)
    ex.model_call_ceiling = 13
    _start(ex)
    run = ex.advance("o", "i.r1")
    split = _rec(run, "split")
    assert split["state"] == "needs_attention"
    assert split["error"] == {"code": "fan_out_budget", "estimate": 14, "ceiling": 13}
    assert run["code"] == "fan_out_budget" and "items" not in run
    assert model.count("step") == 1 and run["model_calls"] == 1

    ex2, _store2, _ = executor_for(tmp_path / "b", _model())
    ex2.model_call_ceiling = 14
    _start(ex2)
    assert ex2.advance("o", "i.r1")["state"] == "waiting_approval"


class Crash(BaseException):
    """Stands in for the process dying mid-step: nothing in the executor catches it."""


def test_crash_mid_segment_resumes_without_recounting(tmp_path):
    base = fan_out_fake_outputs()["extract"]
    state = {"crash": True}

    def extract(input_data):
        if input_data["inputs"]["step:split"]["label"] == "Li Wei" and state["crash"]:
            state["crash"] = False
            raise Crash()
        return base(input_data)
    model = _model(extract=extract)
    ex, store, _ = executor_for(tmp_path, model)
    _start(ex)
    try:
        ex.advance("o", "i.r1")
    except Crash:
        pass
    assert store.read("o", "runs", "i.r1")["state"] == "running"
    crashed_calls = 1
    resumed = WorkflowExecutor(store=ex.store, model=ex.model, decisions=ex.decisions, catalog_for=ex.catalog_for,
                               membership=ex.membership, notifier=ex.notifier, clock=ex.clock)
    run = resumed.advance("o", "i.r1")
    assert run["state"] == "waiting_approval"
    assert len(_calls(model, "extract")) == 3 + crashed_calls  # Priya and Marco came from the cache
    made = sum(1 for c in model.calls if c["operation"] in ("step", "decide")) - crashed_calls
    assert run["model_calls"] == made


def _aggregate_spec():
    spec = recruiting_fan_out_spec()
    spec["steps"] = [s for s in spec["steps"] if s["id"] in ("collect", "split", "extract")]
    spec["steps"] += [
        {"id": "summary", "role": "r", "instructions": "Rank them.", "primitive": "summarize_report",
         "config": {"sections": ["Ranking"]}, "inputs": ["step:extract"]},
        {"id": "package", "role": "r", "instructions": "Pack.", "primitive": "deliver_package",
         "config": {"title": "t"}, "inputs": ["step:summary"]}]
    return spec


def test_aggregate_too_large(tmp_path, monkeypatch):
    model = _model()
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex, _aggregate_spec())
    monkeypatch.setattr(executor_module, "MAX_AGGREGATE_BYTES", 500)
    run = ex.advance("o", "i.r1")
    record = _rec(run, "summary")
    assert record["state"] == "needs_attention" and record["error"]["code"] == "aggregate_too_large"
    assert record["error"]["limit"] == 500 and record["error"]["size"] > 500
    assert model.count("step", "summary") == 0

    monkeypatch.setattr(executor_module, "MAX_AGGREGATE_BYTES", 256 * 1024)
    ex.retry("o", "i.r1", "summary")
    assert ex.advance("o", "i.r1")["state"] == "completed"


def test_single_case_caps_unchanged(tmp_path):
    model = FakeModel()
    ex, _store, _ = executor_for(tmp_path, model, cap=2)
    assert ex._caps({"steps": []}) == (2, 400)
    ex.start(org_id="o", run_id="i.r1", instance_id="i", version=1,
             spec=WorkflowSpec.from_dict(copy.deepcopy(ACCOUNTING["spec"])),
             input_payload={"source_kind": "supplied_input", "content": ACCOUNTING["sample"]})
    run = ex.advance("o", "i.r1")
    assert run["code"] == "model_call_cap" and run["model_calls"] == 2
    assert next(s for s in run["steps"] if s["state"] == "needs_attention")["error"] == {"code": "model_call_cap",
                                                                                         "cap": 2}


# --- one decision review per step across items ---------------------------------------------------------------------

CRITERIA = ("python", "sql", "leadership", "pipelines")


class ContextPort:
    """Decision port double that answers from the item context: low confidence for the people in `low`, and an
    unexpected error for the people in `broken` (that item fails)."""

    def __init__(self, low=(), broken=()):
        self.low, self.broken, self.calls = low, broken, []

    def decide(self, items, *, rubric, org_id):
        self.calls.append(list(items))
        if any(n in item.context for item in items for n in self.broken):
            raise RuntimeError("decider bug")
        out = []
        for item in items:
            confidence = 0.3 if any(n in item.context for n in self.low) else 0.95
            out.append(DecisionAnswer(id=item.id, choice=item.options[0], confidence=confidence,
                                      provider="context-decider", model="context-1"))
        return out


def _review_model():
    return _model()


def _review(store, run, step_id="assess"):
    return store.read("o", "approvals", _rec(run, step_id)["approval_id"])


def test_low_confidence_items_produce_one_review_for_the_step(tmp_path):
    ex, store, notifier = executor_for(tmp_path, _review_model(), decisions=ContextPort(low=("Marco Silva", "Li Wei")))
    _start(ex)
    run = ex.advance("o", "i.r1")
    keys = _keys(run)
    assert run["state"] == "waiting_approval" and _rec(run, "assess")["state"] == "waiting_approval"
    review = _review(store, run)
    assert review["kind"] == "decision_review" and review["status"] == "pending"
    assert review["what_is_approved"] == "8 decision(s) across 2 item(s)"
    assert [i["id"] for i in review["items"]] == ([f"{keys['Marco Silva']}/{c}" for c in CRITERIA]
                                                 + [f"{keys['Li Wei']}/{c}" for c in CRITERIA])
    marco = review["items"][0]
    assert marco["item_key"] == keys["Marco Silva"] and marco["item_label"] == "Marco Silva"
    assert "Marco Silva" in marco["context_excerpt"] and "Priya Shah" not in marco["context_excerpt"]
    assert len([a for a in store.list("o", "approvals") if a["kind"] == "decision_review"]) == 1
    assert len(notifier.events) == 1
    states = {i["label"]: i["state"] for i in _rec(run, "assess")["items"]}
    assert states == {"Priya Shah": "completed", "Marco Silva": "waiting_review", "Li Wei": "waiting_review"}


def _choose_all(ex, store, run, choice="3"):
    review = _review(store, run)
    return ex.decide("o", review["approval_id"], actor="o", decision="choose",
                     content_digest=review["content_digest"], choices={i["id"]: choice for i in review["items"]})


def test_choose_settles_each_item_and_step_completes(tmp_path):
    ex, store, _ = executor_for(tmp_path, _review_model(), decisions=ContextPort(low=("Marco Silva", "Li Wei")))
    _start(ex)
    run = _choose_all(ex, store, ex.advance("o", "i.r1"))
    assert run["state"] == "running" and run["human_interventions"] == 1
    record = _rec(run, "assess")
    assert record["state"] == "completed" and [i["state"] for i in record["items"]] == ["completed"] * 3
    assert all("pending_outcomes" not in i and "review" not in i for i in record["items"])
    scores = {i["label"]: [s["score"] for s in i["output"]["scores"]] for i in record["output"]["items"]}
    assert scores == {"Priya Shah": [0, 0, 0, 0], "Marco Silva": [3, 3, 3, 3], "Li Wei": [3, 3, 3, 3]}
    marco = next(i for i in record["items"] if i["label"] == "Marco Silva")
    assert [d["rung"] for d in marco["decisions"] if d["item_id"] == "python"][-1] == "human"
    run = ex.advance("o", "i.r1")
    assert run["state"] == "waiting_approval" and _rec(run, "approval")["state"] == "waiting_approval"


def test_choose_with_a_failed_item_leaves_items_need_attention(tmp_path):
    ex, store, _ = executor_for(tmp_path, _review_model(),
                                decisions=ContextPort(low=("Marco Silva",), broken=("Priya Shah",)))
    _start(ex)
    run = ex.advance("o", "i.r1")
    keys = _keys(run)
    assert _rec(run, "assess")["state"] == "waiting_approval"  # the review comes first
    run = _choose_all(ex, store, run)
    record = _rec(run, "assess")
    assert record["state"] == "needs_attention"
    assert record["error"] == {"code": "items_need_attention", "items": [keys["Priya Shah"]]}
    assert run["state"] == "needs_attention" and run["code"] == "items_need_attention"
    assert {i["label"]: i["state"] for i in record["items"]}["Marco Silva"] == "completed"


# --- review follow-ups ---------------------------------------------------------------------------------------------

def test_resplit_keeps_exclusions_and_retry_counts(tmp_path):
    model = _model(extract=_hollow_extract)
    ex, store, _ = executor_for(tmp_path, model)
    _start(ex)
    run = ex.advance("o", "i.r1")
    keys = _keys(run)
    ex.retry("o", "i.r1", "extract", item_key=keys["Priya Shah"])
    ex.advance("o", "i.r1")
    ex.exclude_item("o", "i.r1", keys["Priya Shah"], actor="o", note="not this round")
    ex.advance("o", "i.r1")
    ex.retry("o", "i.r1", "split")
    run = ex.advance("o", "i.r1")
    priya = next(i for i in run["items"] if i["key"] == keys["Priya Shah"])
    assert (priya["status"], priya["reason"], priya["retried"]) == ("excluded", "not this round", 1)
    assert run["state"] == "waiting_approval"


def test_unexpected_error_after_a_call_reuses_and_counts_it(tmp_path, monkeypatch):
    real = executor_module.apply_guardrails
    state = {"raised": False}

    def flaky_guardrails(primitive, step, output, inputs, spec):
        if step.id == "extract" and not state["raised"]:
            state["raised"] = True
            raise RuntimeError("post-processing bug")
        return real(primitive, step, output, inputs, spec)
    monkeypatch.setattr(executor_module, "apply_guardrails", flaky_guardrails)
    model = _model()
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex)
    run = ex.advance("o", "i.r1")
    assert run["state"] == "waiting_approval"
    assert len(_calls(model, "extract")) == 3  # the retry reused the cached call
    priya = next(i for i in _rec(run, "extract")["items"] if i["label"] == "Priya Shah")
    assert priya["auto_retried"] is True and priya["attempt"] == 0
    assert run["model_calls"] == sum(1 for c in model.calls if c["operation"] in ("step", "decide"))


def test_no_second_automatic_retry_after_a_cap_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(WorkflowExecutor, "_worst_case_calls", lambda self, org_id, step, catalog, depth=0: 0)
    model = _model(extract=_flaky(5))
    ex, store, _ = executor_for(tmp_path, model, cap=3)
    ex.per_item_model_calls = 0  # split, Priya, Marco's failed call: the cap stops Marco's automatic retry
    _start(ex)
    run = ex.advance("o", "i.r1")
    assert run["code"] == "model_call_cap"
    marco = next(i for i in _rec(run, "extract")["items"] if i["label"] == "Marco Silva")
    assert marco["state"] == "pending" and marco["auto_retried"] is True
    ex.model_call_cap = 20
    ex.retry("o", "i.r1", "extract", item_key=_keys(run)["Li Wei"])  # Li never ran; this continues the run
    run = ex.advance("o", "i.r1")
    marco = next(i for i in _rec(run, "extract")["items"] if i["label"] == "Marco Silva")
    assert marco["state"] == "failed" and len(_calls(model, "extract", "Marco Silva")) == 2


def test_excluding_items_does_not_lower_the_cap(tmp_path):
    ex, _store, _ = executor_for(tmp_path, _model())
    roster = [{"status": "active"}, {"status": "excluded"}, {"status": "excluded"}]
    assert ex._caps({"items": roster}) == (40 + 6 * 3, 400 + 60 * 3)


def test_aggregate_check_covers_readers_of_the_item_list(tmp_path, monkeypatch):
    spec = _aggregate_spec()
    spec["steps"][3]["inputs"] = ["step:split"]
    model = _model()
    ex, _store, _ = executor_for(tmp_path, model)
    _start(ex, spec)
    monkeypatch.setattr(executor_module, "MAX_AGGREGATE_BYTES", 300)
    run = ex.advance("o", "i.r1")
    assert _rec(run, "summary")["error"]["code"] == "aggregate_too_large"


def test_review_items_needs_item_keys():
    from daoharness.executor import ExecutorError

    with pytest.raises(ExecutorError) as info:
        WorkflowExecutor._valid_item_decisions({"item_keys": [], "content": {}}, {})
    assert info.value.code == "invalid_item_decisions"
