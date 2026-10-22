"""Stand-in model and decision ports for demos and tests: deterministic, no provider, no network, no database."""
from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass, field
from typing import Any

from .decision import NONE_OPTION, DecisionAnswer, DecisionUnavailable
from .guardrails import flatten_text
from .ports import ModelError, ModelReceipt
from .spec import REF, digest


def whole_words(text: str, limit: int) -> str:
    """Leading text of at most `limit` characters, cut at a word boundary (models quote whole words)."""
    if len(text) <= limit or text[limit].isspace():
        return text[:limit].rstrip()
    return text[:limit].rsplit(" ", 1)[0].rstrip()


def _default_decision(item: dict, context: str) -> dict:
    choice = next((o for o in item["options"] if o != NONE_OPTION), item["options"][0])
    return {"choice": choice, "confidence": 0.95, "evidence_quotes": [whole_words(context, 40)], "rationale": "fixture"}


def fill_from_schema(schema):
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object":
        return {k: fill_from_schema(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [fill_from_schema(schema["items"])]
    if t == "boolean":
        return True
    if t in ("integer", "number"):
        return schema.get("minimum", 1)
    return "fixture"


def default_step_output(input_data: dict, schema: dict) -> dict:
    step, inputs = input_data["step"], input_data["inputs"]
    primitive, config = step["primitive"], step["config"]
    # Skip ref labels such as "input:documents" (collect_inputs packets carry them) so quotes stay verbatim.
    texts = [t for t in flatten_text(inputs) if t.strip() and not REF.fullmatch(t)]
    first = (whole_words(texts[0].split(".")[0].strip(), 80) if texts else "") or "fixture"
    if primitive == "extract_fields":
        return {"fields": [{"name": f["name"], "found": True, "value": first, "source_quotes": [first]}
                           for f in config["fields"]]}
    # checklist_match / screen_score / classify_route never reach here: they use the decision executor
    # (FakeDecisionPort / FakeModel "decide" handler).
    if primitive == "draft_message":
        return {"subject": "Next steps", "body": "Hello, please send the remaining items. Reference 123-45-6789.",
                "notes": ""}
    if primitive == "summarize_report":
        steps = [r.split(":", 1)[1] for r in inputs if r.startswith("step:")]
        return {"title": "Summary", "summary": "All steps ran.",
                "sections": [{"title": "Overview", "body": "See steps.", "source_steps": steps[:3]}],
                "criteria": [{"criterion": c, "met": True, "evidence": "fixture"}
                             for c in input_data["completion_criteria"]]}
    return fill_from_schema(schema)


@dataclass
class FakeModel:
    decide_overrides: dict = field(default_factory=dict)  # item_id -> partial answer dict merged over the default
    compile_outputs: list = field(default_factory=list)   # dicts or callables(input_data) -> dict, consumed in order
    step_overrides: dict = field(default_factory=dict)    # step_id -> callable(input_data) -> dict (may raise)
    fail_operations: set = field(default_factory=set)
    calls: list = field(default_factory=list)
    _ids: Any = field(default_factory=lambda: itertools.count(1))

    def generate(self, *, org_id, operation, skill_id, input_data, instructions, evidence, schema):
        self.calls.append({"org_id": org_id, "operation": operation, "skill_id": skill_id,
                           "input": copy.deepcopy(dict(input_data)), "instructions": instructions,
                           "evidence": list(evidence), "schema": schema})
        if operation in self.fail_operations:
            raise ModelError("provider_failed")
        if operation == "decide":
            contexts = input_data["contexts"]  # items refer to their context by context_id
            content = {item["id"]: {**_default_decision(item, contexts[item["context_id"]]),
                                    **copy.deepcopy(self.decide_overrides.get(item["id"], {}))}
                       for item in input_data["items"]}
        elif operation == "compile":
            item = self.compile_outputs.pop(0)
            content = item(input_data) if callable(item) else copy.deepcopy(item)
        else:
            override = self.step_overrides.get(input_data["step"]["id"])
            content = override(input_data) if override else default_step_output(input_data, schema)
        n = next(self._ids)
        return content, ModelReceipt(run_id=f"fake-{n}", provider="fake", model="fake-model",
                                     output_digest=digest(content))

    def count(self, operation=None, step_id=None):
        return sum(1 for c in self.calls if (operation is None or c["operation"] == operation)
                   and (step_id is None or c["input"].get("step", {}).get("id") == step_id))


@dataclass
class FakeDecisionPort:
    answers: dict = field(default_factory=dict)  # item_id -> partial DecisionAnswer fields (choice, confidence, ...)
    unavailable: bool = False
    calls: list = field(default_factory=list)

    def decide(self, items, *, rubric, org_id):
        self.calls.append({"items": list(items), "rubric": rubric, "org_id": org_id})
        if self.unavailable:
            raise DecisionUnavailable("decider_unreachable")
        out = []
        for item in items:
            fields = {"choice": item.options[0], "confidence": 0.95,
                      "evidence": (whole_words(item.context, 40),) if item.context.strip() else (),
                      **self.answers.get(item.id, {})}
            out.append(DecisionAnswer(id=item.id, provider="fake-decider", model="fake-decider-1", **fields))
        return out


class Clock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        self.t += 1
        return f"2026-01-01T00:{self.t // 60 % 60:02d}:{self.t % 60:02d}Z"


class LocalMembership:
    def resolve(self, org_id, approver_role):
        return org_id


class RecordingNotifier:
    def __init__(self):
        self.events = []

    def approval_requested(self, org_id, approval):
        self.events.append((org_id, approval["approval_id"]))


def as_model_payload(spec_dict: dict, *, gaps=()) -> dict:
    """Shape a spec dict the way the compiler schema returns it (config as config_json)."""
    steps = []
    for s in spec_dict["steps"]:
        step = {k: v for k, v in s.items() if k not in ("config", "primitive_version", "origin", "guardrails")}
        step["config_json"] = json.dumps(s.get("config", {}))
        step.setdefault("required_sources", [])
        step.setdefault("approval", None)
        step.setdefault("inputs", [])
        step.setdefault("for_each", None)
        steps.append(step)
    spec = {**{k: v for k, v in spec_dict.items() if k != "steps"}, "steps": steps}
    return {"spec": spec, "declared_gaps": list(gaps)}


def executor_for(tmp_path, model, *, decisions=None, connected=frozenset(), cap=40):
    """WorkflowExecutor over a file store under tmp_path; decisions default to a FakeDecisionPort (pass
    decisions=False to have the model decide)."""
    from daoharness.decision import DecisionRouter, ModelDecisions
    from daoharness.executor import WorkflowExecutor
    from daoharness.file_store import FileWorkflowStore
    from daoharness.primitives import builtin_catalog

    store = FileWorkflowStore(tmp_path / "wf")
    notifier = RecordingNotifier()
    provider = None if decisions is False else decisions if decisions is not None else FakeDecisionPort()
    router = DecisionRouter(model=ModelDecisions(model), provider=provider)
    catalog = builtin_catalog()
    executor = WorkflowExecutor(store=store, model=model, decisions=router, catalog_for=lambda org: catalog,
                                membership=LocalMembership(), notifier=notifier, clock=Clock(),
                                connected_sources=frozenset(connected), model_call_cap=cap)
    return executor, store, notifier


def service_for(tmp_path, model, *, decisions=None, submit=None, **kw):
    """WorkflowService over a file store under tmp_path, local membership (every role resolves to the org),
    a recording notifier, the deterministic Clock and an inline submit unless one is given. Decisions default to
    a FakeDecisionPort (pass decisions=False to have the model decide)."""
    from daoharness.file_store import FileWorkflowStore
    from daoharness.service import WorkflowService

    return WorkflowService(
        store=FileWorkflowStore(tmp_path / "wf"), model=model, membership=LocalMembership(),
        notifier=RecordingNotifier(), clock=Clock(),
        submit=submit if submit is not None else (lambda fn: fn()),
        decisions=None if decisions is False else decisions if decisions is not None else FakeDecisionPort(), **kw)
