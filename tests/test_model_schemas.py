"""Guard: every JSON schema the workflow layer sends to a model is provider strict-mode safe.

Strict structured output (OpenAI-style strict mode) needs every object closed (additionalProperties false)
with every property required, an object at the root, and regex patterns in a plain dialect: no "(?...)" groups.
"""
import pytest

from daoharness.compiler import WorkflowCompiler, compile_schema
from daoharness.decision import DecisionItem, ModelDecisions
from daoharness.primitives import BUILTIN_PRIMITIVES, builtin_catalog
from daoharness.validator import ValidationContext
from industry_fixtures import ACCOUNTING
from workflow_fakes import FakeModel, as_model_payload, service_for

BUILTIN_IDS = builtin_catalog().ids()

# Representative configs for every model-executor builtin (the output schema may depend on the config).
MODEL_STEP_CONFIGS = {
    "extract_fields": [{"fields": [{"name": "a", "description": "A", "required": True}]},
                       {"fields": [{"name": f"f{n}", "description": "F", "required": n % 2 == 0} for n in range(5)]}],
    "draft_message": [{"channel": "email", "audience": "recipient", "external": True},
                      {"channel": "document", "audience": "team", "external": False, "rules": ["Be brief."]}],
    "generate_questions": [{"max_questions": 3, "focus": "gaps"}, {}],
    "summarize_report": [{"sections": ["Overview"]}, {"sections": []}],
    "generic_model_step": [{"output_fields": [{"name": "answer", "description": "Answer"}]},
                           {"output_fields": [{"name": f"o{n}", "description": ""} for n in range(20)]}],
    "split_items": [{"item_noun": "entry"}, {"item_noun": "entry", "max_items": 50, "separator": "---"}],
}


def _walk(node, path="$"):
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("enum", "required", "const", "default", "examples"):
                continue
            if key in ("properties", "$defs"):
                for name, child in value.items():
                    yield from _walk(child, f"{path}.{key}.{name}")
            elif isinstance(value, dict):
                yield from _walk(value, f"{path}.{key}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        yield from _walk(item, f"{path}.{key}[{i}]")


def _problems(schema):
    problems = []
    if schema.get("type") != "object":
        problems.append("$: root is not an object")
    for path, node in _walk(schema):
        if node.get("type") == "object" or "properties" in node:
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties is not false")
            if sorted(node.get("required") or []) != sorted(node.get("properties") or {}):
                problems.append(f"{path}: required != properties")
        pattern = node.get("pattern")
        if isinstance(pattern, str) and "(?" in pattern:
            problems.append(f"{path}: pattern {pattern!r} uses a (?...) group")
    return problems


def _ctx():
    return ValidationContext(builtin_catalog(), lambda r: True)


def _decision_items():
    items = [DecisionItem(id="i1", question="Present?", options=("present", "missing", "unclear"), context="c1"),
             DecisionItem(id="rule_1", question="Complies?", options=("yes", "no"), context="c2", kind="yes_no")]
    catalog = builtin_catalog()
    configs = {"checklist_match": {"items": [{"id": "x", "label": "X", "description": "X"}]},
               "screen_score": {"criteria": [{"id": "c", "label": "C", "description": "C", "weight": 2}],
                                "scale_max": 5},
               "classify_route": {"categories": [{"id": "a", "label": "A", "description": "A"},
                                                 {"id": "b", "label": "B", "description": "B"}]}}
    for pid, config in configs.items():
        items += catalog.get(pid).decision_items(config, {"input:notes": "Some notes."}, "Judge carefully.")
    return items


def _captured_schemas(tmp_path):
    """Schemas captured from real calls: a build, a run of the built workflow (its steps and the model's
    decisions) and a decision call with mixed items."""
    model = FakeModel(compile_outputs=[as_model_payload(ACCOUNTING["spec"]), as_model_payload(ACCOUNTING["spec"])])
    WorkflowCompiler(model).compile(org_id="org-1", description=ACCOUNTING["description"], ctx=_ctx(),
                                    primitive_refs=list(BUILTIN_IDS))
    svc = service_for(tmp_path, model, decisions=False)
    created = svc.create("org-1", "build-1", ACCOUNTING["description"])
    iid = created["instance"]["instance_id"]
    svc.confirm("org-1", iid, "save-1", created["draft_id"])
    svc.run("org-1", iid, "run-1", {"version": 1, "content": ACCOUNTING["sample"]})
    ModelDecisions(model).decide(_decision_items(), rubric="r", org_id="org-1")
    return {f"call[{n}].{c['operation']}": c["schema"] for n, c in enumerate(model.calls)}


def _built_schemas():
    schemas = {
        "compile": compile_schema(list(BUILTIN_IDS)),
        "decide.single": ModelDecisions.schema_for(_decision_items()[:1]),
        "decide.mixed": ModelDecisions.schema_for(_decision_items()),
    }
    for p in BUILTIN_PRIMITIVES:
        if p.executor == "model":
            for n, config in enumerate(MODEL_STEP_CONFIGS[p.id]):
                schemas[f"step.{p.id}[{n}]"] = p.output_schema_for(config)
    return schemas


def test_every_model_executor_builtin_has_representative_configs():
    assert {p.id for p in BUILTIN_PRIMITIVES if p.executor == "model"} == set(MODEL_STEP_CONFIGS)


@pytest.mark.parametrize("name", sorted(_built_schemas()))
def test_built_model_schema_is_provider_strict(name):
    assert _problems(_built_schemas()[name]) == []


def test_schemas_captured_from_real_calls_are_provider_strict(tmp_path):
    captured = _captured_schemas(tmp_path)
    operations = {key.split(".", 1)[1] for key in captured}
    assert operations == {"compile", "step", "decide"}
    assert {key: _problems(schema) for key, schema in captured.items() if _problems(schema)} == {}


def test_guard_detects_violations():
    loose = {"type": "object", "properties": {"a": {"type": "string", "pattern": "^a(?!\\n)$"}}, "required": []}
    found = _problems(loose)
    assert any("additionalProperties" in p for p in found)
    assert any("required != properties" in p for p in found)
    assert any("(?...)" in p for p in found)
    assert _problems({"type": "array", "items": {"type": "string"}}) == ["$: root is not an object"]
