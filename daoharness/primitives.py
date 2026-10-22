"""Built-in step primitives (industry-neutral).

Executors:
- "model": a general model fills output_schema_for(config) under the primitive's instructions.
- "decision": the primitive turns config + inputs into DecisionItems (decision_items); the executor routes them
  through DecisionRouter and the primitive builds its output from the settled outcomes (assemble).
- "deterministic": the executor computes the output itself.
- "human": the output comes from a person (approval inbox).
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .decision import MAX_CONTEXT, NONE_OPTION, DecisionItem, DecisionOutcome
from .guardrails import Check, flatten_text
from .items import DEFAULT_MAX_ITEMS, MAX_ITEMS, build_items, slice_by_boundaries

# Schema helpers. Every object built with obj() is compatible with strict structured output: additionalProperties
# false and every property required. Config schemas with optional keys pass `required` explicitly.
def S(n: int = 4000) -> dict:
    return {"type": "string", "maxLength": n}


def obj(props: Mapping[str, Any], required: Iterable[str] | None = None) -> dict:
    return {"type": "object", "additionalProperties": False, "properties": dict(props),
            "required": list(props) if required is None else list(required)}


def arr(items: Mapping[str, Any], lo: int = 0, hi: int = 200) -> dict:
    return {"type": "array", "items": dict(items), "minItems": lo, "maxItems": hi}


# "(?!\n)" keeps the end anchor strict under Python's re.search, where "$" also matches before a final newline.
ID = {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,47}(?!\\n)$"}
B = {"type": "boolean"}
THRESHOLD = {"type": "number", "minimum": 0.5, "maximum": 0.99}

DECISION_RUBRIC = ("Decide each item only from what the context shows. If it does not show enough, say so: answer "
                   "unclear, a lower score or none_apply.")


@dataclass(frozen=True)
class PostContext:
    sources: list[str]
    input_step_ids: tuple[str, ...]
    completion_criteria: tuple[str, ...]
    upstream_step_ids: tuple[str, ...] = ()   # every step that fed this one, directly or transitively


PostFn = Callable[[Mapping, dict, PostContext], "tuple[dict, list[Check]]"]


@dataclass(frozen=True)
class Primitive:
    id: str
    version: int
    title: str
    purpose: str
    executor: str
    config_schema: Mapping[str, Any]
    instructions: str
    output_schema_for: Callable[[Mapping[str, Any]], dict]
    default_guardrails: tuple[str, ...] = ("schema",)
    # ids of the positive claims in an output that come without a quote (the "citations" check)
    cites: Callable[[Any], list[str]] | None = None
    required_paths: tuple[str, ...] = ()
    text_paths: tuple[str, ...] = ()
    external_key: str | None = None
    post: PostFn | None = None
    decision_items: Callable[[Mapping, Mapping[str, Any], str], list[DecisionItem]] | None = None
    assemble: Callable[[Mapping, list[DecisionOutcome]], dict] | None = None
    rubric: str = ""
    required_paths_from: Callable[[Mapping[str, Any]], tuple[str, ...]] | None = None  # per-config required paths

    def external_draft(self, config: Mapping[str, Any]) -> bool:
        return self.external_key is not None and bool(config.get(self.external_key, True))

    def required_paths_for(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        if self.required_paths_from is not None:
            return self.required_paths_from(config)
        return self.required_paths

    def describe(self) -> dict:
        return {"id": self.id, "version": self.version, "title": self.title, "purpose": self.purpose,
                "executor": self.executor, "config_schema": copy.deepcopy(dict(self.config_schema))}


class PrimitiveCatalog:
    def __init__(self, primitives: Iterable[Primitive]) -> None:
        self._by_id: dict[str, Primitive] = {}
        for p in primitives:
            if p.id in self._by_id:
                raise ValueError(f"duplicate_primitive: {p.id}")
            self._by_id[p.id] = p

    def get(self, primitive_id: str) -> Primitive | None:
        return self._by_id.get(primitive_id)

    def ids(self) -> list[str]:
        return list(self._by_id)

    def describe(self) -> list[dict]:
        return [p.describe() for p in self._by_id.values()]


# ---------------------------------------------------------------------------------------------------------------
# Shared helpers

def _rows(output: Any, key: str) -> list[dict]:
    value = output.get(key) if isinstance(output, Mapping) else None
    return [r for r in value if isinstance(r, Mapping)] if isinstance(value, list) else []


def _coverage(expected: Iterable[str], got: Iterable[Any]) -> Check:
    expected = list(expected)
    counts = Counter(g if isinstance(g, str) else repr(g) for g in got)  # malformed ids stay hashable
    missing = [i for i in expected if counts[i] == 0]
    duplicated = [i for i in expected if counts[i] > 1]
    unexpected = [str(i) for i in counts if i not in set(expected)]
    parts = [f"{label}: {', '.join(ids)}" for label, ids in
             (("missing", missing), ("duplicated", duplicated), ("unexpected", unexpected)) if ids]
    return Check("coverage", not parts, "; ".join(parts))


INPUT_SEPARATOR = "\n\n[[next input]]\n\n"


def _context(inputs: Mapping[str, Any]) -> str:
    """The input texts joined with a marker between them, cut to MAX_CONTEXT."""
    return INPUT_SEPARATOR.join(flatten_text(inputs))[:MAX_CONTEXT]


def _by_id(outcomes: Iterable[DecisionOutcome]) -> dict[str, DecisionOutcome]:
    return {o.item_id: o for o in outcomes}


def _first(evidence: tuple[str, ...]) -> str:
    return evidence[0] if evidence else ""


def _quoted(value: Any) -> bool:
    values = value if isinstance(value, list) else [value]
    return any(isinstance(v, str) and v.strip() for v in values)


def _uncited(rows_key: str, id_key: str, quote_key: str, claims: Callable[[Mapping], bool]) -> Callable[[Any], list[str]]:
    """cites for outputs shaped {rows_key: [{id_key, quote_key, ...}]}: the rows that claim something, unquoted."""
    def cites(output: Any) -> list[str]:
        return [str(r.get(id_key)) for r in _rows(output, rows_key) if claims(r) and not _quoted(r.get(quote_key))]
    return cites


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _norm(text: Any) -> str:
    return " ".join(str(text).split()).casefold()


# ---------------------------------------------------------------------------------------------------------------
# collect_inputs

COLLECT_INPUTS = Primitive(
    id="collect_inputs", version=1, title="Collect inputs",
    purpose="Gather the run inputs and upstream outputs into one packet.", executor="deterministic",
    config_schema=obj({}), instructions="",
    output_schema_for=lambda _c: obj({"packet": arr(obj({"ref": S(80), "text": S(200000)}))}))


# ---------------------------------------------------------------------------------------------------------------
# extract_fields

def _merge_field_rows(rows: list[dict]) -> list[dict]:
    """One row per field name, in first-seen order: a field returned once per passage is merged into one row. When
    every found field repeats, the input holds several records, so the rows are left alone and the coverage check
    reports the duplicates."""
    found = [r.get("name") for r in rows if r.get("found") is True]
    if len(set(found)) > 1 and all(found.count(name) > 1 for name in set(found)):
        return rows
    merged: dict[Any, dict] = {}
    for row in rows:
        kept = merged.setdefault(row.get("name"), row)
        if kept is row or row.get("found") is not True:
            continue
        if kept.get("found") is not True:
            merged[row.get("name")] = row
            continue
        before = kept["value"].split("; ") if isinstance(kept.get("value"), str) else []
        values = [v.strip() for v in [*before, row.get("value")] if isinstance(v, str) and v.strip()]
        kept["value"] = "; ".join(dict.fromkeys(values))
        quotes = [q for q in (kept.get("source_quotes") or []) + (row.get("source_quotes") or [])]
        kept["source_quotes"] = list(dict.fromkeys(quotes))
    return list(merged.values())


def _extract_post(config: Mapping, output: dict, ctx: PostContext) -> tuple[dict, list[Check]]:
    out = copy.deepcopy(output)
    if isinstance(out.get("fields"), list):
        out["fields"] = _merge_field_rows([r for r in out["fields"] if isinstance(r, dict)])
    rows = _rows(out, "fields")
    coverage = _coverage([f["name"] for f in config.get("fields", [])], [r.get("name") for r in rows])
    hollow = [str(r.get("name")) for r in rows if r.get("found") is True
              and not (isinstance(r.get("value"), str) and r["value"].strip()
                       and any(isinstance(q, str) and q.strip() for q in r.get("source_quotes") or ()))]
    return out, [coverage, Check("found_fields", not hollow, ", ".join(hollow))]


EXTRACT_FIELDS = Primitive(
    id="extract_fields", version=1, title="Extract fields",
    purpose="Pull configured fields out of the inputs, each with the verbatim passages it comes from.",
    executor="model",
    config_schema=obj({"fields": arr(obj({"name": ID, "description": S(1000), "required": B}), 1, 60)}),
    instructions=("Extract each configured field from the inputs. For every field return found, the value, and "
                  "source_quotes: the passages it comes from, each copied verbatim from the inputs. When the value "
                  "draws on several places, give each place as its own passage and never join passages into one. "
                  "If a field is absent set found=false, leave value empty and give no passages. Never guess."),
    output_schema_for=lambda _c: obj({"fields": arr(obj({
        "name": S(48), "found": B, "value": S(2000), "source_quotes": arr(S(2000), 0, 8)}))}),
    default_guardrails=("schema", "citations"), post=_extract_post,
    cites=_uncited("fields", "name", "source_quotes", lambda r: r.get("found") is True))


# ---------------------------------------------------------------------------------------------------------------
# checklist_match (decision)

CHECKLIST_STATUSES = ("present", "missing", "unclear")


def _checklist_items(config: Mapping, inputs: Mapping[str, Any], _instructions: str) -> list[DecisionItem]:
    context = _context(inputs)
    return [DecisionItem(id=i["id"], question=f"Do the inputs show this item? {i['label']}: {i['description']}",
                         options=CHECKLIST_STATUSES, context=context)
            for i in config["items"]]


def _checklist_assemble(config: Mapping, outcomes: list[DecisionOutcome]) -> dict:
    found = _by_id(outcomes)
    rows = []
    for item in config["items"]:
        o = found.get(item["id"])
        if o is None:
            continue
        note = "human" if o.decided_by == "human" or o.confidence is None else f"{o.decided_by} {o.confidence:.2f}"
        rows.append({"id": item["id"], "status": o.choice, "evidence_quote": _first(o.evidence), "note": note})
    return {"items": rows}


def _checklist_post(config: Mapping, output: dict, ctx: PostContext) -> tuple[dict, list[Check]]:
    out = copy.deepcopy(output)
    rows = _rows(out, "items")
    coverage = _coverage([i["id"] for i in config.get("items", [])], [r.get("id") for r in rows])
    return out, [coverage]


CHECKLIST_MATCH = Primitive(
    id="checklist_match", version=1, title="Checklist match",
    purpose="Mark each configured checklist item present, missing or unclear from the inputs.", executor="decision",
    config_schema=obj({"items": arr(obj({"id": ID, "label": S(300), "description": S(1000)}), 1, 150),
                       "decision_threshold": THRESHOLD}, required=["items"]),
    instructions="",
    output_schema_for=lambda _c: obj({"items": arr(obj({
        "id": S(48), "status": {"type": "string", "enum": list(CHECKLIST_STATUSES)},
        "evidence_quote": S(2000), "note": S(1000)}))}),
    default_guardrails=("schema", "citations"), post=_checklist_post,
    cites=_uncited("items", "id", "evidence_quote", lambda r: r.get("status") == "present"),
    decision_items=_checklist_items, assemble=_checklist_assemble, rubric=DECISION_RUBRIC)


# ---------------------------------------------------------------------------------------------------------------
# screen_score (decision)

def _score_items(config: Mapping, inputs: Mapping[str, Any], _instructions: str) -> list[DecisionItem]:
    context, top = _context(inputs), config["scale_max"]
    return [DecisionItem(
        id=c["id"], question=f"Score this criterion from 0 to {top}. {c['label']}: {c['description']}",
        options=tuple(str(i) for i in range(top + 1)), context=context,
        option_descriptions={"0": "no evidence", str(top): "fully meets"})
        for c in config["criteria"]]


def _score_assemble(config: Mapping, outcomes: list[DecisionOutcome]) -> dict:
    found = _by_id(outcomes)
    scores = [{"criterion_id": c["id"], "score": int(found[c["id"]].choice), "rationale": found[c["id"]].rationale,
               "evidence_quote": _first(found[c["id"]].evidence)} for c in config["criteria"] if c["id"] in found]
    return {"scores": scores, "overall_score": 0, "recommendation": ""}


def _score_post(config: Mapping, output: dict, ctx: PostContext) -> tuple[dict, list[Check]]:
    out = copy.deepcopy(output)
    criteria, top = config.get("criteria", []), config.get("scale_max", 0)
    rows = _rows(out, "scores")
    coverage = _coverage([c["id"] for c in criteria], [r.get("criterion_id") for r in rows])
    out_of_range = [str(r.get("criterion_id")) for r in rows
                    if not (isinstance(r.get("score"), int) and not isinstance(r.get("score"), bool)
                            and 0 <= r["score"] <= top)]
    by_id: dict[Any, int] = {}
    for r in rows:
        if isinstance(r.get("score"), int) and not isinstance(r.get("score"), bool):
            by_id.setdefault(r.get("criterion_id"), r["score"])
    weight_total = sum(c["weight"] for c in criteria)
    earned = sum(c["weight"] * by_id.get(c["id"], 0) for c in criteria)
    overall = round(100 * earned / (weight_total * top), 1) if weight_total and top else 0.0
    if isinstance(out, dict):
        out["overall_score"] = overall
        out["recommendation"] = f"Overall {overall}/100"
    return out, [coverage, Check("score_range", not out_of_range, ", ".join(out_of_range))]


SCREEN_SCORE = Primitive(
    id="screen_score", version=1, title="Screen and score",
    purpose="Score the inputs against weighted criteria and compute an overall score.", executor="decision",
    config_schema=obj({
        "criteria": arr(obj({"id": ID, "label": S(300), "description": S(1000),
                             "weight": {"type": "integer", "minimum": 1, "maximum": 5}}), 1, 30),
        "scale_max": {"type": "integer", "minimum": 2, "maximum": 10},
        "decision_threshold": THRESHOLD}, required=["criteria", "scale_max"]),
    instructions="",
    output_schema_for=lambda _c: obj({
        "scores": arr(obj({"criterion_id": S(48), "score": {"type": "integer"}, "rationale": S(2000),
                           "evidence_quote": S(2000)})),
        "overall_score": {"type": "number"}, "recommendation": S(1000)}),
    default_guardrails=("schema", "citations"), post=_score_post,
    cites=_uncited("scores", "criterion_id", "evidence_quote", lambda r: _is_number(r.get("score")) and r["score"] > 0),
    decision_items=_score_items, assemble=_score_assemble, rubric=DECISION_RUBRIC)


# ---------------------------------------------------------------------------------------------------------------
# classify_route (decision)

def _route_items(config: Mapping, inputs: Mapping[str, Any], instructions: str) -> list[DecisionItem]:
    categories = config["categories"]
    descriptions = {c["id"]: f"{c['label']}: {c['description']}" for c in categories}
    descriptions[NONE_OPTION] = "None of the categories fits"
    return [DecisionItem(
        id="route", question=f"Which category fits these inputs best? {instructions}".strip(),
        options=tuple(c["id"] for c in categories) + (NONE_OPTION,), context=_context(inputs),
        option_descriptions=descriptions)]


def _route_assemble(_config: Mapping, outcomes: list[DecisionOutcome]) -> dict:
    o = _by_id(outcomes).get("route")
    if o is None:  # nothing settled: an empty category fails the post check instead of raising
        return {"category_id": "", "confidence": 0.0, "decided_by": ""}
    return {"category_id": o.choice or "", "confidence": o.confidence or 0.0, "decided_by": o.decided_by or ""}


def _route_post(config: Mapping, output: dict, ctx: PostContext) -> tuple[dict, list[Check]]:
    out = copy.deepcopy(output)
    known = [c["id"] for c in config.get("categories", [])]
    category = out.get("category_id") if isinstance(out, Mapping) else None
    confidence = out.get("confidence") if isinstance(out, Mapping) else None
    problems = []
    if category not in known:
        problems.append(f"unknown category {str(category)[:48]}")
    if not (_is_number(confidence) and 0 <= confidence <= 1):
        problems.append("confidence outside 0..1")
    return out, [Check("category", not problems, "; ".join(problems))]


CLASSIFY_ROUTE = Primitive(
    id="classify_route", version=1, title="Classify and route",
    purpose="Pick the configured category that fits the inputs.", executor="decision",
    # At most 15: the decision item adds NONE_OPTION and allows at most 16 options.
    config_schema=obj({"categories": arr(obj({"id": ID, "label": S(300), "description": S(1000)}), 2, 15),
                       "decision_threshold": THRESHOLD}, required=["categories"]),
    instructions="",
    output_schema_for=lambda _c: obj({"category_id": S(48), "confidence": {"type": "number"}, "decided_by": S(16)}),
    post=_route_post, decision_items=_route_items, assemble=_route_assemble, rubric=DECISION_RUBRIC)


# ---------------------------------------------------------------------------------------------------------------
# draft_message (model) + rule checks the executor decides like any other decision

DRAFT_MESSAGE = Primitive(
    id="draft_message", version=1, title="Draft message",
    purpose="Write a draft message or document for human review.", executor="model",
    config_schema=obj({"channel": {"type": "string", "enum": ["email", "message", "document"]},
                       "audience": S(300), "external": B, "rules": arr(S(500), 0, 10)},
                      required=["channel", "audience", "external"]),
    instructions=("Write a draft for the configured channel and audience from the inputs. It is a draft for human "
                  "review and will not be sent. Do not include identifiers such as identity or reference numbers."),
    output_schema_for=lambda _c: obj({"subject": S(300), "body": S(20000), "notes": S(2000)}),
    default_guardrails=("schema", "required_fields", "mask_sensitive"), required_paths=("subject", "body"),
    text_paths=("subject", "body"), external_key="external")


def rule_items(config: Mapping, output: Mapping) -> list[DecisionItem]:
    """One yes/no decision per configured draft rule; ids rule_1..rule_n. Context is the (masked) subject + body."""
    context = f"{output.get('subject', '')}\n\n{output.get('body', '')}"
    return [DecisionItem(id=f"rule_{n}", question=f"Does this draft comply with the rule: {rule}?",
                         options=("yes", "no"), context=context, kind="yes_no")
            for n, rule in enumerate(config.get("rules", []), 1)]


def rule_check(config: Mapping, outcomes: Iterable[DecisionOutcome]) -> Check:
    """rule_compliance passes only when every rule was settled as "yes"."""
    found = _by_id(outcomes)
    violated, unsettled = [], []
    for n, rule in enumerate(config.get("rules", []), 1):
        o = found.get(f"rule_{n}")
        if o is None or o.needs_review or o.choice not in ("yes", "no"):
            unsettled.append(rule)
        elif o.choice == "no":
            violated.append(rule)
    parts = [f"{label}: {'; '.join(rules)}" for label, rules in (("violated", violated), ("unsettled", unsettled))
             if rules]
    return Check("rule_compliance", not parts, " | ".join(parts))


# ---------------------------------------------------------------------------------------------------------------
# generate_questions

def _questions_post(config: Mapping, output: dict, ctx: PostContext) -> tuple[dict, list[Check]]:
    out = copy.deepcopy(output)
    count, limit = len(_rows(out, "questions")), config.get("max_questions", 30)
    return out, [Check("question_limit", count <= limit, f"{count} of {limit}")]


GENERATE_QUESTIONS = Primitive(
    id="generate_questions", version=1, title="Generate questions",
    purpose="Ask specific questions that close the gaps in the inputs.", executor="model",
    config_schema=obj({"max_questions": {"type": "integer", "minimum": 1, "maximum": 30}, "focus": S(1000)}),
    instructions=("Write up to max_questions specific questions that close the gaps visible in the inputs, each "
                  "with the reason it matters."),
    output_schema_for=lambda _c: obj({"questions": arr(obj({"question": S(1000), "reason": S(1000)}), 1, 30)}),
    post=_questions_post)


# ---------------------------------------------------------------------------------------------------------------
# summarize_report

def _summary_post(config: Mapping, output: dict, ctx: PostContext) -> tuple[dict, list[Check]]:
    out = copy.deepcopy(output)
    allowed = set(ctx.upstream_step_ids or ctx.input_step_ids)
    removed: set[str] = set()
    for section in _rows(out, "sections"):
        kept: list[str] = []
        for cited in section.get("source_steps") or []:
            name = str(cited).strip().removeprefix("step:").strip()
            if name in allowed:
                if name not in kept:
                    kept.append(name)
            else:
                removed.add(name or str(cited))
        section["source_steps"] = kept
    assessed = {_norm(c.get("criterion", "")) for c in _rows(out, "criteria")}
    unassessed = [c for c in ctx.completion_criteria if _norm(c) not in assessed]
    # Citations to steps that did not feed this report are removed, so the report never shows invented provenance.
    detail = ("removed citations to steps that did not feed this report: " + ", ".join(sorted(removed))) if removed else ""
    return out, [Check("source_steps", True, detail),
                 Check("criteria_coverage", not unassessed, "; ".join(unassessed))]


SUMMARIZE_REPORT = Primitive(
    id="summarize_report", version=1, title="Summarize report",
    purpose="Compile upstream step outputs into a report that assesses the completion criteria.", executor="model",
    config_schema=obj({"sections": arr(S(200), 0, 12)}),
    instructions=("Compile a report from the upstream step outputs. Cite contributing step ids in source_steps, as "
                  "bare ids without the 'step:' prefix (for example extract, not step:extract). Assess every "
                  "completion criterion with met and evidence."),
    output_schema_for=lambda _c: obj({
        "title": S(300), "summary": S(4000),
        "sections": arr(obj({"title": S(300), "body": S(20000), "source_steps": arr(S(48), 0, 16)})),
        "criteria": arr(obj({"criterion": S(500), "met": B, "evidence": S(2000)}))}),
    default_guardrails=("schema", "required_fields"), required_paths=("title", "summary"), post=_summary_post)


# ---------------------------------------------------------------------------------------------------------------
# human_approval, deliver_package, generic_model_step

HUMAN_APPROVAL = Primitive(
    id="human_approval", version=1, title="Human approval",
    purpose="Pause until the assigned approver approves, edits or rejects the upstream output.", executor="human",
    config_schema=obj({}), instructions="", output_schema_for=lambda _c: {"type": "object"}, default_guardrails=())

DELIVER_PACKAGE = Primitive(
    id="deliver_package", version=1, title="Deliver package",
    purpose="Bundle the upstream outputs into the final deliverable.", executor="deterministic",
    config_schema=obj({"title": S(300)}), instructions="",
    output_schema_for=lambda _c: {
        "type": "object", "required": ["title", "items"],
        "properties": {"title": S(300), "items": {"type": "array", "items": {
            "type": "object", "required": ["step_id", "role", "content"]}}}})


def _output_field_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(f["name"] for f in config.get("output_fields", []))


GENERIC_MODEL_STEP = Primitive(
    id="generic_model_step", version=1, title="Generic model step",
    purpose="Perform free-form step instructions and fill the configured output fields.", executor="model",
    config_schema=obj({"output_fields": arr(obj({"name": ID, "description": S(1000)}), 1, 20)}),
    instructions="Perform the step instructions on the inputs and fill every output field.",
    output_schema_for=lambda c: obj({name: S(20000) for name in _output_field_names(c)}),
    default_guardrails=("schema", "required_fields"), required_paths_from=_output_field_names)


# ---------------------------------------------------------------------------------------------------------------
# split_items (model, or deterministic with a separator): one input becomes items that later steps run once each

def finish_split(config: Mapping, labels: list[str], texts: list[str],
                 problems: list[str]) -> tuple[dict, list[str], list[Check]]:
    """({"items": [{key, label, text}]}, duplicate labels, checks) from item texts cut out of the input. Nothing is
    truncated: more items than max_items fails item_count. Identical items are collapsed and recorded."""
    labels = [(label.strip() if isinstance(label, str) else "") or f"{config.get('item_noun', 'item')} {n}"
              for n, label in enumerate(labels, 1)]
    items, duplicates = build_items(labels, texts) if not problems else ([], [])
    limit = config.get("max_items", DEFAULT_MAX_ITEMS)
    count = len(items) if not problems else len(labels)
    return {"items": items}, duplicates, [
        Check("item_boundaries", not problems, "; ".join(problems)),
        Check("item_count", 1 <= count <= limit, f"{count} of at most {limit}"),
        Check("duplicate_items", True, ("collapsed: " + ", ".join(duplicates)) if duplicates else "")]


def _split_post(config: Mapping, output: dict, ctx: PostContext) -> tuple[dict, list[Check]]:
    # The executor passes exactly the split input as the only source.
    packet = ctx.sources[0] if ctx.sources else ""
    rows = _rows(output, "items")
    texts, problems = slice_by_boundaries(packet, [r.get("start_quote", "") for r in rows])
    out, _duplicates, checks = finish_split(config, [r.get("label", "") for r in rows], texts, problems)
    return out, checks


SPLIT_ITEMS = Primitive(
    id="split_items", version=1, title="Split into items",
    purpose="Split one input into separate items so the following steps run once per item.", executor="model",
    config_schema=obj({"item_noun": {"type": "string", "minLength": 1, "maxLength": 40},
                       "max_items": {"type": "integer", "minimum": 1, "maximum": MAX_ITEMS},
                       "separator": {"type": "string", "minLength": 1, "maxLength": 40,
                                     "description": ("Optional. Exact literal text that appears between items in "
                                                     "the input, such as a line of ---; not a description of the "
                                                     "items. Leave it out when the input has no such marker.")}},
                      required=["item_noun"]),
    instructions=("The input holds several separate items. For each item, in order, return a short label and "
                  "start_quote: the exact text, copied verbatim, at which that item begins. Choose a start_quote "
                  "of at least a few words that appears only once. Do not rewrite or summarise items."),
    # One row more than MAX_ITEMS, so a packet with too many items fails item_count instead of being cut short.
    output_schema_for=lambda _c: obj({"items": arr(obj({"label": S(200), "start_quote": S(300)}), 0,
                                                   MAX_ITEMS + 1)}),
    post=_split_post)


BUILTIN_PRIMITIVES: tuple[Primitive, ...] = (
    COLLECT_INPUTS, EXTRACT_FIELDS, CHECKLIST_MATCH, SCREEN_SCORE, CLASSIFY_ROUTE, DRAFT_MESSAGE,
    GENERATE_QUESTIONS, SUMMARIZE_REPORT, HUMAN_APPROVAL, DELIVER_PACKAGE, GENERIC_MODEL_STEP, SPLIT_ITEMS,
)


def builtin_catalog() -> PrimitiveCatalog:
    return PrimitiveCatalog(BUILTIN_PRIMITIVES)


def catalog_with(records: Iterable[Mapping] = ()) -> PrimitiveCatalog:
    """The built-in catalog. This engine keeps no library of an organization's own steps, so records are ignored."""
    return PrimitiveCatalog(BUILTIN_PRIMITIVES)
