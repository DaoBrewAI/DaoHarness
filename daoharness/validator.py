"""Semantic checks on a structurally valid WorkflowSpec, plus a few counts about it.

Errors are strings "code:step_id:detail"; the code is the prefix before the first ":". A source the engine cannot
reach is a gap, not an error: the workflow can still be saved and a run stops at that step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator

from .decision import NONE_OPTION
from .items import DEFAULT_MAX_ITEMS
from .primitives import Primitive, PrimitiveCatalog
from .spec import REF, WorkflowSpec, WorkflowStep

NEEDS_INPUTS = frozenset({"collect_inputs", "deliver_package", "human_approval"})
# Config lists whose entries are addressed by id at run time: (list key, id key).
ID_LISTS = (("fields", "name"), ("items", "id"), ("criteria", "id"), ("categories", "id"), ("output_fields", "name"))
# "route" is the decision item id of classify_route and NONE_OPTION is its escape option.
RESERVED_CATEGORY_IDS = frozenset({NONE_OPTION, "route"})
MAX_SEGMENT_STEPS = 8


@dataclass(frozen=True)
class ValidationContext:
    catalog: PrimitiveCatalog
    approver_resolves: Callable[[str], bool]
    connected_sources: frozenset = frozenset()


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    gaps: tuple[dict, ...]
    metrics: dict

    @property
    def ok(self) -> bool:
        return not self.errors


class ValidationFailed(Exception):
    def __init__(self, errors) -> None:
        super().__init__("; ".join(errors))
        self.errors = tuple(errors)


def _is_human(step: WorkflowStep, primitive: Primitive | None) -> bool:
    if primitive is not None:
        return primitive.executor == "human"
    return step.primitive == "human_approval"


def _config_errors(primitive: Primitive, config: Mapping[str, Any]) -> list[str]:
    problems = []
    for error in sorted(Draft202012Validator(primitive.config_schema).iter_errors(config), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in error.path) or "config"
        problems.append(f"{where} {error.message}"[:300])
    for key, id_key in ID_LISTS:
        rows = config.get(key)
        if not isinstance(rows, list):
            continue
        ids = [r.get(id_key) for r in rows if isinstance(r, Mapping) and isinstance(r.get(id_key), str)]
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        if duplicated:
            problems.append(f"{key} duplicate {id_key}: {', '.join(duplicated)}")
        if key == "categories":
            reserved = sorted(RESERVED_CATEGORY_IDS.intersection(ids))
            if reserved:
                problems.append(f"categories reserved id: {', '.join(reserved)}")
    return problems


def is_split(step: WorkflowStep, catalog: PrimitiveCatalog) -> bool:
    primitive = catalog.get(step.primitive)
    return primitive is not None and primitive.id == "split_items"


def _fan_out_errors(spec: WorkflowSpec, catalog: PrimitiveCatalog) -> list[str]:
    """One split step, followed directly by one contiguous segment of per-item steps (for_each = the split id).
    Per-item steps are automatic and cannot be the deliverable; approvals come after the segment."""
    splits = [i for i, s in enumerate(spec.steps) if is_split(s, catalog)]
    per_item = [i for i, s in enumerate(spec.steps) if s.for_each is not None]
    if not splits and not per_item:
        return []
    errors = []
    index = {s.id: i for i, s in enumerate(spec.steps)}
    for i in per_item:
        step = spec.steps[i]
        target = index.get(step.for_each)
        if target is None or target >= i or not is_split(spec.steps[target], catalog):
            errors.append(f"invalid_for_each:{step.id}:{step.for_each} is not an earlier split_items step")
        primitive = catalog.get(step.primitive)
        if (_is_human(step, primitive) or is_split(step, catalog)
                or (primitive is not None and primitive.id == "deliver_package")):
            errors.append(f"not_per_item:{step.id}:{step.primitive} cannot run once per item")
    for extra in splits[1:]:
        errors.append(f"multiple_splits:{spec.steps[extra].id}:only one split_items step per workflow")
    for i in splits:
        split = spec.steps[i]
        input_refs = [r for r in split.inputs if r.startswith("input:")]
        if len(input_refs) != 1:
            errors.append(f"split_input:{split.id}:a split reads exactly one input:<name> to split")
        segment = [j for j in per_item if spec.steps[j].for_each == split.id]
        if not segment:
            errors.append(f"split_without_items:{split.id}:no step has for_each {split.id}")
            continue
        if segment != list(range(i + 1, i + 1 + len(segment))):
            first = next((j for j in range(i + 1, i + 1 + len(segment)) if j not in segment), segment[-1])
            errors.append(f"segment_not_contiguous:{spec.steps[first].id}:per-item steps must directly follow "
                          f"{split.id}, with no other step between them")
        if len(segment) > MAX_SEGMENT_STEPS:
            errors.append(f"segment_too_long:{split.id}:{len(segment)} per-item steps, at most {MAX_SEGMENT_STEPS}")
        # A per-item step reads its own item (the split, or an earlier per-item step), never the whole split input:
        # that would show it every item, and its citations would pass against all of them.
        own = {f"step:{split.id}"}
        for j in segment:
            step = spec.steps[j]
            if not own & set(step.inputs):
                errors.append(f"per_item_input:{step.id}:read step:{split.id} or an earlier per-item step")
            if set(input_refs) & set(step.inputs):
                errors.append(f"per_item_input:{step.id}:{input_refs[0]} holds every item; read step:{split.id}")
            own.add(f"step:{step.id}")
    return errors


def fan_out(spec: WorkflowSpec, catalog: PrimitiveCatalog) -> dict | None:
    split = next((s for s in spec.steps if is_split(s, catalog)), None)
    if split is None:
        return None
    return {"split_step": split.id, "per_item_steps": [s.id for s in spec.steps if s.for_each == split.id],
            "max_items": split.config.get("max_items", DEFAULT_MAX_ITEMS)}


def validate_spec(spec: WorkflowSpec, ctx: ValidationContext) -> ValidationReport:
    errors: list[str] = []
    gaps: list[dict] = []
    input_names = {r.name for r in spec.input_requirements}
    earlier: set[str] = set()
    for index, step in enumerate(spec.steps):
        for ref in step.inputs:
            match = REF.fullmatch(ref)
            kind, name = (match.group(1), match.group(2)) if match else ("", ref)
            if kind == "input" and name not in input_names:
                errors.append(f"unknown_input:{step.id}:{name}")
            elif kind == "step" and name not in earlier:
                errors.append(f"forward_or_unknown_step:{step.id}:{name}")
            elif not match:
                errors.append(f"unknown_input:{step.id}:{ref}")

        primitive = ctx.catalog.get(step.primitive)
        if primitive is None:
            errors.append(f"unknown_primitive:{step.id}:{step.primitive}")
        else:
            errors.extend(f"invalid_config:{step.id}:{p}" for p in _config_errors(primitive, step.config))

        human = _is_human(step, primitive)
        if human != (step.approval is not None):
            errors.append(f"approval_mismatch:{step.id}:{'approval required' if human else 'approval not allowed'}")
        if human and step.approval is not None and not ctx.approver_resolves(step.approval.approver_role):
            errors.append(f"approver_not_member:{step.id}:{step.approval.approver_role}")
        if human and not step.inputs:
            errors.append(f"approval_without_content:{step.id}:no inputs")

        if primitive is not None and primitive.external_draft(step.config):
            ref = f"step:{step.id}"
            if not any(ref in later.inputs and _is_human(later, ctx.catalog.get(later.primitive))
                       for later in spec.steps[index + 1:]):
                errors.append(f"external_draft_requires_approval:{step.id}:{step.primitive}")

        for source in dict.fromkeys(step.required_sources):
            if source not in ctx.connected_sources:
                gaps.append({"step_id": step.id, "kind": "needs_capability", "detail": source})

        if step.primitive in NEEDS_INPUTS and not step.inputs:
            errors.append(f"missing_inputs:{step.id}:{step.primitive}")
        earlier.add(step.id)

    errors.extend(_fan_out_errors(spec, ctx.catalog))
    last = spec.steps[-1] if spec.steps else None
    if last is None or last.primitive != "deliver_package":
        errors.append(f"missing_deliverable:{last.id if last else ''}:last step must be deliver_package")
    return ValidationReport(tuple(errors), tuple(gaps), workflow_metrics(spec, ctx.catalog))


def workflow_metrics(spec: WorkflowSpec, catalog: PrimitiveCatalog) -> dict:
    n = len(spec.steps)
    automatic = sum(1 for s in spec.steps if not _is_human(s, catalog.get(s.primitive)))
    metrics = {"total_steps": n, "automatic_steps": automatic,
               "automation_ratio": round(automatic / n, 2) if n else 0.0}
    segment = fan_out(spec, catalog)
    if segment is not None:  # single-case metrics keep their exact shape
        metrics["fan_out"] = segment
    return metrics
