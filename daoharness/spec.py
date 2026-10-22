"""Versioned, industry-neutral workflow specification (structure only; semantics live in validator.py)."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

SPEC_SCHEMA = "daoharness-workflow-spec/v1"
NAME = re.compile(r"[a-z][a-z0-9_]{0,47}")  # always used with fullmatch
REF = re.compile(r"(input|step):([a-z][a-z0-9_]{0,47})")
PRIMITIVE_REF = re.compile(r"[a-z][a-z0-9_]{0,63}")  # always used with fullmatch
EXTRA_GUARDRAILS = frozenset({"citations", "required_fields", "mask_sensitive"})
MAX_STEPS = 24


class WorkflowSpecError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _fail(path: str) -> WorkflowSpecError:
    return WorkflowSpecError("invalid_spec", path)


def _text(data: Mapping, key: str, path: str, limit: int, *, optional: bool = False) -> str:
    value = data.get(key, "" if optional else None)
    if not isinstance(value, str) or len(value) > limit or (not optional and not value.strip()):
        raise _fail(f"{path}.{key}")
    return value.strip()


def _texts(data: Mapping, key: str, path: str, *, limit: int, item_limit: int = 500) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list) or len(value) > limit:
        raise _fail(f"{path}.{key}")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > item_limit:
            raise _fail(f"{path}.{key}")
        out.append(item.strip())
    return tuple(out)


def _json_object(value: Any, path: str) -> dict:
    if not isinstance(value, Mapping):
        raise _fail(path)
    try:
        canonical_json(value)
        return copy.deepcopy(dict(value))
    except (TypeError, ValueError, RecursionError):
        raise _fail(path) from None


@dataclass(frozen=True)
class InputRequirement:
    name: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class Approval:
    approver_role: str
    what_is_approved: str


@dataclass(frozen=True)
class StepOrigin:
    primitive_id: str
    primitive_version: int
    edits: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    role: str
    instructions: str
    primitive: str
    primitive_version: int = 1
    config: Mapping[str, Any] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()
    guardrails: tuple[str, ...] = ()
    required_sources: tuple[str, ...] = ()
    approval: Approval | None = None
    origin: StepOrigin | None = None
    for_each: str | None = None  # id of the split step this step runs once per item for

    def to_dict(self) -> dict:
        out = {
            "id": self.id, "role": self.role, "instructions": self.instructions,
            "primitive": self.primitive, "primitive_version": self.primitive_version,
            "config": copy.deepcopy(dict(self.config)), "inputs": list(self.inputs),
            "guardrails": list(self.guardrails), "required_sources": list(self.required_sources),
            "approval": None if self.approval is None else {
                "approver_role": self.approval.approver_role, "what_is_approved": self.approval.what_is_approved},
            "origin": None if self.origin is None else {
                "primitive_id": self.origin.primitive_id, "primitive_version": self.origin.primitive_version,
                "edits": list(self.origin.edits)},
        }
        if self.for_each is not None:  # absent when unset, so specs without it keep their digests
            out["for_each"] = self.for_each
        return out


@dataclass(frozen=True)
class WorkflowSpec:
    domain: str
    purpose: str
    input_requirements: tuple[InputRequirement, ...]
    deliverables: tuple[str, ...]
    completion_criteria: tuple[str, ...]
    steps: tuple[WorkflowStep, ...]

    def to_dict(self) -> dict:
        return {
            "domain": self.domain, "purpose": self.purpose,
            "input_requirements": [{"name": i.name, "description": i.description, "required": i.required}
                                   for i in self.input_requirements],
            "deliverables": list(self.deliverables), "completion_criteria": list(self.completion_criteria),
            "steps": [s.to_dict() for s in self.steps],
        }

    def digest(self) -> str:
        return digest(self.to_dict())

    def step(self, step_id: str) -> WorkflowStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise WorkflowSpecError("unknown_step", step_id)

    @classmethod
    def from_dict(cls, data: Any) -> "WorkflowSpec":
        if not isinstance(data, Mapping):
            raise _fail("spec")
        inputs = data.get("input_requirements", [])
        if not isinstance(inputs, list) or len(inputs) > 16:
            raise _fail("input_requirements")
        reqs, names = [], set()
        for i, item in enumerate(inputs):
            path = f"input_requirements[{i}]"
            if not isinstance(item, Mapping):
                raise _fail(path)
            name = _text(item, "name", path, 48)
            if not NAME.fullmatch(name):
                raise _fail(f"{path}.name")
            if name in names:
                raise _fail(f"duplicate_input:{name}")
            names.add(name)
            required = item.get("required", True)
            if not isinstance(required, bool):
                raise _fail(f"{path}.required")
            reqs.append(InputRequirement(name, _text(item, "description", path, 1000), required))
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_STEPS:
            raise _fail("steps")
        steps, ids = [], set()
        for i, raw in enumerate(raw_steps):
            path = f"steps[{i}]"
            if not isinstance(raw, Mapping):
                raise _fail(path)
            sid = raw.get("id")
            if not isinstance(sid, str) or not NAME.fullmatch(sid):
                raise _fail(f"{path}.id")
            if sid in ids:
                raise _fail(f"duplicate_step_id:{sid}")
            ids.add(sid)
            primitive = raw.get("primitive")
            if not isinstance(primitive, str) or not PRIMITIVE_REF.fullmatch(primitive):
                raise _fail(f"{path}.primitive")
            version = raw.get("primitive_version", 1)
            if not isinstance(version, int) or isinstance(version, bool) or not 1 <= version <= 999999:
                raise _fail(f"{path}.primitive_version")
            refs = raw.get("inputs", [])
            if not isinstance(refs, list) or len(refs) > 16 or not all(isinstance(r, str) and REF.fullmatch(r) for r in refs):
                raise _fail(f"{path}.inputs")
            guards = raw.get("guardrails", [])
            if not isinstance(guards, list) or not all(g in EXTRA_GUARDRAILS for g in guards):
                raise _fail(f"{path}.guardrails")
            sources = _texts(raw, "required_sources", path, limit=8, item_limit=64)
            approval = raw.get("approval")
            if approval is not None:
                if not isinstance(approval, Mapping):
                    raise _fail(f"{path}.approval")
                approval = Approval(_text(approval, "approver_role", f"{path}.approval", 200),
                                    _text(approval, "what_is_approved", f"{path}.approval", 1000))
            origin = raw.get("origin")
            if origin is not None:
                if not isinstance(origin, Mapping):
                    raise _fail(f"{path}.origin")
                origin_version = origin.get("primitive_version", 1)
                if (not isinstance(origin_version, int) or isinstance(origin_version, bool)
                        or not 1 <= origin_version <= 999999):
                    raise _fail(f"{path}.origin.primitive_version")
                origin = StepOrigin(_text(origin, "primitive_id", f"{path}.origin", 128),
                                    origin_version,
                                    _texts(origin, "edits", f"{path}.origin", limit=32))
            for_each = raw.get("for_each")
            if for_each is not None and not (isinstance(for_each, str) and NAME.fullmatch(for_each)):
                raise _fail(f"{path}.for_each")
            steps.append(WorkflowStep(
                id=sid, role=_text(raw, "role", path, 200), instructions=_text(raw, "instructions", path, 4000),
                primitive=primitive, primitive_version=version,
                config=_json_object(raw.get("config", {}), f"{path}.config"),
                inputs=tuple(refs), guardrails=tuple(dict.fromkeys(guards)), required_sources=sources,
                approval=approval, origin=origin, for_each=for_each))
        return cls(
            domain=_text(data, "domain", "spec", 200), purpose=_text(data, "purpose", "spec", 2000),
            input_requirements=tuple(reqs),
            deliverables=_texts(data, "deliverables", "spec", limit=12),
            completion_criteria=_texts(data, "completion_criteria", "spec", limit=16),
            steps=tuple(steps))
