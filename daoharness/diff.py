"""Deterministic structural diff between two workflow spec versions."""
from __future__ import annotations

from .spec import WorkflowSpec

META_FIELDS = ("domain", "purpose", "input_requirements", "deliverables", "completion_criteria")
IGNORED_STEP_KEYS = frozenset({"origin"})


def spec_diff(old: WorkflowSpec, new: WorkflowSpec) -> dict:
    before, after = old.to_dict(), new.to_dict()
    old_steps = {s["id"]: s for s in before["steps"]}
    new_steps = {s["id"]: s for s in after["steps"]}
    common_old = [sid for sid in old_steps if sid in new_steps]
    common_new = [sid for sid in new_steps if sid in old_steps]
    changed = []
    for sid in common_new:
        a, b = old_steps[sid], new_steps[sid]
        fields = sorted(k for k in set(a) | set(b) if k not in IGNORED_STEP_KEYS and a.get(k) != b.get(k))
        if fields:
            changed.append({"step_id": sid, "fields": fields})
    return {
        "added": [sid for sid in new_steps if sid not in old_steps],
        "removed": [sid for sid in old_steps if sid not in new_steps],
        "changed": changed,
        "reordered": common_old != common_new,
        "meta_changed": [k for k in META_FIELDS if before[k] != after[k]],
    }
