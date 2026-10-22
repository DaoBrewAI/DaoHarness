"""The application layer: build a workflow from a description, save it, run it, and decide what a run waits for.

One person on one machine. A workflow record holds its draft and, once saved, its version; every save is its own
record. Every action carries an action id and is applied once: the same request again (a dropped connection, a
command run twice) waits for the first and returns its result, so nothing is built, saved or run twice.
"""
from __future__ import annotations

import copy
import functools
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .compiler import CompileError, WorkflowCompiler
from .decision import DEFAULT_THRESHOLD, DecisionPort, DecisionRouter, ModelDecisions
from .executor import RUNS_DESPITE_GAPS, ExecutorError, WorkflowExecutor
from .file_store import WorkflowStoreError
from .messages import MODEL_ERRORS, model_error_message, validation_message, validation_problems
from .ports import MembershipPort, ModelError, ModelPort, Notifier, WorkflowStore
from .primitives import builtin_catalog
from .spec import WorkflowSpec, WorkflowSpecError, digest
from .validator import ValidationContext, validate_spec

logger = logging.getLogger(__name__)

DYNAMIC_SCHEMA = "daoharness-dynamic-capability/v1"
ROUTE_BASE = "/web/business-capabilities"
MAX_NEED = 16000
MAX_NOTE = 4000
MAX_INPUT_BYTES = 64 * 1024
ACTION_ID = re.compile(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*")   # fullmatch, at most 128 characters
HEX_ID = re.compile(r"[0-9a-f]{32}")                        # workflow, draft and approval ids
SETTLED_STATES = frozenset({"completed", "rejected", "failed", "needs_attention", "needs_access"})


class WorkflowServiceError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _bad_input(message: str) -> WorkflowServiceError:
    return WorkflowServiceError(400, "invalid_dynamic_input", message)


def _convert(exc: BaseException) -> WorkflowServiceError | None:
    if isinstance(exc, WorkflowServiceError):
        return exc
    if isinstance(exc, ExecutorError):
        return WorkflowServiceError(exc.status, exc.code, exc.message)
    if isinstance(exc, CompileError):
        return WorkflowServiceError(422, "compile_invalid", validation_message([*exc.hints, *exc.errors]))
    if isinstance(exc, WorkflowSpecError):
        return WorkflowServiceError(422, "compile_invalid", f"The workflow is not valid: {exc}"[:500])
    if isinstance(exc, ModelError):   # the code says which; the message says what to do about it
        return WorkflowServiceError(503, exc.code, model_error_message(exc.code, exc.detail))
    if isinstance(exc, WorkflowStoreError):
        return _bad_input(f"Invalid identifier ({exc.code}).")
    return None


def _public(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as exc:
            err = _convert(exc)
            if err is None or err is exc:
                raise
            raise err from exc
    return wrapper


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _item_view(item: Mapping) -> dict:
    """An item record as the API returns it: the pending decisions and the review copy stay internal."""
    return {k: copy.deepcopy(v) for k, v in item.items() if k not in ("pending_outcomes", "review")}


def _step_error(error: Any) -> Any:
    """A step's error as the API shows it: one from the model gets the sentence for its code."""
    error = copy.deepcopy(error)
    if isinstance(error, dict) and not error.get("message") and error.get("code") in MODEL_ERRORS:
        error["message"] = model_error_message(error["code"])
    return error


def _step_view(record: Mapping) -> dict:
    view = {"step_id": record["step_id"], "role": record.get("role", ""), "primitive": record.get("primitive", ""),
            "state": record.get("state"), "attempt": record.get("attempt", 0),
            "output": copy.deepcopy(record.get("output")), "checks": copy.deepcopy(list(record.get("checks") or [])),
            "approval_id": record.get("approval_id"), "error": _step_error(record.get("error")),
            "decisions": copy.deepcopy(list(record.get("decisions") or [])),
            "flags": copy.deepcopy(list(record.get("flags") or [])),
            "started_at": record.get("started_at"), "finished_at": record.get("finished_at")}
    if "items" in record:
        view["items"] = [_item_view(i) for i in record.get("items") or []]
    return view


RECEIPT_KEYS = ("run_id", "model", "provider", "output_digest")


def run_usage(run: Mapping) -> dict:
    """Model calls a run made and the tokens they used: estimated when a provider reported none for some call."""
    def count(key: str) -> int:
        value = run.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    return {"calls": count("model_calls"), "input_tokens": count("input_tokens"),
            "output_tokens": count("output_tokens"), "estimated": count("estimated_calls") > 0}


def _execution_receipt(steps: Iterable[Mapping]) -> dict | None:
    """The model receipt of the last model-produced step that has one, or None."""
    found = None
    for record in steps:
        receipt = record.get("receipt")
        if isinstance(receipt, Mapping) and all(isinstance(receipt.get(k), str) for k in RECEIPT_KEYS):
            found = {k: receipt[k] for k in RECEIPT_KEYS}
    return found


def _checks_text(checks: Iterable[Mapping]) -> str:
    return ", ".join(f"{c.get('id')} {'passed' if c.get('passed') else 'failed'}" for c in checks)


def capability_result(run: Mapping) -> dict:
    """The result of a completed run: its last summarize_report output plus a table of the steps."""
    spec = run["spec"]
    reports = [s for s in run["steps"] if s.get("primitive") == "summarize_report" and isinstance(s.get("output"), Mapping)]
    report = reports[-1]["output"] if reports else {}
    sections = [{"title": str(x.get("title", "")), "body": str(x.get("body", ""))}
                for x in report.get("sections") or [] if isinstance(x, Mapping)]
    criteria = [{"criterion": str(c.get("criterion", "")), "met": c.get("met") is True,
                 "evidence": str(c.get("evidence", ""))} for c in report.get("criteria") or [] if isinstance(c, Mapping)]
    limitations = []
    if any(c.get("id") == "draft_only" for s in run["steps"] for c in s.get("checks") or []):
        limitations.append("Drafts were not sent.")
    limitations.extend(f"Not met: {c['criterion']}" for c in criteria if not c["met"])
    limitations.extend(f"Excluded: {i.get('label', '')} ({i.get('reason') or 'excluded'})"
                       for i in run.get("items") or [] if i.get("status") == "excluded")
    limitations.extend(f"Outside this workflow: {gap.get('detail')}" for step in spec["steps"]
                       if step.get("primitive") in RUNS_DESPITE_GAPS
                       for gap in (run.get("blocked_steps") or {}).get(step.get("id")) or [])
    tables = [{"title": "Step record", "columns": ["Step", "Role", "State", "Checks"],
               "rows": [[str(s["step_id"]), str(s.get("role", "")), str(s.get("state", "")),
                         _checks_text(s.get("checks") or [])] for s in run["steps"]]}]
    return {"title": str(report.get("title") or spec["domain"]), "summary": str(report.get("summary") or spec["purpose"]),
            "sections": sections, "tables": tables, "limitations": limitations, "criteria": criteria}


def result_markdown(run: Mapping) -> str:
    """A completed run's result as Markdown: what the delivered folder holds as result.md."""
    result = capability_result(run)
    lines = [f"# {result['title']}", "", result["summary"], ""]
    for section in result["sections"]:
        lines += [f"## {section['title']}", "", section["body"], ""]
    if result["criteria"]:
        lines += ["## Done checks", ""]
        lines += [f"- [{'x' if c['met'] else ' '}] {c['criterion']}" + (f": {c['evidence']}" if c["evidence"] else "")
                  for c in result["criteria"]]
        lines.append("")
    if result["limitations"]:
        lines += ["## Limitations", ""] + [f"- {item}" for item in result["limitations"]] + [""]
    return "\n".join(lines)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40].strip("-")
    return slug or "workflow"


def _write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _next_action(state: str) -> tuple[bool, str | None]:
    if state == "interrupted":
        return False, "resume"
    if state in ("running", "completed", "rejected"):
        return False, None
    return True, "retry"


def _approval_view(approval: Mapping) -> dict:
    keys = ("approval_id", "kind", "workflow_domain", "content", "content_digest", "approver_role",
            "what_is_approved", "step_id", "run_id", "instance_id", "version", "status", "created_at")
    view = {key: copy.deepcopy(approval.get(key)) for key in keys}
    view["kind"] = view["kind"] or "approval"
    if view["kind"] == "approval":
        view["flags"] = copy.deepcopy(list(approval.get("flags") or []))
        if approval.get("itemized"):
            view.update(itemized=True, item_keys=list(approval.get("item_keys") or []),
                        item_refs=list(approval.get("item_refs") or []))
    if view["kind"] == "decision_review":
        view["items"] = copy.deepcopy(list(approval.get("items") or []))
    return view


def _check_content(spec: WorkflowSpec, content: Any) -> dict:
    if not isinstance(content, Mapping):
        raise _bad_input("content must map input names to text.")
    known = {r.name for r in spec.input_requirements}
    unknown = sorted(str(k) for k in content if k not in known)
    if unknown:
        raise _bad_input(f"Unknown inputs: {', '.join(unknown)[:300]}.")
    for name, value in content.items():
        if not isinstance(value, str):
            raise _bad_input(f"{name} must be text.")
        if len(value.encode("utf-8")) > MAX_INPUT_BYTES:
            raise _bad_input(f"{name} is longer than 64 KiB.")
    empty = [r.name for r in spec.input_requirements
             if r.required and not (isinstance(content.get(r.name), str) and content[r.name].strip())]
    if empty:
        raise _bad_input(f"Required inputs are empty: {', '.join(empty)}.")
    return dict(content)


class WorkflowService:
    def __init__(self, *, store: WorkflowStore, model: ModelPort, membership: MembershipPort, notifier: Notifier,
                 clock: Callable[[], str], submit: Callable[[Callable[[], None]], Any],
                 decisions: DecisionPort | None = None, decision_threshold: float = DEFAULT_THRESHOLD,
                 model_call_cap: int = 40, new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
                 deliveries: Path | None = None) -> None:
        """decisions: a decider to use instead of the model for decision steps. submit(job) runs a job in the
        background (a thread); runs advance there. deliveries: a folder where each completed run leaves its
        result.md and package.json."""
        self.store = store
        self.model = model
        self.membership = membership
        self.notifier = notifier
        self.clock = clock
        self.submit = submit
        self.new_id = new_id
        self.deliveries = Path(deliveries) if deliveries is not None else None
        self.primitives = builtin_catalog()
        self.compiler = WorkflowCompiler(model)
        self.executor = WorkflowExecutor(
            store=store, model=model, decisions=DecisionRouter(model=ModelDecisions(model), provider=decisions),
            catalog_for=lambda org: self.primitives, membership=membership, notifier=notifier, clock=clock,
            model_call_cap=model_call_cap, decision_threshold=decision_threshold)
        self._guard = threading.Lock()
        self._action_locks: dict[str, threading.Lock] = {}

    # -- records ------------------------------------------------------------------------------------------------

    def _ctx(self, org: str) -> ValidationContext:
        return ValidationContext(self.primitives,
                                 approver_resolves=lambda role: self.membership.resolve(org, role) is not None)

    def _instances(self, org: str) -> list[dict]:
        records = [r for r in self.store.list(org, "workflows") if isinstance(r, dict) and r.get("instance_id")]
        return sorted(records, key=lambda r: (r.get("created_at") or "", r["instance_id"]))

    def _load(self, org: str, iid: Any) -> dict:
        rec = self.store.read(org, "workflows", iid) if isinstance(iid, str) and HEX_ID.fullmatch(iid) else None
        if rec is None:
            raise WorkflowServiceError(404, "workflow_not_found", "Workflow not found.")
        return rec

    def _write(self, org: str, rec: dict) -> None:
        self.store.write(org, "workflows", rec["instance_id"], rec)

    # -- actions: each applied once ----------------------------------------------------------------------------

    @staticmethod
    def _check_action(action_id: Any) -> str:
        if not isinstance(action_id, str) or len(action_id) > 128 or not ACTION_ID.fullmatch(action_id):
            raise _bad_input("action_id must be 1 to 128 lowercase letters, digits, '_', '-' or '.'.")
        return action_id

    def _once(self, org: str, kind: str, scope: str, action_id: str, request: Mapping, fn: Callable[[], Any],
              *, replay: Callable[[], Any] | None = None) -> Any:
        """Run fn once per (kind, scope, action_id). The same request while it runs waits for it; afterwards it
        gets the saved result (or `replay`, a fresh view) and a saved failure is raised again. Another request
        under the same action id is refused."""
        self._check_action(action_id)
        key = f"{kind}.{scope}.{action_id}"
        try:
            request_digest = digest(request)
        except (TypeError, ValueError):
            raise _bad_input("The request is not valid JSON.") from None
        with self._guard:
            lock = self._action_locks.setdefault(key, threading.Lock())
        with lock:
            saved = self.store.read(org, "actions", key)
            if saved is not None:
                if saved.get("request_digest") != request_digest:
                    raise WorkflowServiceError(409, "action_conflict", "This action id was used for a different request.")
                error = saved.get("error")
                if error:
                    raise WorkflowServiceError(error["status"], error["code"], error["message"])
                return replay() if replay is not None else copy.deepcopy(saved.get("result"))
            try:
                result = fn()
            except Exception as exc:
                err = _convert(exc)
                if err is None:
                    raise
                self.store.write(org, "actions", key, {"request_digest": request_digest, "error": {
                    "status": err.status, "code": err.code, "message": err.message}})
                if err is exc:
                    raise
                raise err from exc
            self.store.write(org, "actions", key, {"request_digest": request_digest, "result": result})
            return result

    # -- views --------------------------------------------------------------------------------------------------

    def _creation_view(self, rec: Mapping) -> dict:
        saved, draft = rec.get("version"), rec.get("draft") or {}
        source = saved or draft
        view: dict[str, Any] = {
            "schema": DYNAMIC_SCHEMA, "instance": {"instance_id": rec["instance_id"]}, "need": rec["need"],
            "action_id": rec["action_id"], "created_at": rec.get("created_at"),
            "status": "confirmed" if saved else "draft", "version": 1, "draft_id": draft.get("draft_id"),
            "question": None, "sample_ids": [], "run_actions": list(rec.get("run_actions") or []),
            "workflow": {"spec": copy.deepcopy(source.get("spec")), "metrics": copy.deepcopy(source.get("metrics")),
                         "gaps": copy.deepcopy(source.get("gaps") or []), "questions": [], "diff": None}}
        if draft.get("receipts"):
            view["authoring_receipt"] = dict(draft["receipts"][-1])
        if draft.get("notes"):
            view["authoring_notes"] = list(draft["notes"])
        return view

    def _readiness(self, org: str, rec: Mapping) -> tuple[dict, list[str], dict]:
        """(readiness, structural errors, blocked steps): gaps stop a run at their step; errors stop it starting."""
        saved = rec.get("version")
        if saved is None:
            return {"status": "needs_input"}, ["not_saved"], {}
        report = validate_spec(WorkflowSpec.from_dict(saved["spec"]), self._ctx(org))
        gaps = list(dict.fromkeys((g.get("step_id"), g["kind"], g["detail"])
                                  for g in [*report.gaps, *(saved.get("gaps") or [])]))
        roles = {s["id"]: s.get("role", "") for s in saved["spec"]["steps"]}
        blocked_steps = [{"step_id": s, "role": roles.get(s, ""), "kind": k, "detail": d} for s, k, d in gaps]
        blocked: dict[str, list] = {}
        for step_id, kind, detail in gaps:
            if step_id:
                blocked.setdefault(step_id, []).append({"kind": kind, "detail": detail})
        if report.errors:
            return ({"status": "needs_capability", "missing": list(report.errors), "blocked_steps": blocked_steps},
                    list(report.errors), blocked)
        if gaps:
            return ({"status": "needs_capability", "missing": [d for _s, _k, d in gaps], "blocked_steps": blocked_steps},
                    [], blocked)
        return {"status": "configured"}, [], blocked

    def _interrupted(self, org: str, run: Mapping) -> bool:
        return run["state"] == "running" and not self.executor.live(org, run)

    def _run_summaries(self, org: str, runs: Iterable[Mapping]) -> list[dict]:
        out = []
        for run in sorted(runs, key=lambda r: (r.get("started_at") or "", r.get("run_id") or ""), reverse=True):
            state = "interrupted" if self._interrupted(org, run) else run["state"]
            stopped = None
            if state not in ("completed", "running"):
                step = next((s for s in run.get("steps") or [] if s.get("state") != "completed"), None)
                if step is not None:
                    stopped = {"step_id": step["step_id"], "role": step.get("role", "")}
            items = run.get("items") or []
            out.append({"action_id": run["run_id"].partition(".")[2], "version": run["version"], "state": state,
                        "started_at": run.get("started_at"), "finished_at": run.get("completed_at"),
                        "stopped_at": stopped,
                        "items": sum(1 for i in items if i.get("status") == "active") if items else None})
        return out

    def _snapshot(self, org: str, rec: Mapping, runs: Iterable[Mapping]) -> dict:
        readiness, errors, _blocked = self._readiness(org, rec)
        saved = rec.get("version")
        definition: dict[str, Any] = {
            "execution_kind": "workflow_steps",
            "completion_criteria": list(saved["spec"]["completion_criteria"]) if saved else [],
            "action": {"label": "Run workflow", "enabled": not errors,
                       "path": f"{ROUTE_BASE}/{rec['instance_id']}/runs"}}
        if saved is not None:
            definition = {"version": 1, **definition}
        return {"creation": self._creation_view(rec), "readiness": readiness, "definition": definition,
                "runs": self._run_summaries(org, runs), "family": None}

    def _run_view(self, org: str, run: Mapping) -> dict:
        state = "interrupted" if self._interrupted(org, run) else run["state"]
        started, finished = _parse_time(run.get("started_at")), _parse_time(run.get("completed_at"))
        retryable, next_action = _next_action(state)
        payload = run.get("input") or {}
        view: dict[str, Any] = {
            "status": "complete" if state == "completed" else state,
            "action_id": run["run_id"].partition(".")[2], "version": run["version"],
            "input": {"source_kind": payload.get("source_kind"), "content": copy.deepcopy(payload.get("content"))},
            "execution_receipt": _execution_receipt(run["steps"]),
            "workflow_run": {
                "state": state, "steps": [_step_view(s) for s in run["steps"]],
                "metrics": {"human_interventions": run.get("human_interventions", 0),
                            "elapsed_seconds": (finished - started).total_seconds() if started and finished else None,
                            "model_calls": run.get("model_calls", 0), "decision_calls": run.get("decision_calls", 0)},
                "deliverable": copy.deepcopy(run.get("deliverable")),
                "pending_approval_ids": [s["approval_id"] for s in run["steps"]
                                         if s.get("state") == "waiting_approval" and s.get("approval_id")],
                "missing_sources": list(run.get("missing_sources") or []),
                "retryable": retryable, "next_action": next_action},
            "usage": run_usage(run)}
        if run.get("items"):
            view["workflow_run"]["items"] = [{k: copy.deepcopy(i.get(k)) for k in ("key", "label", "status", "reason")}
                                             for i in run["items"]]
        if run.get("code"):
            view["code"] = run["code"]
            if run["code"] in MODEL_ERRORS:
                view["message"] = model_error_message(run["code"])
        if state == "interrupted":
            view["message"] = "The run stopped before it finished. Continue it to pick up where it left off."
        if state == "completed":
            view["result"] = capability_result(run)
            folder = self._delivery_folder(run)
            if folder is not None and (folder / "result.md").is_file():
                view["delivered_to"] = str(folder)
        return view

    def _read_run(self, org: str, iid: str, action_id: Any) -> dict:
        if not isinstance(action_id, str) or len(action_id) > 128 or not ACTION_ID.fullmatch(action_id):
            raise WorkflowServiceError(404, "run_not_found", "Run not found.")
        run = self.executor.read(org, f"{iid}.{action_id}")
        if run.get("instance_id") != iid:
            raise WorkflowServiceError(404, "run_not_found", "Run not found.")
        return run

    # -- running ------------------------------------------------------------------------------------------------

    def _schedule(self, org: str, run_id: str) -> None:
        """Hand the run to a background job when it can go on and no job has it yet."""
        if self.executor.wake(org, run_id) != "dispatch":
            return
        try:
            self.submit(lambda: self.drive(org, run_id))
        except Exception as exc:
            logger.warning("workflow run %s could not be started (%s)", run_id, type(exc).__name__)
            self.executor.abandon(org, run_id)

    def drive(self, org: str, run_id: str) -> str:
        """Advance the run until it pauses or ends. "done": it paused or ended, or another thread has it; "failed":
        an unexpected error (the run then reads as interrupted and can be continued)."""
        try:
            run = self.executor.advance(org, run_id)
        except Exception as exc:
            logger.warning("workflow run %s failed to advance (%s)", run_id, type(exc).__name__)
            return "failed"
        self._deliver(run)
        return "done"

    def _delivery_folder(self, run: Mapping) -> Path | None:
        """deliveries/<workflow>/<started, as YYYYMMDD-hhmm>-<8 characters of the run's digest>/"""
        if self.deliveries is None:
            return None
        started = re.sub(r"[^0-9]", "", str(run.get("started_at") or ""))[:12]
        when = f"{started[:8]}-{started[8:]}-" if len(started) == 12 else ""
        run_key = digest(run["run_id"]).removeprefix("sha256:")[:8]
        return self.deliveries / _slug(str(run["spec"].get("domain") or "")) / f"{when}{run_key}"

    def _deliver(self, run: Mapping) -> None:
        """A completed run leaves result.md and package.json in its folder. A disk error is logged, never raised:
        the run itself is done and keeps its result."""
        folder = self._delivery_folder(run)
        if folder is None or run.get("state") != "completed":
            return
        try:
            folder.mkdir(parents=True, exist_ok=True)
            _write(folder / "package.json", json.dumps(run.get("deliverable"), ensure_ascii=False, indent=2) + "\n")
            _write(folder / "result.md", result_markdown(run))
        except OSError as exc:
            logger.warning("workflow run %s could not be saved to its folder (%s)", run["run_id"],
                           type(exc).__name__)

    # -- public API ---------------------------------------------------------------------------------------------

    @_public
    def catalog(self, org: str) -> dict:
        runs: dict[str, list] = {}
        for run in self.store.list(org, "runs"):
            if isinstance(run, Mapping) and run.get("instance_id"):
                runs.setdefault(run["instance_id"], []).append(run)
        return {"creations": [self._snapshot(org, r, runs.get(r["instance_id"], [])) for r in self._instances(org)]}

    @_public
    def detail(self, org: str, iid: str) -> dict:
        rec = self._load(org, iid)
        runs = [r for r in self.store.list(org, "runs") if isinstance(r, Mapping) and r.get("instance_id") == iid]
        return self._snapshot(org, rec, runs)

    @_public
    def create(self, org: str, action_id: str, need: str, source: str | None = None) -> dict:
        """A new workflow draft from a description of the job."""
        if not isinstance(need, str) or not need.strip() or len(need) > MAX_NEED:
            raise _bad_input(f"need must be 1 to {MAX_NEED} characters.")
        if source is not None and source not in ("chat", "workflows"):
            raise _bad_input("source must be 'chat' or 'workflows'.")

        def fn() -> dict:
            result = self.compiler.compile(org_id=org, description=need, ctx=self._ctx(org),
                                           primitive_refs=self.primitives.ids())
            iid, now = self.new_id(), self.clock()
            rec = {"instance_id": iid, "need": need, "action_id": action_id, "created_at": now, "source": source,
                   "draft": {"draft_id": self.new_id(), "spec": result.spec.to_dict(), "metrics": result.metrics,
                             "gaps": [dict(g) for g in result.gaps], "receipts": [dict(r) for r in result.receipts],
                             "created_at": now},
                   "version": None, "run_actions": []}
            if result.notes:   # what was tidied to read the model's output; absent from a clean build
                rec["draft"]["notes"] = list(result.notes)
            with self.store.locked(org):
                self._write(org, rec)
            return self._creation_view(rec)

        return self._once(org, "create", "new", action_id, {"need": need}, fn)

    @_public
    def confirm(self, org: str, iid: str, action_id: str, draft_id: str) -> dict:
        """Save the draft: it becomes the version runs use."""
        self._load(org, iid)

        def fn() -> dict:
            with self.store.locked(org):
                rec = self._load(org, iid)
                draft = rec.get("draft") or {}
                if draft.get("draft_id") != draft_id:
                    raise WorkflowServiceError(404, "draft_not_found", "Draft not found.")
                if rec.get("version") is not None:
                    raise WorkflowServiceError(409, "already_saved", "This workflow is already saved.")
                spec = WorkflowSpec.from_dict(draft["spec"])
                report = validate_spec(spec, self._ctx(org))
                if report.errors:
                    raise WorkflowServiceError(422, "compile_invalid",
                                               f"The workflow is no longer valid: {validation_problems(report.errors)}")
                gaps = list(dict.fromkeys((g["step_id"], g["kind"], g["detail"]) for g in [*report.gaps, *draft["gaps"]]))
                rec["version"] = {"version": 1, "draft_id": draft_id, "spec": spec.to_dict(), "spec_digest": spec.digest(),
                                  "metrics": report.metrics,
                                  "gaps": [{"step_id": s, "kind": k, "detail": d} for s, k, d in gaps],
                                  "confirmed_at": self.clock()}
                self._write(org, rec)
            return self._creation_view(rec)

        return self._once(org, "confirm", iid, action_id, {"draft_id": draft_id}, fn)

    @_public
    def run(self, org: str, iid: str, action_id: str, input: Any) -> dict:
        """Start a run of the saved version on the given input: {"version": 1, "content": {name: text}}."""
        self._check_action(action_id)
        rec = self._load(org, iid)
        run_id = f"{iid}.{action_id}"
        existing = self.store.read(org, "runs", run_id)
        if existing is not None and existing.get("instance_id") == iid:
            return self._run_view(org, existing)
        if not isinstance(input, Mapping) or set(input) - {"version", "content"} or "content" not in input:
            raise _bad_input('input needs "version" and "content" (input names to text).')
        if input.get("version", 1) != 1 or rec.get("version") is None:
            raise WorkflowServiceError(409, "workflow_not_ready", "Save the workflow before running it.")
        _readiness, errors, blocked = self._readiness(org, rec)
        if errors:
            raise WorkflowServiceError(409, "workflow_not_ready",
                                       f"The workflow cannot run: {validation_problems(errors)}")
        spec = WorkflowSpec.from_dict(rec["version"]["spec"])
        content = _check_content(spec, input["content"])

        def fn() -> dict:
            self.executor.start(org_id=org, run_id=run_id, instance_id=iid, version=1, spec=spec,
                                input_payload={"source_kind": "supplied_input", "content": content},
                                blocked_steps=blocked)
            with self.store.locked(org):
                current = self._load(org, iid)
                if action_id not in current["run_actions"]:
                    current["run_actions"].append(action_id)
                    self._write(org, current)
            self._schedule(org, run_id)
            return {"run_id": run_id}

        self._once(org, "run", iid, action_id, dict(input), fn)
        return self.run_view(org, iid, action_id)

    @_public
    def run_view(self, org: str, iid: str, action_id: str) -> dict:
        self._load(org, iid)
        return self._run_view(org, self._read_run(org, iid, action_id))

    @_public
    def resume_run(self, org: str, iid: str, action_id: str) -> dict:
        """Continue a run that stopped before it finished (its process ended)."""
        self._load(org, iid)
        run = self._read_run(org, iid, action_id)
        if not self._interrupted(org, run):
            raise WorkflowServiceError(409, "run_not_retryable", "Only an interrupted run can be continued.")
        self._schedule(org, run["run_id"])
        return self.run_view(org, iid, action_id)

    @_public
    def retry(self, org: str, iid: str, action_id: str, step_id: str | None = None,
              item_key: str | None = None) -> dict:
        self._load(org, iid)
        run = self._read_run(org, iid, action_id)
        if step_id is not None and not isinstance(step_id, str):
            raise WorkflowServiceError(404, "step_not_found", "Step not found.")
        if item_key is not None and not isinstance(item_key, str):
            raise WorkflowServiceError(404, "item_not_found", "Item not found.")
        self.executor.retry(org, run["run_id"], step_id, item_key=item_key)
        self._schedule(org, run["run_id"])
        return self.run_view(org, iid, action_id)

    @_public
    def approvals(self, org: str, user: str) -> dict:
        pending = [a for a in self.store.list_by(org, "approvals", status="pending", assignee=user)
                   if a.get("status") == "pending" and a.get("assignee") == user]
        pending.sort(key=lambda a: (a.get("created_at") or "", a.get("approval_id") or ""), reverse=True)
        return {"approvals": [_approval_view(a) for a in pending]}

    @_public
    def decide(self, org: str, user: str, approval_id: str, action_id: str, decision: str, content_digest: str,
               edited_content: Any = None, note: str | None = None, choices: Any = None,
               item_decisions: Any = None) -> dict:
        if not isinstance(approval_id, str) or not HEX_ID.fullmatch(approval_id):
            raise WorkflowServiceError(404, "approval_not_found", "Approval not found.")
        if not isinstance(decision, str) or not isinstance(content_digest, str):
            raise WorkflowServiceError(400, "invalid_decision", "decision and content_digest are required.")
        if note is not None and (not isinstance(note, str) or len(note) > MAX_NOTE):
            raise _bad_input(f"note must be at most {MAX_NOTE} characters.")
        approval = self.store.read(org, "approvals", approval_id)
        if approval is None:
            raise WorkflowServiceError(404, "approval_not_found", "Approval not found.")
        run_id = approval["run_id"]
        iid, _, run_action = run_id.partition(".")
        request = {"approval_id": approval_id, "decision": decision, "content_digest": content_digest,
                   "edited_content": edited_content, "note": note, "choices": choices, "item_decisions": item_decisions}

        def fn() -> dict:
            run = self.executor.decide(org, approval_id, actor=user, decision=decision, content_digest=content_digest,
                                       edited_content=edited_content, note=note, choices=choices,
                                       item_decisions=item_decisions)
            if run["state"] == "running":
                self._schedule(org, run_id)
            return {"run_id": run_id}

        self._once(org, "decide", approval_id, action_id, request, fn)
        return self.run_view(org, iid, run_action)
