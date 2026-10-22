"""/web/business-capabilities on the local server: the routes the terminal uses, over WorkflowService.

The same paths, bodies and replies as the hosted service, so one terminal works with both. Only what the local
engine does is here: list, build, save, run, follow a run, continue or retry it, and decide approvals. Reads need
the local session; changes also need the X-DaoBrew-Web header. Service errors become {"detail": {"code",
"message"}} with the service's status. A run shows its model calls and token usage.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Callable, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, StringConstraints

from daoharness.service import WorkflowService, WorkflowServiceError

# The service's form: lowercase dot-separated segments, at most 128 characters.
ActionId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_-]+(?:\.[a-z0-9_-]+)*$", max_length=128)]
_INSTANCE_ID = re.compile(r"[0-9a-f]{32}")


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateBody(_Body):
    action_id: ActionId
    need: str
    domain: Any = None          # sent by some callers; the compiler derives the name
    configuration: Any = None
    source: Optional[Literal["chat", "workflows"]] = None


class ConfirmBody(_Body):
    action_id: ActionId
    draft_id: str


class RunBody(_Body):
    action_id: ActionId
    input: dict  # the service validates its shape


class EmptyBody(_Body):
    pass


class RetryBody(_Body):
    step_id: Optional[str] = None
    item_key: Optional[str] = None


class DecisionBody(_Body):
    action_id: ActionId
    decision: str
    content_digest: str
    edited_content: Any = None
    note: Optional[str] = None
    choices: Any = None
    item_decisions: Any = None


def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except WorkflowServiceError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message}) from None


def _instance(instance_id: str) -> str:
    if not _INSTANCE_ID.fullmatch(instance_id):
        raise HTTPException(status_code=404, detail={"code": "workflow_not_found", "message": "Workflow not found."})
    return instance_id


def create_workflow_router(service: WorkflowService, *, reader: Callable[..., str],
                           writer: Callable[..., str]) -> APIRouter:
    """reader / writer: FastAPI dependencies that return the signed-in person (the local session)."""
    router = APIRouter(prefix="/web/business-capabilities", tags=["workflows"])

    @router.get("/catalog")
    def catalog(user: str = Depends(reader)) -> dict:
        return _call(service.catalog, user)

    @router.get("/approvals")
    def approvals(user: str = Depends(reader)) -> dict:
        return _call(service.approvals, user, user)

    @router.post("/approvals/{approval_id}/decision")
    def decide(approval_id: str, body: DecisionBody, user: str = Depends(writer)) -> dict:
        return _call(service.decide, user, user, approval_id, body.action_id, body.decision, body.content_digest,
                     edited_content=body.edited_content, note=body.note, choices=body.choices,
                     item_decisions=body.item_decisions)

    @router.post("/creations")
    def create(body: CreateBody, user: str = Depends(writer)) -> dict:
        return _call(service.create, user, body.action_id, body.need, body.source)

    @router.get("/{instance_id}")
    def detail(instance_id: str, user: str = Depends(reader)) -> dict:
        return _call(service.detail, user, _instance(instance_id))

    @router.post("/{instance_id}/confirm")
    def confirm(instance_id: str, body: ConfirmBody, user: str = Depends(writer)) -> dict:
        return _call(service.confirm, user, _instance(instance_id), body.action_id, body.draft_id)

    @router.post("/{instance_id}/runs")
    def run(instance_id: str, body: RunBody, user: str = Depends(writer)) -> dict:
        return _call(service.run, user, _instance(instance_id), body.action_id, body.input)

    @router.get("/{instance_id}/runs/{action_id}")
    def run_view(instance_id: str, action_id: str, user: str = Depends(reader)) -> dict:
        return _call(service.run_view, user, _instance(instance_id), action_id)

    @router.post("/{instance_id}/runs/{action_id}/resume")
    def resume(instance_id: str, action_id: str, user: str = Depends(writer),
               body: Optional[EmptyBody] = None) -> dict:
        return _call(service.resume_run, user, _instance(instance_id), action_id)

    @router.post("/{instance_id}/runs/{action_id}/retry")
    def retry(instance_id: str, action_id: str, user: str = Depends(writer),
              body: Optional[RetryBody] = None) -> dict:
        return _call(service.retry, user, _instance(instance_id), action_id, body.step_id if body else None,
                     item_key=body.item_key if body else None)

    return router
