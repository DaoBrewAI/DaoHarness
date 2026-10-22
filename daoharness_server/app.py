"""The local server that `daoharness serve` starts: 127.0.0.1 only, one person, files under ~/.daoharness.

Workflows and runs live in ~/.daoharness/workflows; each completed run also leaves result.md and package.json in
~/.daoharness/deliveries/<workflow>/<run>/.

It runs the engine with your model (see daoharness.adapters) and serves the routes the terminal uses. Nothing is
sent anywhere but your model provider. There is no web research: a step that needs the web shows as a gap.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from daoharness.file_store import FileWorkflowStore
from daoharness.service import WorkflowService

from .local_session import ensure_token, guards
from .routes import create_workflow_router

DEFAULT_PORT = 8767


class OneMember:
    """Every approver role is you."""

    def resolve(self, org_id: str, approver_role: str) -> str | None:
        return org_id if isinstance(approver_role, str) and approver_role.strip() else None


class NoNotifications:
    def approval_requested(self, org_id: str, approval: dict) -> None:
        pass


def local_service(home: Path, model: Any, *, submit=None, decisions=None) -> WorkflowService:
    """The engine on your model: it builds, runs steps and decides (decisions: a DecisionPort to decide instead)."""
    return WorkflowService(
        store=FileWorkflowStore(Path(home) / "workflows"), model=model, membership=OneMember(),
        notifier=NoNotifications(), clock=lambda: datetime.now(timezone.utc).isoformat(),
        submit=submit or (lambda job: threading.Thread(target=job, daemon=True).start()), decisions=decisions,
        deliveries=Path(home) / "deliveries")


def create_app(service: WorkflowService, *, token: str, port: int) -> FastAPI:
    app = FastAPI(title="DaoHarness local", docs_url=None, redoc_url=None, openapi_url=None)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{host}" for host in hosts}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        """Only this machine, by its loopback name, and never a request another site's page started."""
        origin, site = request.headers.get("origin"), request.headers.get("sec-fetch-site")
        if (request.headers.get("host", "").lower() not in hosts
                or (origin is not None and origin.rstrip("/") not in origins)
                or (site is not None and site not in ("same-origin", "none"))):
            return JSONResponse(status_code=403, content={"detail": {
                "code": "LOCAL_ORIGIN_REQUIRED", "message": "The local server only answers this machine."}})
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    reader, writer = guards(token)
    app.include_router(create_workflow_router(service, reader=reader, writer=writer))

    @app.get("/", response_class=PlainTextResponse)
    def home() -> str:
        return "DaoHarness local server. Use the terminal: daoharness new\n"

    return app


def serve(*, home: Path, port: int = DEFAULT_PORT, model: Any = None) -> None:
    import uvicorn

    from daoharness.adapters import OpenAICompatibleModel

    model = model if model is not None else OpenAICompatibleModel.from_env()
    token = ensure_token(home)
    app = create_app(local_service(home, model), token=token, port=port)
    print(f"DaoHarness local server on http://127.0.0.1:{port} · model {getattr(model, 'model', '?')} · "
          f"workflows in {Path(home) / 'workflows'}")
    print("In another terminal: daoharness new")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
