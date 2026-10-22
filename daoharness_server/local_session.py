"""Sign-in for the local server: one random token in ~/.daoharness/local-session, readable only by you.

The terminal reads the token from that file. A change also needs the X-DaoBrew-Web header, which a web page on
another site cannot send, so a page you visit cannot act on your workflows.
"""
from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path

from fastapi import HTTPException, Request

COOKIE = "daobrew_web_session"   # the hosted service's cookie name, so the terminal sends the same request to both
LOCAL_USER = "local"


def token_path(home: Path) -> Path:
    return Path(home) / "local-session"


def ensure_token(home: Path) -> str:
    """The saved token, or a new one written with owner-only permissions."""
    path = token_path(home)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        token = path.read_text("utf-8").strip()
        if len(token) >= 32:
            return token
    except OSError:
        pass
    token = secrets.token_urlsafe(32)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token)
    return token


def guards(token: str):
    """(reader, writer) FastAPI dependencies: the session token for both, the header too for changes."""

    def reader(request: Request) -> str:
        if not hmac.compare_digest(request.cookies.get(COOKIE, ""), token):
            raise HTTPException(status_code=401, detail={"code": "WEB_SESSION_REQUIRED",
                                                         "message": "Local session required"})
        return LOCAL_USER

    def writer(request: Request) -> str:
        user = reader(request)
        if request.headers.get("X-DaoBrew-Web") != "1":
            raise HTTPException(status_code=403, detail={"code": "WEB_CSRF_REQUIRED",
                                                         "message": "Web CSRF header required"})
        return user

    return reader, writer
