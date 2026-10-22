"""Local JSON file store: one private directory per org, atomic writes, per-org reentrant lock."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from .spec import canonical_json

COLLECTIONS = frozenset({"workflows", "runs", "samples", "actions", "approvals", "notifications",
                         "decision_labels", "step_cache", "primitives", "primitive_drafts", "metrics", "settings"})
# Lowercase only: APFS is case-insensitive, so mixed-case keys could alias one file. Used with fullmatch.
KEY = re.compile(r"[a-z0-9_][a-z0-9_.-]{0,199}")


class WorkflowStoreError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class FileWorkflowStore:
    def __init__(self, root: Path) -> None:
        root = Path(root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root = root.resolve()
        self._guard = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}
        self._held = threading.local()

    def _org(self, org_id: str) -> Path:
        if not isinstance(org_id, str) or not org_id or len(org_id) > 256:
            raise WorkflowStoreError("invalid_org")
        path = self.root / hashlib.sha256(org_id.encode("utf-8")).hexdigest()[:32]
        path.mkdir(mode=0o700, exist_ok=True)
        return path

    def _file(self, org_id: str, collection: str, key: str) -> Path:
        if collection not in COLLECTIONS:
            raise WorkflowStoreError("invalid_collection")
        if not isinstance(key, str) or not KEY.fullmatch(key) or ".." in key:
            raise WorkflowStoreError("invalid_key")
        folder = self._org(org_id) / collection
        folder.mkdir(mode=0o700, exist_ok=True)
        return folder / f"{key}.json"

    def read(self, org_id: str, collection: str, key: str) -> dict | None:
        path = self._file(org_id, collection, key)
        try:
            return json.loads(path.read_text("utf-8"))
        except FileNotFoundError:
            return None

    def write(self, org_id: str, collection: str, key: str, value: Mapping[str, Any]) -> None:
        path = self._file(org_id, collection, key)
        data = canonical_json(value).encode("utf-8")
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def create(self, org_id: str, collection: str, key: str, value: Mapping[str, Any]) -> bool:
        with self.locked(org_id):
            if self.read(org_id, collection, key) is not None:
                return False
            self.write(org_id, collection, key, value)
            return True

    def list(self, org_id: str, collection: str) -> list[dict]:
        if collection not in COLLECTIONS:
            raise WorkflowStoreError("invalid_collection")
        folder = self._org(org_id) / collection
        if not folder.is_dir():
            return []
        return [json.loads(p.read_text("utf-8")) for p in sorted(folder.glob("*.json")) if not p.name.startswith(".")]

    def list_by(self, org_id: str, collection: str, **equals: str) -> list[dict]:
        """The records of list() whose top-level fields equal the given strings."""
        return [r for r in self.list(org_id, collection) if all(r.get(k) == v for k, v in equals.items())]

    @contextmanager
    def locked(self, org_id: str) -> Iterator[None]:
        org_dir = self._org(org_id)
        with self._guard:
            lock = self._locks.setdefault(org_id, threading.RLock())
        held = getattr(self._held, "depth", None)
        if held is None:
            held = self._held.depth = {}
        with lock:
            depth = held.get(org_id, 0)
            held[org_id] = depth + 1
            handle = None
            try:
                if depth == 0:
                    fd = os.open(org_dir / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
                    handle = os.fdopen(fd, "r+")
                    os.fchmod(fd, 0o600)  # O_CREAT's mode applies only on creation; repair an older lock file
                    fcntl.flock(handle, fcntl.LOCK_EX)
                yield
            finally:
                if handle is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()
                held[org_id] = depth
