"""Local JSON file store: one private directory per org, atomic writes, per-org reentrant lock."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from .spec import canonical_json

try:
    import fcntl
except ImportError:   # Windows has no fcntl: msvcrt locks a byte range of the file instead
    fcntl = None
    import msvcrt
LOCK_BACKEND = "fcntl" if fcntl is not None else "msvcrt"
LOCK_TIMED_OUT = getattr(errno, "EDEADLOCK", 36)   # what msvcrt.locking raises after ten seconds of waiting
COLLECTIONS = frozenset({"workflows", "runs", "samples", "actions", "approvals", "notifications",
                         "decision_labels", "step_cache", "primitives", "primitive_drafts", "metrics", "settings"})
# Lowercase only: APFS is case-insensitive, so mixed-case keys could alias one file. Used with fullmatch.
KEY = re.compile(r"[a-z0-9_][a-z0-9_.-]{0,199}")


def _lock(handle) -> None:
    """Take an exclusive lock on the open lock file, waiting until it is free."""
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return
    handle.seek(0)
    while True:   # msvcrt.locking gives up after about ten seconds; flock waits for ever, so this waits on
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return
        except OSError as exc:
            if exc.errno != LOCK_TIMED_OUT:
                raise


def _unlock(handle) -> None:
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_UN)
        return
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:   # closing the file lets go of it anyway
        pass


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
                    if hasattr(os, "fchmod"):   # Windows has it only from Python 3.13
                        os.fchmod(fd, 0o600)  # O_CREAT's mode applies only on creation; repair an older lock file
                    _lock(handle)
                yield
            finally:
                if handle is not None:
                    _unlock(handle)
                    handle.close()
                held[org_id] = depth
