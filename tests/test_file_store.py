import errno
import importlib.util
import os
import sys
import threading
import types

import pytest

from daoharness import file_store
from daoharness.file_store import FileWorkflowStore, WorkflowStoreError


def test_write_read_list_and_org_isolation(tmp_path):
    store = FileWorkflowStore(tmp_path / "wf")
    store.write("org-a", "workflows", "k1", {"x": 1})
    store.write("org-a", "workflows", "k0", {"x": 0})
    assert store.read("org-a", "workflows", "k1") == {"x": 1}
    assert store.read("org-b", "workflows", "k1") is None
    assert [r["x"] for r in store.list("org-a", "workflows")] == [0, 1]
    assert store.list("org-b", "workflows") == []


def test_create_is_first_writer_wins(tmp_path):
    store = FileWorkflowStore(tmp_path)
    assert store.create("o", "actions", "a.1", {"v": 1}) is True
    assert store.create("o", "actions", "a.1", {"v": 2}) is False
    assert store.read("o", "actions", "a.1") == {"v": 1}


@pytest.mark.parametrize("collection,key", [("nope", "k"), ("runs", "../x"), ("runs", ""), ("runs", "a/b"), ("runs", ".hidden"),
                                            ("runs", "k\n"), ("runs", "a.1\n"), ("runs", "Abc"), ("runs", "ABC.run")])
def test_rejects_bad_names(tmp_path, collection, key):
    with pytest.raises(WorkflowStoreError):
        FileWorkflowStore(tmp_path).write("o", collection, key, {})


def test_writes_are_atomic_and_private(tmp_path):
    store = FileWorkflowStore(tmp_path / "wf")
    store.write("o", "runs", "r", {"a": 1})
    files = list((tmp_path / "wf").rglob("*.json"))
    assert len(files) == 1 and oct(files[0].stat().st_mode & 0o777) == "0o600"
    assert not list((tmp_path / "wf").rglob("*.tmp"))


def test_lock_file_and_directories_are_private(tmp_path):
    store = FileWorkflowStore(tmp_path / "wf")
    with store.locked("o"):
        store.write("o", "runs", "r", {"a": 1})
    for path in (tmp_path / "wf").rglob("*"):
        mode = path.stat().st_mode & 0o777
        assert mode == (0o700 if path.is_dir() else 0o600), (path, oct(mode))


def test_existing_lock_file_is_made_private_on_open(tmp_path):
    # A .lock left at 0644 by an earlier version keeps its mode (O_CREAT's mode applies only on creation), so the
    # store repairs it when it opens the file.
    store = FileWorkflowStore(tmp_path / "wf")
    store.write("o", "runs", "r", {"a": 1})
    [run_file] = (tmp_path / "wf").rglob("*.json")
    lock = run_file.parent.parent / ".lock"          # <org dir>/runs/r.json -> <org dir>/.lock
    lock.touch()
    lock.chmod(0o644)
    with store.locked("o"):
        pass
    assert oct(lock.stat().st_mode & 0o777) == "0o600"


def test_failed_write_keeps_old_value_and_leaves_no_temp_file(tmp_path, monkeypatch):
    from daoharness import file_store

    store = FileWorkflowStore(tmp_path / "wf")
    store.write("o", "runs", "r", {"a": 1})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(file_store.os, "replace", boom)
    with pytest.raises(OSError):
        store.write("o", "runs", "r", {"a": 2})
    monkeypatch.undo()
    assert store.read("o", "runs", "r") == {"a": 1}
    assert not list((tmp_path / "wf").rglob("*.tmp"))


def test_locked_serializes_threads_and_is_reentrant(tmp_path):
    store = FileWorkflowStore(tmp_path)
    store.write("o", "runs", "c", {"n": 0})

    def bump():
        for _ in range(50):
            with store.locked("o"):
                with store.locked("o"):
                    value = store.read("o", "runs", "c")
                    store.write("o", "runs", "c", {"n": value["n"] + 1})

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.read("o", "runs", "c") == {"n": 200}


def test_lowercase_keys_with_dots_and_dashes_are_accepted(tmp_path):
    store = FileWorkflowStore(tmp_path)
    store.write("o", "runs", "0abc.def-1_x", {"v": 1})
    assert store.read("o", "runs", "0abc.def-1_x") == {"v": 1}


# --- locking on Windows ------------------------------------------------------------------------------------------

class FakeMsvcrt(types.ModuleType):
    """Windows' msvcrt as far as the store uses it. Every locking call is kept with the file position it was made at.
    `busy` lock attempts give up the way LK_LOCK does after ten seconds; `fail` is raised by every call."""

    LK_LOCK, LK_UNLCK = 1, 0

    def __init__(self, busy=0, fail=None):
        super().__init__("msvcrt")
        self.calls, self.busy, self.fail = [], busy, fail

    def locking(self, fd, mode, nbytes):
        self.calls.append((mode, nbytes, os.lseek(fd, 0, os.SEEK_CUR)))
        if self.fail:
            raise self.fail
        if mode == self.LK_LOCK and self.busy:
            self.busy -= 1
            raise OSError(file_store.LOCK_TIMED_OUT, "Resource deadlock avoided")


def windows_store(monkeypatch, tmp_path, msvcrt):
    """file_store.py loaded as on Windows, where there is no fcntl to import and msvcrt takes its place."""
    monkeypatch.setitem(sys.modules, "fcntl", None)      # "import fcntl" now raises ImportError
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt)
    spec = importlib.util.spec_from_file_location("daoharness.file_store_on_windows", file_store.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.FileWorkflowStore(tmp_path / "wf")


def test_the_store_imports_without_fcntl_and_locks_with_msvcrt(monkeypatch, tmp_path):
    module, _ = windows_store(monkeypatch, tmp_path, FakeMsvcrt())
    assert module.LOCK_BACKEND == "msvcrt" and module.fcntl is None


def test_where_there_is_fcntl_the_store_uses_it_exactly_as_before(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    assert file_store.LOCK_BACKEND == "fcntl" and file_store.fcntl is fcntl
    calls, real = [], fcntl.flock
    monkeypatch.setattr(fcntl, "flock", lambda handle, operation: (calls.append(operation), real(handle, operation))[1])
    store = FileWorkflowStore(tmp_path / "wf")
    with store.locked("o"):
        with store.locked("o"):
            store.write("o", "runs", "r", {"a": 1})
    assert calls == [fcntl.LOCK_EX, fcntl.LOCK_UN]


def test_the_windows_lock_is_one_byte_at_the_start_of_the_lock_file_and_is_reentrant(monkeypatch, tmp_path):
    msvcrt = FakeMsvcrt()
    _, store = windows_store(monkeypatch, tmp_path, msvcrt)
    with store.locked("o"):
        with store.locked("o"):
            store.write("o", "runs", "r", {"a": 1})
        assert len(msvcrt.calls) == 1
    assert msvcrt.calls == [(msvcrt.LK_LOCK, 1, 0), (msvcrt.LK_UNLCK, 1, 0)]
    assert store.read("o", "runs", "r") == {"a": 1}
    assert store.create("o", "actions", "a.1", {"v": 1}) is True and store.create("o", "actions", "a.1", {}) is False


def test_a_lock_another_process_holds_is_waited_for_not_given_up_on(monkeypatch, tmp_path):
    msvcrt = FakeMsvcrt(busy=2)       # LK_LOCK gives up after ten seconds; flock would still be waiting
    _, store = windows_store(monkeypatch, tmp_path, msvcrt)
    with store.locked("o"):
        pass
    assert [mode for mode, _, _ in msvcrt.calls] == [msvcrt.LK_LOCK] * 3 + [msvcrt.LK_UNLCK]


def test_any_other_locking_error_is_raised_and_the_store_stays_usable(monkeypatch, tmp_path):
    msvcrt = FakeMsvcrt(fail=OSError(errno.EBADF, "Bad file descriptor"))
    _, store = windows_store(monkeypatch, tmp_path, msvcrt)
    with pytest.raises(OSError) as raised:
        with store.locked("o"):
            pass
    assert raised.value.errno == errno.EBADF
    msvcrt.fail = None
    with store.locked("o"):
        store.write("o", "runs", "r", {"a": 2})
    assert store.read("o", "runs", "r") == {"a": 2}


def test_the_store_works_where_os_has_no_fchmod(monkeypatch, tmp_path):
    monkeypatch.delattr(os, "fchmod")       # Windows before Python 3.13
    _, store = windows_store(monkeypatch, tmp_path, FakeMsvcrt())
    with store.locked("o"):
        store.write("o", "runs", "r", {"a": 3})
    assert store.read("o", "runs", "r") == {"a": 3}


def test_two_threads_still_take_turns_under_the_windows_lock(monkeypatch, tmp_path):
    _, store = windows_store(monkeypatch, tmp_path, FakeMsvcrt())
    store.write("o", "runs", "c", {"n": 0})

    def bump():
        for _ in range(25):
            with store.locked("o"):
                value = store.read("o", "runs", "c")
                store.write("o", "runs", "c", {"n": value["n"] + 1})

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert store.read("o", "runs", "c") == {"n": 100}
