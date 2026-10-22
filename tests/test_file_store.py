import threading

import pytest

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
