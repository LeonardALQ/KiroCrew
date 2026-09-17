"""Private scan failures retain safe branch facts, never filesystem payloads."""

from __future__ import annotations

import errno
import os
import shutil
import threading
from pathlib import Path

import pytest

from kiro_crew import platform_compat, sandbox, session_map
from kiro_crew.workflows.runner import describe_agent_error

PREFIX = "memory_unavailable: cannot verify protected memory hardlinks"
PRIVATE = "private-canary-秘密-opaque"


@pytest.fixture
def scan_home(tmp_path, monkeypatch):
    root = (tmp_path / PRIVATE).resolve()
    root.mkdir()
    monkeypatch.setattr(sandbox, "config_dir", lambda: root)
    monkeypatch.setattr(session_map, "config_dir", lambda: root)
    # Every scan receives this explicit layout; never discover host roots.
    layout = sandbox._PrivateMemoryLayout((str(root),), (), ())
    return root, layout


def assert_diagnostic(error, operation, tree, code, *, winerror=None):
    windows_code = f" winerror={winerror}" if winerror is not None else ""
    expected = f"{PREFIX} (operation={operation} tree={tree} errno={code}{windows_code})"
    assert str(error) == expected
    assert describe_agent_error(error) == f"RuntimeError: {expected}"
    assert PRIVATE not in str(error)
    assert PRIVATE not in describe_agent_error(error)


@pytest.mark.parametrize("tree", ["root_tmp", "sessions", "snapshots", "memory"])
def test_actual_map_publication_race_is_tolerated_in_every_tree(scan_home, monkeypatch, tree):
    """A staged file the writer publishes mid-scan is gone, not a failure."""
    root, layout = scan_home
    mapping = session_map.SessionMap()
    if tree != "root_tmp":
        mapping._path = root / tree / f"{PRIVATE}.json"
    staged, publish, done = threading.Event(), threading.Event(), threading.Event()
    temporary, failures = [], []
    original_replace, original_stat = os.replace, Path.stat

    def replace(source, destination, *args, **kwargs):
        if str(destination) == str(mapping._path):
            temporary.append(Path(source))
            staged.set()
            assert publish.wait(5), "scanner did not reach the staged file"
        return original_replace(source, destination, *args, **kwargs)

    def scan_stat(path, *args, **kwargs):
        if temporary and path == temporary[0]:
            # The file really exists with one link before the writer publishes.
            assert original_stat(path).st_nlink == 1
            publish.set()
            assert done.wait(5), "writer did not publish"
        # No fabricated exception: the OS stats the name removed by os.replace.
        return original_stat(path, *args, **kwargs)

    def write():
        try:
            mapping._write_payload('{"fixture":true}', 1)
        except BaseException as exc:
            failures.append(exc)
        finally:
            done.set()

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(Path, "stat", scan_stat)
    thread = threading.Thread(target=write)
    thread.start()
    try:
        assert staged.wait(5), "writer did not stage"
        log_dir = sandbox._prepare_private_log_dir(layout)
    finally:
        publish.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not failures
    assert temporary, "scanner never reached the staged file"
    assert not temporary[0].exists()  # the OS really removed the name mid-scan
    assert mapping._written_seq == 1
    assert mapping._path.read_text(encoding="utf-8") == '{"fixture":true}'
    assert Path(log_dir).is_dir()
    assert Path(log_dir).parent == root / "memory_stores" / ".execution-logs"


@pytest.mark.parametrize("tree", ["sessions", "snapshots", "memory"])
def test_directory_removed_between_stat_and_listing_is_tolerated(scan_home, monkeypatch, tree):
    """A listed directory that vanishes before its own listing is gone, not a failure."""
    root, layout = scan_home
    parent = root / tree if tree != "memory" else root / "memory"
    doomed = parent / PRIVATE
    doomed.mkdir(parents=True)
    (doomed / "transcript.jsonl").write_text(PRIVATE, encoding="utf-8")
    original_iterdir = Path.iterdir
    removed = []

    def scan_iterdir(path):
        if path == doomed:
            # A real rmtree between the scanner's stat() and iterdir(): the OS
            # raises the ENOENT, nothing is fabricated.
            shutil.rmtree(path)
            removed.append(path)
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", scan_iterdir)
    log_dir = sandbox._prepare_private_log_dir(layout)
    assert removed == [doomed]
    assert not doomed.exists()
    assert Path(log_dir).is_dir()


@pytest.mark.parametrize("tree", ["root_tmp", "sessions", "snapshots", "memory"])
def test_vanished_parent_at_entry_stat_is_tolerated(scan_home, monkeypatch, tree):
    """ENOTDIR at entry_stat is the parent vanishing; it is the same race."""
    root, layout = scan_home
    if tree == "root_tmp":
        entry = root / f"{PRIVATE}.tmp"
    else:
        entry = root / tree / PRIVATE
        entry.parent.mkdir()
    entry.write_text(PRIVATE, encoding="utf-8")
    original_stat = Path.stat

    def scan_stat(path, *args, **kwargs):
        if path == entry:
            raise NotADirectoryError(errno.ENOTDIR, "Not a directory", str(path))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", scan_stat)
    assert Path(sandbox._prepare_private_log_dir(layout)).is_dir()


@pytest.mark.parametrize("tree", ["root_tmp", "sessions", "snapshots", "memory"])
@pytest.mark.parametrize(
    ("code", "error_type"),
    [(errno.EACCES, PermissionError), (errno.EIO, OSError), (errno.ELOOP, OSError)],
)
def test_other_entry_stat_failures_still_refuse(scan_home, monkeypatch, tree, code, error_type):
    """Only a vanished entry is skipped; every other stat failure stays fail-closed."""
    root, layout = scan_home
    if tree == "root_tmp":
        entry = root / f"{PRIVATE}.tmp"
    else:
        entry = root / tree / PRIVATE
        entry.parent.mkdir()
    entry.write_text(PRIVATE, encoding="utf-8")
    original_stat = Path.stat

    def scan_stat(path, *args, **kwargs):
        if path == entry:
            raise error_type(code, os.strerror(code), str(path))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", scan_stat)
    with pytest.raises(RuntimeError, match=PREFIX) as caught:
        sandbox._prepare_private_log_dir(layout)
    assert isinstance(caught.value.__cause__, error_type)
    assert_diagnostic(caught.value, "entry_stat", tree, code)
    assert not (root / "memory_stores" / ".execution-logs").exists()


def test_vanished_entry_predicate_names_exactly_two_codes():
    """The skip is keyed on errno alone, never on a tree or a path."""
    assert sandbox._private_memory_entry_vanished(FileNotFoundError(errno.ENOENT, "gone"))
    assert sandbox._private_memory_entry_vanished(NotADirectoryError(errno.ENOTDIR, "gone"))
    for code in (errno.EACCES, errno.EPERM, errno.EIO, errno.ELOOP, errno.ENAMETOOLONG):
        assert not sandbox._private_memory_entry_vanished(OSError(code, os.strerror(code)))
    assert not sandbox._private_memory_entry_vanished(OSError("no errno at all"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory mode enforcement")
@pytest.mark.parametrize("at_root", [True, False])
def test_real_unreadable_directory_still_refuses(scan_home, at_root):
    if platform_compat.local_user_id() == 0:
        pytest.skip("root bypasses POSIX directory mode checks")
    root, layout = scan_home
    target = root if at_root else root / "sessions" / PRIVATE
    target.mkdir(parents=True, exist_ok=True)
    platform_compat.chmod_safe(target, 0)
    try:
        with pytest.raises(RuntimeError, match=PREFIX) as caught:
            sandbox._prepare_private_log_dir(layout)
    finally:
        platform_compat.chmod_safe(target, 0o700)
    assert isinstance(caught.value.__cause__, PermissionError)
    assert_diagnostic(
        caught.value,
        "root_iterdir" if at_root else "entry_iterdir",
        "memory" if at_root else "sessions",
        errno.EACCES,
    )
    assert not (root / "memory_stores" / ".execution-logs").exists()


def test_real_hardlink_retains_distinct_refusal(scan_home):
    root, layout = scan_home
    protected = root / "memory.db"
    protected.write_text(PRIVATE, encoding="utf-8")
    alias = root / PRIVATE
    alias.hardlink_to(protected)
    assert protected.stat().st_nlink == 2
    with pytest.raises(RuntimeError, match="protected memory has a hardlink alias") as caught:
        sandbox._prepare_private_log_dir(layout)
    assert caught.value.__cause__ is None
    assert "operation=" not in str(caught.value)
    assert PRIVATE not in describe_agent_error(caught.value)
    assert not (root / "memory_stores" / ".execution-logs").exists()


def test_missing_required_workspace_retains_existing_refusal(scan_home):
    root, _ = scan_home
    missing = str(root / "required-workspace")
    layout = sandbox._PrivateMemoryLayout((str(root),), (missing,), (missing,))
    with pytest.raises(RuntimeError, match="required workspace directory") as caught:
        sandbox._prepare_private_log_dir(layout)
    assert missing in str(caught.value)  # Existing required-root remedy stays unchanged.
    assert "operation=" not in str(caught.value)
    assert not (root / "memory_stores" / ".execution-logs").exists()


@pytest.mark.parametrize("value", [None, True, PRIVATE, -1, 1 << 100])
def test_non_numeric_or_unbounded_codes_are_not_serialized(value):
    cause = OSError(errno.EACCES, PRIVATE, f"/{PRIVATE}")
    cause.errno = value
    cause.winerror = value
    error = sandbox._private_memory_scan_failure("entry_stat", "memory", cause)
    assert str(error) == f"{PREFIX} (operation=entry_stat tree=memory)"
    assert PRIVATE not in describe_agent_error(error)


def test_numeric_windows_code_survives_workflow_serialization():
    cause = OSError(errno.EACCES, PRIVATE, f"/{PRIVATE}")
    cause.winerror = 32
    error = sandbox._private_memory_scan_failure("entry_iterdir", "snapshots", cause)
    expected = f"{PREFIX} (operation=entry_iterdir tree=snapshots errno=13 winerror=32)"
    assert_diagnostic(error, "entry_iterdir", "snapshots", errno.EACCES, winerror=32)
    assert str(error) == expected
    assert describe_agent_error(error) == f"RuntimeError: {expected}"
    assert PRIVATE not in describe_agent_error(error)
