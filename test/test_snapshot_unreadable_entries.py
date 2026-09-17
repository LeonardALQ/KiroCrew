"""A data-home entry the process cannot stat must not abort ``kirocrew snapshot``.

Two independent halves, and only the first one is what a user hits:

* the pre-flight size estimate walked the WHOLE data home before the component
  selection was applied, so a protected file the process cannot ``stat`` -- one
  that no component even declares -- raised out of the generator and the command
  died with a traceback. ``--components`` could not work around it, because the
  walk ran first.
* the staging pass reads the selected trees themselves, so an unreadable entry
  INSIDE one of them ended the snapshot too. There the file is genuinely wanted,
  so it is skipped, warned about, and recorded in ``MANIFEST.json`` rather than
  dropped silently.

The refusal is simulated rather than produced with a mode bit: ``chmod 000`` does
nothing for root, is not honoured the same way on Windows, and the class of file
this reproduces (a platform-protected path) cannot be created by a test at all.
Denying the three syscalls for ONE uniquely-named entry is deterministic on every
platform the suite runs on.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import tarfile

import pytest
from test_snapshot import _setup_fake_kirocrew, unpinnable_argv

from kiro_crew import pinned_fs
from kiro_crew import snapshot as snap

# Unique enough that the interception below can key on the base name alone, which is
# what makes it work for a bare name opened relative to a directory descriptor as
# well as for a full path.
VICTIM = "protected-entry.md"
#: A DIRECTORY the process may not list. Separate from VICTIM because the two are
#: refused at different syscalls and reach different branches of the walk.
DIR_VICTIM = "protected-dir"
#: A `.db` the process may not read. Named for its SUFFIX: the pass that re-copies
#: databases screens on that before it asks the filesystem anything.
DB_VICTIM = "protected.db"

#: The platform's REAL answer, taken at import before any fixture patches ``os``.
#: ``supports_pinned_tree_walk`` asks whether ``os.listdir``/``os.stat`` are members
#: of ``os.supports_fd``/``os.supports_dir_fd``, and a wrapper installed over either
#: one is not -- so a denial fixture silently flipped the probe to False and sent
#: every test here down the by-name branch, leaving the pinned branch that actually
#: ships on Linux and macOS untested. Each fixture pins this value back.
PINNED_TREE_WALK = pinned_fs.supports_pinned_tree_walk()


def _is_denied(path, name: str, home) -> bool:
    """Whether *path* names the live *name* under *home*, or names it bare.

    A BARE name is how the pinned walks ask -- through a directory descriptor -- so it has
    to be denied for the refusal to be visible to them. An absolute path is denied only
    inside the data home: the staged copy in the bundle has the same base name, and
    refusing that would fake a failure the product never produces (a staged path that is
    not there raises ENOENT) and would refuse reads of the staging tree these tests do not
    target.
    """
    if isinstance(path, int):
        return False
    try:
        text = os.fsdecode(path)
    except (TypeError, ValueError):
        return False
    if os.path.basename(text) != name:
        return False
    if not os.path.dirname(text):
        return True
    return text.startswith(str(home)) or text.startswith(os.path.realpath(home))


@pytest.fixture
def home(tmp_path, monkeypatch):
    d = tmp_path / "home"
    d.mkdir()
    monkeypatch.setenv("KIROCREW_HOME", str(d))
    monkeypatch.setenv("KIROCREW_ASSUME_GATEWAY_RUNNING", "0")
    _setup_fake_kirocrew(d)
    return d


@pytest.fixture
def deny_victim(home, monkeypatch):
    """Make every access to a file named :data:`VICTIM` fail with ``EPERM``.

    ``stat``, ``lstat`` and ``open`` together are what "the process cannot touch
    this file" means: a walk that only stats is stopped by the first two, and the
    by-name staging fallback reaches the third.
    """
    real = {"stat": os.stat, "lstat": os.lstat, "open": os.open}
    monkeypatch.setattr(pinned_fs, "supports_pinned_tree_walk", lambda: PINNED_TREE_WALK)

    def _deny(key):
        def _fn(path, *args, **kwargs):
            if _is_denied(path, VICTIM, home):
                raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
            return real[key](path, *args, **kwargs)

        return _fn

    for key in real:
        monkeypatch.setattr(os, key, _deny(key))


@pytest.fixture
def deny_open_only(home, monkeypatch):
    """Readable metadata, refused bytes -- mode 000 inside a readable directory.

    Distinct from :func:`deny_victim` on purpose: when the stat is refused too, the
    walk classifies the entry and skips it before it ever tries to copy, so the copy's
    own handler is never reached and a mutation to it survives.
    """
    real = os.open
    monkeypatch.setattr(pinned_fs, "supports_pinned_tree_walk", lambda: PINNED_TREE_WALK)

    def _fn(path, *args, **kwargs):
        if _is_denied(path, VICTIM, home):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", _fn)


@pytest.fixture
def unpinned(monkeypatch):
    """Take the by-name traversal a platform without directory descriptors takes."""
    monkeypatch.setattr(snap.pinned_fs, "supports_pinned_tree_walk", lambda: False)


def _deny_listing(monkeypatch, name: str, home) -> None:
    """Refuse ``scandir``/``listdir``/``open`` for the live directory called *name*."""
    real = {"scandir": os.scandir, "listdir": os.listdir, "open": os.open}
    monkeypatch.setattr(pinned_fs, "supports_pinned_tree_walk", lambda: PINNED_TREE_WALK)

    def _deny(key):
        def _fn(path=".", *args, **kwargs):
            if _is_denied(path, name, home):
                raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
            return real[key](path, *args, **kwargs)

        return _fn

    for key in real:
        monkeypatch.setattr(os, key, _deny(key))


@pytest.fixture
def deny_dir(home, monkeypatch):
    """Make :data:`DIR_VICTIM` a directory this process may not list."""
    _deny_listing(monkeypatch, DIR_VICTIM, home)


@pytest.fixture
def deny_victim_db(home, monkeypatch):
    """Refuse every access to the live :data:`DB_VICTIM`."""
    real = {"stat": os.stat, "lstat": os.lstat, "open": os.open}
    monkeypatch.setattr(pinned_fs, "supports_pinned_tree_walk", lambda: PINNED_TREE_WALK)

    def _deny(key):
        def _fn(path, *args, **kwargs):
            if _is_denied(path, DB_VICTIM, home):
                raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
            return real[key](path, *args, **kwargs)

        return _fn

    for key in real:
        monkeypatch.setattr(os, key, _deny(key))


@pytest.fixture
def deny_dir_stat(home, monkeypatch):
    """Make :data:`DIR_VICTIM` a directory whose METADATA cannot be read either.

    The harder case, and the reported one: when the stat itself is refused,
    ``os.path.isdir`` and ``os.path.islink`` both answer False rather than raising, so
    a screen that asks either of them first reads the refusal as an ordinary file.
    """
    _deny_listing(monkeypatch, DIR_VICTIM, home)
    real = {"stat": os.stat, "lstat": os.lstat}

    def _deny(key):
        def _fn(path, *args, **kwargs):
            if _is_denied(path, DIR_VICTIM, home):
                raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
            return real[key](path, *args, **kwargs)

        return _fn

    for key in real:
        monkeypatch.setattr(os, key, _deny(key))


def _bundles(out) -> list:
    """Every bundle, either prefix. These tests skip something, so most are partial."""
    return sorted(out.glob(f"{snap.SNAPSHOT_PREFIX}*.tar.gz")) + sorted(
        out.glob(f"{snap.PARTIAL_PREFIX}*.tar.gz")
    )


def _manifest_of(out, tmp_path) -> dict:
    tarballs = _bundles(out)
    assert tarballs, "no bundle was written"
    extract = tmp_path / "extract"
    extract.mkdir()
    with tarfile.open(str(tarballs[0])) as tar:
        tar.extractall(extract, filter=lambda t, _d="": t)
    roots = [d for d in extract.iterdir() if d.is_dir()]
    assert len(roots) == 1
    return json.loads((roots[0] / "MANIFEST.json").read_text(encoding="utf-8"))


class TestAProtectedFileNoComponentWantsIsNeverTouched:
    """The reported crash: the file is not in any component, and still killed the run."""

    def test_snapshot_succeeds(self, home, deny_victim, tmp_path, capsys):
        (home / VICTIM).write_text("a protected file at the data home root\n")
        out = tmp_path / "out"
        rc = snap.snapshot_main([str(out), *unpinnable_argv()])
        assert rc == 0, capsys.readouterr()
        assert _bundles(out), "no bundle was written"

    def test_a_scoped_snapshot_succeeds(self, home, deny_victim, tmp_path, capsys):
        """``--components config,memory`` failed identically before the fix."""
        (home / VICTIM).write_text("a protected file at the data home root\n")
        out = tmp_path / "out"
        rc = snap.snapshot_main([str(out), "--components", "config,memory", *unpinnable_argv()])
        assert rc == 0, capsys.readouterr()
        assert _bundles(out), "no bundle was written"

    def test_the_estimate_does_not_report_it_as_skipped(self, home, deny_victim, tmp_path, capsys):
        """It is outside every component, so the estimate never looks at it at all."""
        (home / VICTIM).write_text("a protected file at the data home root\n")
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), *unpinnable_argv()]) == 0
        assert "unreadable" not in capsys.readouterr().err


class TestABundleThatOmittedSomethingIsNamedPartial:
    """Retention rotates per prefix, so this is what keeps a complete backup safe."""

    def test_the_tarball_carries_the_partial_prefix(self, home, deny_victim, tmp_path):
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()]) == 0
        names = [b.name for b in _bundles(out)]
        assert names, "no bundle was written"
        assert all(n.startswith(snap.PARTIAL_PREFIX) for n in names), names

    def test_a_complete_run_keeps_the_ordinary_prefix(self, home, tmp_path):
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), *unpinnable_argv()]) == 0
        names = [b.name for b in _bundles(out)]
        assert names and all(n.startswith(snap.SNAPSHOT_PREFIX) for n in names), names

    def test_a_partial_does_not_evict_a_complete_one_at_keep_1(self, home, deny_victim, tmp_path):
        """The reported hazard: `--keep 1` deleting the last good backup for a partial."""
        out = tmp_path / "out"
        out.mkdir()
        existing = out / f"{snap.SNAPSHOT_PREFIX}20260101T000000Z.tar.gz"
        with tarfile.open(existing, "w:gz") as tf:
            payload = tmp_path / "payload"
            payload.mkdir()
            tf.add(str(payload), arcname=f"{snap.SNAPSHOT_PREFIX}20260101T000000Z")
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        rc = snap.snapshot_main(
            [str(out), "--components", "workspace", "--keep", "1", *unpinnable_argv()]
        )
        assert rc == 0
        assert existing.is_file(), "a partial bundle pruned the last complete backup"
        assert sorted(out.glob(f"{snap.PARTIAL_PREFIX}*.tar.gz")), "no partial was written"

    def test_partials_themselves_still_rotate(self, home, deny_victim, tmp_path):
        """Bounded, so tolerating a refusal cannot fill the disk either."""
        out = tmp_path / "out"
        out.mkdir()
        stale = out / f"{snap.PARTIAL_PREFIX}20260101T000000Z.tar.gz"
        with tarfile.open(stale, "w:gz") as tf:
            payload = tmp_path / "payload"
            payload.mkdir()
            tf.add(str(payload), arcname=f"{snap.PARTIAL_PREFIX}20260101T000000Z")
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        rc = snap.snapshot_main(
            [str(out), "--components", "workspace", "--keep", "1", *unpinnable_argv()]
        )
        assert rc == 0
        assert not stale.exists(), "the older partial was not rotated out"
        assert len(sorted(out.glob(f"{snap.PARTIAL_PREFIX}*.tar.gz"))) == 1

    def test_list_shows_a_partial_bundle(self, home, deny_victim, tmp_path, capsys):
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()]) == 0
        capsys.readouterr()
        assert snap.snapshot_main([str(out), "--list"]) == 0
        assert snap.PARTIAL_PREFIX in capsys.readouterr().out

    def test_the_success_line_says_it_is_partial(self, home, deny_victim, tmp_path, capsys):
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()]) == 0
        assert "PARTIAL bundle" in capsys.readouterr().out


class TestTheEstimateIsScopedToTheSelectedComponents:
    def test_it_counts_only_what_the_selection_stages(self, home):
        big = b"x" * 4096
        (home / "workspace" / "big.bin").write_bytes(big)
        (home / "crons.json").write_bytes(b"y" * 32)
        crons_only, unreadable = snap._estimate_selected_bytes(home, ["crons"])
        assert crons_only == 32
        assert unreadable == 0
        with_workspace, _ = snap._estimate_selected_bytes(home, ["crons", "workspace"])
        assert with_workspace >= 32 + len(big)

    def test_an_overlapping_tree_is_counted_once(self, home):
        """`memory` names workspace/memory, which lives inside `workspace`."""
        (home / "workspace" / "memory" / "note.md").write_bytes(b"z" * 100)
        both, _ = snap._estimate_selected_bytes(home, ["memory", "workspace"])
        workspace_only, _ = snap._estimate_selected_bytes(home, ["workspace"])
        own_files = sum(
            (home / f).stat().st_size
            for f in snap.COMPONENTS["memory"].files
            if (home / f).is_file()
        )
        assert own_files > 0, "fixture no longer has the memory component's own files"
        assert both == workspace_only + own_files


class TestTheEstimateAbsorbsOnlyThePermissionClass:
    """A number cannot say "the storage is failing"."""

    def test_a_failing_stat_is_raised_not_counted(self, home, monkeypatch):
        real = os.lstat
        target = home / "workspace" / "doc.md"

        def _fn(path, *args, **kwargs):
            if not isinstance(path, int):
                try:
                    hit = os.fspath(path) == str(target)
                except TypeError:
                    hit = False
                if hit:
                    raise OSError(errno.EIO, "Input/output error", str(path))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(os, "lstat", _fn)
        with pytest.raises(OSError) as caught:
            snap._estimate_selected_bytes(home, ["workspace"])
        assert caught.value.errno == errno.EIO

    def test_a_failing_directory_listing_is_raised_not_counted(self, home, monkeypatch):
        real = os.scandir
        target = home / "workspace" / "memory"

        def _fn(path=".", *args, **kwargs):
            if not isinstance(path, int):
                try:
                    hit = os.fspath(path) == str(target)
                except TypeError:
                    hit = False
                if hit:
                    raise OSError(errno.EIO, "Input/output error", str(path))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(os, "scandir", _fn)
        with pytest.raises(OSError) as caught:
            snap._estimate_selected_bytes(home, ["workspace"])
        assert caught.value.errno == errno.EIO


class TestARefusalInAnOverlappingSelectionIsCountedOnce:
    """`memory` names workspace/memory, which sits under `workspace`."""

    def test_one_refused_directory_is_reported_once(self, home, deny_dir, tmp_path):
        (home / "workspace" / "memory" / DIR_VICTIM).mkdir()
        _, unreadable = snap._estimate_selected_bytes(home, ["memory", "workspace"])
        assert unreadable == 1

    def test_the_stderr_line_says_one_not_two(self, home, deny_dir, tmp_path, capsys):
        (home / "workspace" / "memory" / DIR_VICTIM).mkdir()
        out = tmp_path / "out"
        rc = snap.snapshot_main([str(out), "--components", "memory,workspace", *unpinnable_argv()])
        assert rc == 0
        assert "skipped 1 unreadable entry while estimating the size" in capsys.readouterr().err


class TestAnUnreadableEntryInsideASelectedTree:
    def test_the_estimate_counts_it_instead_of_raising(self, home, deny_victim):
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        total, unreadable = snap._estimate_selected_bytes(home, ["workspace"])
        assert unreadable >= 1
        assert total >= 0

    def test_it_says_how_many_on_stderr(self, home, deny_victim, tmp_path, capsys):
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()]) == 0
        assert "skipped 1 unreadable entry while estimating the size" in capsys.readouterr().err

    def test_a_file_whose_bytes_alone_are_refused_is_recorded(self, home, deny_open_only, tmp_path):
        """Reaches the refusal at the copy rather than at the classification."""
        (home / "workspace" / VICTIM).write_text("readable metadata, refused bytes\n")
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()]) == 0
        manifest = _manifest_of(out, tmp_path)
        omitted = [s for s in manifest.get("skipped", []) if s["path"].endswith(VICTIM)]
        assert omitted, f"manifest does not record the omission: {manifest.get('skipped')}"
        assert omitted[0]["reason"] == "unreadable_entry"

    def test_the_snapshot_still_completes_and_says_how_many(self, home, deny_victim, tmp_path):
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        out = tmp_path / "out"
        rc = snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()])
        assert rc == 0
        assert _bundles(out), "no bundle was written"

    def test_the_bundle_records_the_omission(self, home, deny_victim, tmp_path):
        """A backup that could not read a file must SAY so, not omit it silently."""
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()]) == 0
        manifest = _manifest_of(out, tmp_path)
        omitted = [s for s in manifest.get("skipped", []) if s["path"].endswith(VICTIM)]
        assert omitted, f"manifest does not record the omission: {manifest.get('skipped')}"
        assert omitted[0]["reason"] == "unreadable_entry"


class TestAFileWhoseBytesAloneAreRefusedOnTheByNameWalk:
    """The by-name copy has its own tolerance, and only this reaches it.

    Every other by-name case here denies the stat as well, so the ignore screen
    classifies the entry and drops it before any copy is attempted.
    """

    def test_it_is_skipped_and_recorded(self, home, deny_open_only, unpinned, tmp_path):
        (home / "workspace" / VICTIM).write_text("readable metadata, refused bytes\n")
        out = tmp_path / "out"
        rc = snap.snapshot_main([str(out), "--components", "workspace", "--allow-unpinned-staging"])
        assert rc == 0
        manifest = _manifest_of(out, tmp_path)
        omitted = [s for s in manifest.get("skipped", []) if s["path"].endswith(VICTIM)]
        assert omitted, f"manifest does not record the omission: {manifest.get('skipped')}"
        assert omitted[0]["reason"] == "unreadable_entry"


class TestARefusedDIRECTORYIsTheSameOmission:
    """A tolerance that covered only files still ended the snapshot on a directory."""

    def test_a_refused_subdirectory_is_skipped_and_recorded(self, home, deny_dir, tmp_path):
        (home / "workspace" / DIR_VICTIM).mkdir()
        (home / "workspace" / DIR_VICTIM / "inside.md").write_text("unreachable\n")
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()]) == 0
        manifest = _manifest_of(out, tmp_path)
        omitted = [s for s in manifest.get("skipped", []) if s["path"].endswith(DIR_VICTIM)]
        assert omitted, f"manifest does not record the omission: {manifest.get('skipped')}"
        assert omitted[0]["reason"] == "unreadable_entry"

    def test_a_refused_subdirectory_is_skipped_on_the_by_name_walk(
        self, home, deny_dir, unpinned, tmp_path
    ):
        (home / "workspace" / DIR_VICTIM).mkdir()
        (home / "workspace" / DIR_VICTIM / "inside.md").write_text("unreachable\n")
        out = tmp_path / "out"
        rc = snap.snapshot_main([str(out), "--components", "workspace", "--allow-unpinned-staging"])
        assert rc == 0
        manifest = _manifest_of(out, tmp_path)
        omitted = [s for s in manifest.get("skipped", []) if s["path"].endswith(DIR_VICTIM)]
        assert omitted, f"manifest does not record the omission: {manifest.get('skipped')}"
        assert omitted[0]["reason"] == "unreadable_entry"

    def test_a_refused_component_ROOT_is_skipped_and_recorded(self, home, monkeypatch, tmp_path):
        """The root is listed before any per-entry screen runs, on both traversals."""
        _deny_listing(monkeypatch, "plan_memory", home)
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()]) == 0
        manifest = _manifest_of(out, tmp_path)
        omitted = [s for s in manifest.get("skipped", []) if s["path"].endswith("plan_memory")]
        assert omitted, f"manifest does not record the omission: {manifest.get('skipped')}"
        assert omitted[0]["reason"] == "unreadable_entry"


class TestTheByNameRootsOwnRefusal:
    """`copytree` lists the tree's own directory before any per-entry screen runs."""

    def test_it_is_reported_and_the_tree_omitted(self, home, tmp_path, monkeypatch):
        _deny_listing(monkeypatch, "plan_memory", home)
        monkeypatch.setattr(snap.pinned_fs, "supports_pinned_tree_walk", lambda: False)
        seen: list[tuple[str, str]] = []
        snap._copytree_safe(
            home / "plan_memory",
            tmp_path / "dst",
            allow_unpinned=True,
            on_skip=lambda reason, path: seen.append((reason, path)),
            skip_unreadable=True,
        )
        assert [p for r, p in seen if r == "unreadable_entry" and p.endswith("plan_memory")], seen

    def test_it_still_raises_when_the_tolerance_is_off(self, home, tmp_path, monkeypatch):
        _deny_listing(monkeypatch, "plan_memory", home)
        monkeypatch.setattr(snap.pinned_fs, "supports_pinned_tree_walk", lambda: False)
        with pytest.raises((PermissionError, shutil.Error)):
            snap._copytree_safe(home / "plan_memory", tmp_path / "dst", allow_unpinned=True)


class TestADirectoryWhoseMetadataIsRefusedToo:
    """`os.path.isdir` answers False for a path it cannot stat, so it cannot screen."""

    def test_the_pinned_walk_skips_and_records_it(self, home, deny_dir_stat, tmp_path):
        (home / "workspace" / DIR_VICTIM).mkdir()
        (home / "workspace" / DIR_VICTIM / "inside.md").write_text("unreachable\n")
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()]) == 0
        manifest = _manifest_of(out, tmp_path)
        omitted = [s for s in manifest.get("skipped", []) if s["path"].endswith(DIR_VICTIM)]
        assert omitted, f"manifest does not record the omission: {manifest.get('skipped')}"
        assert omitted[0]["reason"] == "unreadable_entry"

    def test_the_by_name_walk_skips_and_records_it(self, home, deny_dir_stat, unpinned, tmp_path):
        (home / "workspace" / DIR_VICTIM).mkdir()
        (home / "workspace" / DIR_VICTIM / "inside.md").write_text("unreachable\n")
        out = tmp_path / "out"
        rc = snap.snapshot_main([str(out), "--components", "workspace", "--allow-unpinned-staging"])
        assert rc == 0
        manifest = _manifest_of(out, tmp_path)
        omitted = [s for s in manifest.get("skipped", []) if s["path"].endswith(DIR_VICTIM)]
        assert omitted, f"manifest does not record the omission: {manifest.get('skipped')}"
        assert omitted[0]["reason"] == "unreadable_entry"


class TestARefusedDatabaseNeverReachesTheRestagePass:
    """The destination is asked about first, and that is what keeps this safe.

    A statically unreadable `.db` is skipped by the tree walk, so no byte copy exists at
    the destination and the restage pass ends the iteration before it stats the source.
    Asking the SOURCE first -- which is what the pre-change `src.is_file()` did -- raises
    the refusal out of the loop and ends the whole snapshot.
    """

    def test_the_snapshot_completes_and_records_it(self, home, deny_victim_db, tmp_path):
        (home / "workspace" / DB_VICTIM).write_bytes(b"SQLite format 3\x00stub")
        out = tmp_path / "out"
        assert snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()]) == 0
        manifest = _manifest_of(out, tmp_path)
        omitted = [s for s in manifest.get("skipped", []) if s["path"].endswith(DB_VICTIM)]
        assert omitted, f"manifest does not record the omission: {manifest.get('skipped')}"
        assert omitted[0]["reason"] == "unreadable_entry"


class TestTheRestagePassFailsClosedOnAnUnreadableDestination(object):
    """An `EIO` reading the staged copy is not "there is nothing staged"."""

    def test_it_propagates_rather_than_keeping_the_byte_copy(self, home, tmp_path, monkeypatch):
        (home / "workspace" / "wal.db").write_bytes(b"SQLite format 3\x00stub")
        staged_name = "wal.db"
        real = os.stat
        fired = {"n": 0}

        def _fn(path, *args, **kwargs):
            if not isinstance(path, int) and fired["n"] == 0:
                try:
                    text = os.fsdecode(path)
                except (TypeError, ValueError):
                    text = ""
                # The STAGED copy only: under the bundle, not under the data home. Fired
                # ONCE, so a swallowed error lets the run finish rather than resurfacing at
                # the next pass that stats the same file -- which is what makes this able
                # to tell the narrow catch from the broad one.
                staged = (
                    os.path.basename(text) == staged_name
                    # A real directory component: the walks ask through a descriptor with
                    # a BARE name, and matching that fired on the live tree's own
                    # classification instead of on the staged copy.
                    and os.path.dirname(text)
                    and not text.startswith(str(home))
                    and not text.startswith(os.path.realpath(home))
                )
                if staged:
                    fired["n"] += 1
                    raise OSError(errno.EIO, "Input/output error", text)
            return real(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", _fn)
        monkeypatch.setattr(pinned_fs, "supports_pinned_tree_walk", lambda: PINNED_TREE_WALK)
        out = tmp_path / "out"
        with pytest.raises(OSError) as caught:
            snap.snapshot_main([str(out), "--components", "workspace", *unpinnable_argv()])
        assert caught.value.errno == errno.EIO


class TestTheEstimateDoesNotFollowASymlinkedTreeRoot:
    """Staging refuses such a root outright, but only after the estimate has run."""

    def test_its_bytes_are_not_counted(self, home, tmp_path):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "big.bin").write_bytes(b"x" * 8192)
        shutil.rmtree(home / "plan_memory")
        try:
            (home / "plan_memory").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("this platform cannot create a directory symlink")
        total, _ = snap._estimate_selected_bytes(home, ["workspace"])
        assert total < 8192, "the estimate followed a symlinked component root"


class TestADestinationRootThatCannotBeCreated:
    """Reported as an unreadable SOURCE, a bundle shipped without the whole tree."""

    def test_it_propagates_instead_of_being_recorded(self, home, tmp_path, monkeypatch):
        monkeypatch.setattr(snap.pinned_fs, "supports_pinned_tree_walk", lambda: False)
        dst = tmp_path / "dst-root"
        real = os.makedirs

        def _fn(path, *args, **kwargs):
            if str(path).startswith(str(dst)):
                raise PermissionError(errno.EACCES, "Permission denied", str(path))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(os, "makedirs", _fn)
        seen: list[tuple[str, str]] = []
        with pytest.raises(PermissionError):
            snap._copytree_safe(
                home / "workspace",
                dst,
                allow_unpinned=True,
                on_skip=lambda reason, path: seen.append((reason, path)),
                skip_unreadable=True,
            )
        assert not [r for r, _ in seen if r == "unreadable_entry"], seen


class TestOnlyTheSOURCEReadIsTolerated:
    """A refusal to WRITE is not the operator's file being unreadable.

    Recording it as one omits data from the bundle, blames the source for it, and
    reports success -- so retention can then prune a bundle that was complete.
    """

    def _deny_open_of(self, monkeypatch, target):
        real = os.open

        def _fn(path, *args, **kwargs):
            if not isinstance(path, int):
                try:
                    hit = os.fspath(path) == str(target)
                except TypeError:
                    hit = False
                if hit:
                    raise PermissionError(errno.EACCES, "Permission denied", str(path))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", _fn)

    def test_a_destination_refusal_propagates(self, tmp_path, monkeypatch):
        src = tmp_path / "src.txt"
        src.write_text("payload\n")
        dst = tmp_path / "out" / "dst.txt"
        dst.parent.mkdir()
        self._deny_open_of(monkeypatch, dst)
        seen: list[tuple[str, str]] = []
        with pytest.raises(PermissionError):
            pinned_fs.copy_file_pinned(
                str(src),
                str(dst),
                skip_unreadable=True,
                on_skip=lambda reason, path: seen.append((reason, path)),
            )
        assert not [r for r, _ in seen if r == "unreadable_entry"], seen

    def test_a_source_refusal_is_recorded_and_skipped(self, tmp_path, monkeypatch):
        src = tmp_path / "src.txt"
        src.write_text("payload\n")
        dst = tmp_path / "out" / "dst.txt"
        dst.parent.mkdir()
        self._deny_open_of(monkeypatch, src)
        seen: list[tuple[str, str]] = []
        copied = pinned_fs.copy_file_pinned(
            str(src),
            str(dst),
            skip_unreadable=True,
            on_skip=lambda reason, path: seen.append((reason, path)),
        )
        assert copied is False
        assert [r for r, _ in seen if r == "unreadable_entry"], seen
        assert not dst.exists()

    def test_a_source_refusal_still_propagates_by_default(self, tmp_path, monkeypatch):
        src = tmp_path / "src.txt"
        src.write_text("payload\n")
        dst = tmp_path / "out" / "dst.txt"
        dst.parent.mkdir()
        self._deny_open_of(monkeypatch, src)
        with pytest.raises(PermissionError):
            pinned_fs.copy_file_pinned(str(src), str(dst))


class TestTheToleranceIsOffUnlessAskedFor:
    """Restore and merge must keep ending on a refusal, so the OFF side needs pinning.

    Without these, deleting every ``if not skip_unreadable: raise`` guard reds no test
    and restore silently becomes skip-and-drop -- the archive's own content, dropped
    from what the operator asked to have put back.
    """

    @pytest.mark.skipif(not PINNED_TREE_WALK, reason="no directory descriptors on this platform")
    def test_the_pinned_walk_raises_by_default(self, home, deny_victim, tmp_path):
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        with pytest.raises(PermissionError):
            pinned_fs.stage_tree_pinned(
                home / "workspace", tmp_path / "dst", what="tree 'workspace'"
            )

    def test_the_by_name_walk_raises_by_default(self, home, deny_victim, unpinned, tmp_path):
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        with pytest.raises((PermissionError, shutil.Error)):
            snap._copytree_safe(home / "workspace", tmp_path / "dst", allow_unpinned=True)

    def test_a_refused_directory_still_ends_a_default_walk(self, home, deny_dir, tmp_path):
        (home / "workspace" / DIR_VICTIM).mkdir()
        with pytest.raises((PermissionError, shutil.Error)):
            snap._copytree_safe(
                home / "workspace", tmp_path / "dst", allow_unpinned=True, skip_unreadable=False
            )

    def test_nothing_is_recorded_when_the_tolerance_is_off(self, home, deny_victim, tmp_path):
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        seen: list[tuple[str, str]] = []
        with pytest.raises((PermissionError, shutil.Error)):
            snap._copytree_safe(
                home / "workspace",
                tmp_path / "dst",
                allow_unpinned=True,
                on_skip=lambda reason, path: seen.append((reason, path)),
            )
        assert not [r for r, _ in seen if r == "unreadable_entry"]


class TestTheByNameStagingFallbackToleratesItToo:
    """The branch a platform without directory descriptors takes.

    Forced rather than skipped-unless-Windows: the branch ships on a real platform, and
    its screens reach the refusal at a different syscall -- ``os.path.islink`` answers
    False for a path it cannot ``lstat`` instead of raising, so the entry arrives as a
    copy that fails rather than as a classification that does.
    """

    def test_it_completes_and_records_the_omission(self, home, deny_victim, unpinned, tmp_path):
        (home / "workspace" / VICTIM).write_text("wanted, but unreadable\n")
        out = tmp_path / "out"
        rc = snap.snapshot_main([str(out), "--components", "workspace", "--allow-unpinned-staging"])
        assert rc == 0
        manifest = _manifest_of(out, tmp_path)
        omitted = [s for s in manifest.get("skipped", []) if s["path"].endswith(VICTIM)]
        assert omitted, f"manifest does not record the omission: {manifest.get('skipped')}"
        assert omitted[0]["reason"] == "unreadable_entry"
