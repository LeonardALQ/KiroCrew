"""Recovering a cloud crew whose Kiro sign-in did not finish.

A managed launch can register a working, billing crew and still end up with no
Kiro session: the device code timed out, nobody approved it, the gateway
restarted mid-wait, or the owner cancelled. Three seams, no AWS and no live
gateway:

* ``cloud/login.py`` — :func:`~kiro_crew.cloud.login.cancel_device_login` stops a
  cancelled login WITHOUT signing the box out (which ``logout`` would do to a
  session the cancelled attempt never established).
* ``cloud/launch_job.py`` — :func:`~kiro_crew.cloud.launch_job.run_signin_retry`
  re-runs only the sign-in step, a cancel stops the remote login, and a restart
  mid-sign-in keeps a crew that is already signed in.
* ``dashboard/handlers_cloud.py`` — ``POST /api/cloud/launch/{id}/signin/restart``
  is owner-only, serialized, and never re-provisions.

The identity model itself (``KiroLoginTarget``, the pty login driver, launch-body
field validation) is covered by ``test_cloud_login_target.py``,
``test_cloud_login.py`` and ``test_cloud_handlers.py``.
"""

from __future__ import annotations

import json
import re
import threading
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from kiro_crew.cloud import launch_engine as le
from kiro_crew.cloud import launch_job as lj
from kiro_crew.cloud import login, ssm
from kiro_crew.cloud.login_target import KiroLoginTarget
from kiro_crew.dashboard import handlers_cloud as hc

COMPANY = KiroLoginTarget.from_fields(
    license="pro", start_url="https://amzn.awsapps.com/start", region="us-east-1"
)


# ── login.py: stopping a login is not signing out ─────────────────────────────
class TestCancelDeviceLogin:
    def test_cancelling_stops_the_login_without_signing_the_box_out(self):
        """A cancelled attempt must not drop a session it never established."""
        cmd = login._cancel_login_command()
        assert "kiro-cli login --use-device-flow" in cmd
        assert "logout" not in cmd
        for path in (login._LOGIN_LOG_PATH, login._LOGIN_PID_PATH, login._LOGIN_FIFO_PATH):
            assert path in cmd

    def test_it_never_kills_a_pid_read_from_the_shared_tmp_file(self):
        """The PID path is a fixed, world-writable ``/tmp`` location, so a local
        user who writes their chosen PID there would pick what this command kills
        — as whatever user the box's SSM agent runs. The kill must be the
        uid-scoped, command-line-matched ``pkill``, and the PID file only removed.
        """
        cmd = login._cancel_login_command()
        assert 'pkill -u "$(id -u)" -f "kiro-cli login --use-device-flow"' in cmd
        for line in cmd.splitlines():
            stripped = line.strip()
            if login._LOGIN_PID_PATH in line:
                assert stripped.startswith("rm -f "), f"PID file used for more than cleanup: {line}"
            assert not re.match(r"kill\b", stripped), f"an unscoped kill: {line}"
        assert "cat " not in cmd, "the PID file must not be read at all"

    def test_a_box_without_pkill_reports_the_cancel_as_unfinished(self, monkeypatch):
        """Nothing was killed, so this must NOT answer like a clean stop — the
        alternative (guessing a PID from the file) is the defect above."""
        cmd = login._cancel_login_command()
        assert login._CANCEL_NO_PKILL_SENTINEL in cmd
        assert "exit 1" in cmd
        # The whole point of the non-zero exit: the caller reports the failure.
        monkeypatch.setattr(
            ssm, "run_command", lambda *a, **k: ssm.CommandResult("Failed", "", "", 1)
        )
        assert login.cancel_device_login("i-0abc", "dev", "us-east-1") is False

    def test_cancel_device_login_reports_whether_the_cleanup_ran(self, monkeypatch):
        seen: dict = {}

        def fake_run(instance_id, command, *_a, **_k):
            seen["iid"] = instance_id
            seen["cmd"] = command
            return ssm.CommandResult("Success", "", "", 0)

        monkeypatch.setattr(ssm, "run_command", fake_run)
        assert login.cancel_device_login("i-0abc", "dev", "us-east-1") is True
        assert seen["iid"] == "i-0abc"
        assert "pkill" in seen["cmd"]

        monkeypatch.setattr(
            ssm, "run_command", lambda *a, **k: ssm.CommandResult("TimedOut", "", "", 1)
        )
        assert login.cancel_device_login("i-0abc", "dev", "us-east-1") is False


# ── launch_job.py fakes ───────────────────────────────────────────────────────
class FakeHandle:
    def __init__(self, *, already=False, url="", code="", signed=True, error=""):
        self.already_logged_in = already
        self.url = url
        self.code = code
        self.ports: list = []
        self.error = error
        self._signed = signed
        self.closed = False

    def wait(self, cancel: threading.Event) -> bool:
        return self._signed

    def close(self) -> None:
        self.closed = True


class AbortableHandle(FakeHandle):
    """A handle that can stop its remote login, and counts being asked to."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.aborted = 0

    def abort(self) -> bool:
        self.aborted += 1
        return True


class FailingAbortHandle(AbortableHandle):
    """Its abort RAN and reported that the remote login is still polling."""

    def abort(self) -> bool:
        super().abort()
        return False


class RaisingAbortHandle(AbortableHandle):
    """Its abort could not even be attempted (transport error)."""

    def abort(self) -> bool:
        super().abort()
        raise RuntimeError("ssm unreachable")


class NoAbortHandle(FakeHandle):
    """A handle with no ``abort`` at all -- the shape `FargateSigninHandle` has.

    Subclassing and deleting is not possible for a method the parent defines, so
    this asserts its own premise instead: the attribute must be absent for the
    test using it to mean anything.
    """

    def __getattribute__(self, name):  # noqa: D105 - see the class docstring
        if name == "abort":
            raise AttributeError("abort")
        return super().__getattribute__(name)


class LegacyAbortHandle(AbortableHandle):
    """An older handle on the answer-less contract: ``None`` is not a failure."""

    def abort(self) -> None:  # type: ignore[override]
        self.aborted += 1
        return None


class TargetEngine:
    """An engine on main's contract, recording the login target it was handed."""

    def __init__(self, handle=None):
        self.handle = handle or FakeHandle(already=True)
        self.seen: list = []

    def preflight(self, profile, region):
        pass

    def provision(self, *, tag, size_key, profile, region):
        return "i-0abc123456789def0"

    def begin_signin(self, *, instance_id, profile, region, login_target=None):
        self.seen.append(login_target)
        return self.handle

    def register(self, *, instance_id, tag, profile, region):
        pass

    def teardown(self, *, tag, profile, region):
        return True


class CountingEngine(TargetEngine):
    """Counts every engine call, so "it was never asked" is asserted, not inferred.

    A double that RAISED instead would be swallowed by ``run_signin_retry``'s
    ``except Exception`` (and by the route's worker), leaving the guard untested.
    """

    def __init__(self, handle=None):
        super().__init__(handle)
        self.preflights = 0
        self.provisions = 0
        self.signins = 0
        self.registrations = 0

    def preflight(self, profile, region):
        self.preflights += 1

    def provision(self, *, tag, size_key, profile, region):
        self.provisions += 1
        return "i-0deadbeefdeadbeef"

    def begin_signin(self, *, instance_id, profile, region, login_target=None):
        self.signins += 1
        return super().begin_signin(
            instance_id=instance_id, profile=profile, region=region, login_target=login_target
        )

    def register(self, *, instance_id, tag, profile, region):
        self.registrations += 1


def _store(tmp_path):
    return lj.LaunchJobStore(root=tmp_path / "launch-jobs")


# ── launch_job.py: a cancelled LAUNCH stops the remote login ──────────────────
class TestLaunchCancellation:
    """A cancelled LAUNCH must stop the remote login, not only forget the code.

    ``_rollback_cancelled_stack`` can end in DELETE_FAILED — reported, not ruled
    out — and an instance that survives with a login still polling would
    authenticate the crew minutes after the owner cancelled.
    """

    def _cancel_during_signin(self, tmp_path, handle, teardown_confirms=True):
        s = _store(tmp_path)
        job = s.create(profile="dev", region="us-east-1", size_key="balanced")
        cancel = threading.Event()

        class CancellingEngine(TargetEngine):
            def __init__(self, h):
                super().__init__(h)
                self.torn_down = 0

            def begin_signin(self, *, instance_id, profile, region, login_target=None):
                # The human gives up while the code is on screen.
                cancel.set()
                return super().begin_signin(
                    instance_id=instance_id,
                    profile=profile,
                    region=region,
                    login_target=login_target,
                )

            def teardown(self, *, tag, profile, region):
                self.torn_down += 1
                return teardown_confirms

        engine = CancellingEngine(handle)
        out = lj.run_launch(job, s, engine, cancel=cancel)
        return out, engine

    def test_a_cancelled_launch_stops_the_remote_login(self, tmp_path):
        handle = AbortableHandle(url="https://x/?user_code=A", signed=False)
        out, engine = self._cancel_during_signin(tmp_path, handle)
        assert out.status == lj.CANCELLED
        assert handle.aborted == 1
        assert engine.torn_down == 1

    def test_it_stops_the_login_even_when_the_teardown_does_not_confirm(self, tmp_path):
        """The case that makes this a defect rather than a tidiness point: the box
        outlives the cancel, so a login left polling can still sign it in."""
        handle = AbortableHandle(url="https://x/?user_code=A", signed=False)
        out, _ = self._cancel_during_signin(tmp_path, handle, teardown_confirms=False)
        assert out.status == lj.CANCELLED
        assert handle.aborted == 1
        assert "did NOT confirm" in out.error

    def test_it_stops_the_login_before_the_teardown_is_asked_for(self, tmp_path):
        """A teardown takes minutes; the browser tab is already open."""
        order: list = []

        class OrderedHandle(AbortableHandle):
            def abort(self) -> None:
                super().abort()
                order.append("abort")

        s = _store(tmp_path)
        job = s.create(profile="dev", region="us-east-1", size_key="balanced")
        cancel = threading.Event()

        class Engine(TargetEngine):
            def begin_signin(self, *, instance_id, profile, region, login_target=None):
                cancel.set()
                return self.handle

            def teardown(self, *, tag, profile, region):
                order.append("teardown")
                return True

        lj.run_launch(job, s, Engine(OrderedHandle(url="https://x/?user_code=A")), cancel=cancel)
        assert order == ["abort", "teardown"]

    def test_it_stops_the_login_before_the_steps_are_rewritten(self, tmp_path):
        """``_abort_signin`` is the FIRST statement of the cancel arm.

        The step rewrite and the rollback both run after it, so an abort placed
        later would be reached only once the (minutes-long) teardown returned.
        """
        seen: list = []
        s = _store(tmp_path)
        job = s.create(profile="dev", region="us-east-1", size_key="balanced")
        cancel = threading.Event()

        class WatchingHandle(AbortableHandle):
            def abort(self) -> None:
                super().abort()
                seen.append(job.step(lj.STEP_SIGNIN).state)

        class Engine(TargetEngine):
            def begin_signin(self, *, instance_id, profile, region, login_target=None):
                cancel.set()
                return self.handle

        lj.run_launch(job, s, Engine(WatchingHandle(url="https://x/?user_code=A")), cancel=cancel)
        assert seen == [lj.STEP_ACTIVE], "the step was already rewritten to skipped"

    def test_a_cancel_before_the_sign_in_has_no_login_to_stop(self, tmp_path):
        """Nothing was started, so nothing is aborted — and it must not raise."""
        s = _store(tmp_path)
        job = s.create(profile="dev", region="us-east-1", size_key="balanced")
        cancel = threading.Event()
        handle = AbortableHandle()

        class EarlyCancel(TargetEngine):
            def provision(self, *, tag, size_key, profile, region):
                cancel.set()
                return "i-0abc123456789def0"

        out = lj.run_launch(job, s, EarlyCancel(handle), cancel=cancel)
        assert out.status == lj.CANCELLED
        assert handle.aborted == 0

    def test_a_login_that_did_not_stop_is_recorded_on_the_job(self, tmp_path, caplog):
        """``cancel_device_login`` answers whether the cleanup ran; discarding that
        answer reported the one state the abort exists to prevent — a login still
        polling on a box the owner cancelled — as a clean teardown."""
        handle = FailingAbortHandle(url="https://x/?user_code=A", signed=False)
        with caplog.at_level("WARNING"):
            out, _ = self._cancel_during_signin(tmp_path, handle)
        assert handle.aborted == 1
        assert out.status == lj.CANCELLED
        assert "NOT confirmed stopped" in out.error
        assert any(
            r.levelname == "WARNING" and "not confirmed stopped" in r.getMessage()
            for r in caplog.records
        ), "a security-relevant cleanup that did not happen must not be logged at INFO"

    def test_an_abort_that_raises_is_recorded_too(self, tmp_path):
        handle = RaisingAbortHandle(url="https://x/?user_code=A", signed=False)
        out, _ = self._cancel_during_signin(tmp_path, handle)
        assert out.status == lj.CANCELLED, "a failed abort must not break the unwind"
        assert "NOT confirmed stopped" in out.error

    def test_a_stopped_login_adds_no_note(self, tmp_path):
        handle = AbortableHandle(url="https://x/?user_code=A", signed=False)
        out, _ = self._cancel_during_signin(tmp_path, handle)
        assert out.error == "", "a confirmed stop is not something to warn about"

    def test_a_handle_that_answers_nothing_is_not_a_confirmed_stop(self, tmp_path):
        """Only ``True`` confirms. A handle that HAS `abort` and returns ``None``
        has told us nothing, and nothing is not evidence the login died."""
        handle = LegacyAbortHandle(url="https://x/?user_code=A", signed=False)
        out, _ = self._cancel_during_signin(tmp_path, handle)
        assert handle.aborted == 1
        assert "NOT confirmed stopped" in out.error

    def test_a_handle_with_no_abort_is_not_a_confirmed_stop(self, tmp_path):
        """Absence of a stopper is not evidence the login stopped.

        A future engine's handle that drives a real remote login but forgets
        `abort` must NOT get a cancel that reports clean while the login keeps
        polling. The missing method raises, and the job carries the note.
        """
        handle = NoAbortHandle(url="", signed=False)
        out, _ = self._cancel_during_signin(tmp_path, handle)
        assert "NOT confirmed stopped" in out.error

    def test_the_fargate_lane_declares_there_is_nothing_to_stop(self, tmp_path) -> None:
        """The lane with no login says so explicitly, in code, not by omission.

        `FargateSigninHandle.abort()` returns True with the reason in its
        docstring, so a cancelled Fargate sign-in records no false note -- and
        `_abort_signin` never infers anything from a missing method.
        """
        from kiro_crew.cloud.fargate_engine import FargateSigninHandle

        h = FargateSigninHandle("arn:aws:ecs:x:1:task/y")
        assert h.abort() is True
        assert "no remote login" in (FargateSigninHandle.abort.__doc__ or "").lower()
        # Every in-tree handle now answers for itself.
        from kiro_crew.cloud import launch_engine as le_mod

        assert callable(getattr(le_mod._RealSigninHandle, "abort", None))

    def test_an_unstopped_login_survives_a_failed_rollback_message(self, tmp_path):
        """The worst case needs BOTH facts: the instance is still up AND its login
        is still polling. The rollback note must not erase the abort note."""
        handle = FailingAbortHandle(url="https://x/?user_code=A", signed=False)
        out, _ = self._cancel_during_signin(tmp_path, handle, teardown_confirms=False)
        assert "NOT confirmed stopped" in out.error
        assert "did NOT confirm" in out.error

    def test_a_completed_launch_leaves_its_login_alone(self, tmp_path):
        """An unconfirmed sign-in keeps its login on purpose: that is what makes the
        preserved code finishable from the dashboard."""
        s = _store(tmp_path)
        job = s.create(profile="dev", region="us-east-1", size_key="balanced")
        handle = AbortableHandle(url="https://x/?user_code=A", signed=False)
        lj.run_launch(job, s, TargetEngine(handle))
        assert handle.aborted == 0


# ── launch_job.py: run_signin_retry ───────────────────────────────────────────
class TestSigninRetry:
    def _registered_unsigned(self, tmp_path):
        s = _store(tmp_path)
        job = s.create(profile="dev", region="us-east-1", size_key="balanced", login_target=COMPANY)
        lj.run_launch(job, s, TargetEngine(FakeHandle(url="https://x/?user_code=A", signed=False)))
        return s, s.get(job.id)

    def test_a_new_code_is_published_with_the_stored_target(self, tmp_path):
        s, job = self._registered_unsigned(tmp_path)
        engine = TargetEngine(
            FakeHandle(url="https://x/?user_code=NEW-1", code="NEW-1", signed=False)
        )
        out = lj.run_signin_retry(job, s, engine)
        assert engine.seen == [COMPANY], "an Identity Center crew must not get a Builder ID code"
        assert out.signin is not None and out.signin.code == "NEW-1"
        assert out.status == lj.DONE
        assert out.signin_detected is False

    def test_the_old_code_is_dropped_before_the_new_one_is_asked_for(self, tmp_path):
        """Two codes on screen at once is a user typing the dead one."""
        s, job = self._registered_unsigned(tmp_path)
        assert job.signin is not None
        seen: list = []

        class Recording(TargetEngine):
            def begin_signin(self, *, instance_id, profile, region, login_target=None):
                seen.append(s.get(job.id).signin)
                return self.handle

        lj.run_signin_retry(job, s, Recording(FakeHandle(url="https://x/?user_code=NEW-1")))
        assert seen == [None]

    def test_a_confirmed_sign_in_clears_the_prompt(self, tmp_path):
        s, job = self._registered_unsigned(tmp_path)
        out = lj.run_signin_retry(
            job, s, TargetEngine(FakeHandle(url="https://x/?user_code=A", signed=True))
        )
        assert out.signin_detected is True
        assert out.signin is None
        assert out.step(lj.STEP_SIGNIN).state == lj.STEP_DONE

    def test_an_already_signed_in_box_needs_no_code(self, tmp_path):
        s, job = self._registered_unsigned(tmp_path)
        out = lj.run_signin_retry(job, s, TargetEngine(FakeHandle(already=True)))
        assert out.signin_detected is True
        assert out.signin is None

    def test_the_retry_never_re_runs_provisioning(self, tmp_path):
        """A COUNTER, not a raising double: ``run_signin_retry`` wraps the attempt in
        ``except Exception``, so a double that raised would be swallowed and this
        would pass with a second instance quietly billed."""
        s, job = self._registered_unsigned(tmp_path)
        engine = CountingEngine(FakeHandle(url="https://x/?user_code=A"))
        out = lj.run_signin_retry(job, s, engine)
        assert engine.provisions == 0, "a sign-in retry must not provision a second instance"
        assert engine.preflights == 0
        assert engine.registrations == 0
        assert engine.signins == 1
        assert out.step(lj.STEP_PROVISION).state == lj.STEP_DONE
        assert out.instance_id == "i-0abc123456789def0"

    def test_a_verified_identity_refusal_is_recorded_as_a_failure(self, tmp_path):
        """The box holds a session for a DIFFERENT identity: not a benign no-code."""
        s, job = self._registered_unsigned(tmp_path)
        out = lj.run_signin_retry(
            job, s, TargetEngine(FakeHandle(error="signed in to a different Kiro identity"))
        )
        assert out.step(lj.STEP_SIGNIN).state == lj.STEP_FAILED
        assert "different Kiro identity" in out.error
        assert out.signin_detected is False

    def test_a_failing_engine_leaves_the_crew_alone(self, tmp_path):
        s, job = self._registered_unsigned(tmp_path)

        class Broken(TargetEngine):
            def begin_signin(self, *, instance_id, profile, region, login_target=None):
                raise RuntimeError("ssm unreachable")

        out = lj.run_signin_retry(job, s, Broken())
        assert out.status == lj.DONE
        assert out.signin_detected is False
        assert "ssm unreachable" in out.step(lj.STEP_SIGNIN).detail
        assert out.step(lj.STEP_CONNECT).state == lj.STEP_DONE

    def test_a_cancelled_retry_drops_the_code(self, tmp_path):
        s, job = self._registered_unsigned(tmp_path)
        cancel = threading.Event()
        cancel.set()
        out = lj.run_signin_retry(
            job,
            s,
            TargetEngine(FakeHandle(url="https://x/?user_code=A", signed=False)),
            cancel=cancel,
        )
        assert out.status == lj.DONE
        assert out.signin is None

    def test_a_cancelled_retry_stops_the_remote_login(self, tmp_path):
        """The code is already in a browser; dropping it locally cancels nothing.

        A login left polling completes the sign-in minutes after the owner said
        no — the crew ends up signed in by a cancelled attempt.
        """
        s, job = self._registered_unsigned(tmp_path)
        handle = AbortableHandle(url="https://x/?user_code=A", signed=False)
        cancel = threading.Event()
        cancel.set()
        lj.run_signin_retry(job, s, TargetEngine(handle), cancel=cancel)
        assert handle.aborted == 1

    def test_a_cancelled_retry_says_so_when_the_login_did_not_stop(self, tmp_path):
        """The step detail IS this route's outcome, so "Sign-in cancelled." on a
        login that is still polling is the reassurance that hid the defect."""
        s, job = self._registered_unsigned(tmp_path)
        cancel = threading.Event()
        cancel.set()
        handle = FailingAbortHandle(url="https://x/?user_code=A", signed=False)
        out = lj.run_signin_retry(job, s, TargetEngine(handle), cancel=cancel)
        assert handle.aborted == 1
        assert "NOT confirmed stopped" in out.step(lj.STEP_SIGNIN).detail
        assert "NOT confirmed stopped" in out.error

    def test_a_cancelled_retry_that_did_stop_the_login_stays_quiet(self, tmp_path):
        s, job = self._registered_unsigned(tmp_path)
        cancel = threading.Event()
        cancel.set()
        out = lj.run_signin_retry(
            job,
            s,
            TargetEngine(AbortableHandle(url="https://x/?user_code=A", signed=False)),
            cancel=cancel,
        )
        assert out.step(lj.STEP_SIGNIN).detail == "Sign-in cancelled."
        assert out.error == ""

    def test_a_completed_retry_leaves_the_login_alone(self, tmp_path):
        """The unconfirmed path keeps its login on purpose: that is what makes the
        preserved code finishable from the dashboard."""
        s, job = self._registered_unsigned(tmp_path)
        for handle in (
            AbortableHandle(url="https://x/?user_code=A", signed=False),
            AbortableHandle(url="https://x/?user_code=A", signed=True),
            AbortableHandle(already=True),
        ):
            lj.run_signin_retry(s.get(job.id), s, TargetEngine(handle))
            assert handle.aborted == 0

    def test_a_handle_without_abort_is_recorded_as_unconfirmed(self, tmp_path):
        """A retry cancelled on a handle with no `abort` must not report clean:
        nothing stopped the login, so nothing may say it stopped."""
        s, job = self._registered_unsigned(tmp_path)
        handle = NoAbortHandle(url="https://x/?user_code=A", signed=False)
        cancel = threading.Event()
        cancel.set()
        out = lj.run_signin_retry(job, s, TargetEngine(handle), cancel=cancel)
        assert "NOT confirmed stopped" in out.error

    def test_the_real_handle_aborts_by_stopping_the_remote_login(self, monkeypatch):
        monkeypatch.setattr(
            le.login,
            "start_device_login",
            lambda *a, **k: SimpleNamespace(
                already_logged_in=False,
                url="https://x/?user_code=A-1",
                code="A-1",
                ports=[],
                error="",
            ),
        )
        seen: dict = {}

        def _cancel(iid, profile, region):
            seen.update(iid=iid, profile=profile, region=region)
            return True

        monkeypatch.setattr(le.login, "cancel_device_login", _cancel)
        handle = le.RealLaunchEngine().begin_signin(
            instance_id="i-0abc", profile="dev", region="us-east-1"
        )
        assert handle.abort() is True
        assert seen == {"iid": "i-0abc", "profile": "dev", "region": "us-east-1"}

    def test_the_real_handle_reports_a_login_it_could_not_stop(self, monkeypatch, caplog):
        """``cancel_device_login`` returning False means the remote cleanup did not
        run end to end — the login may still be polling, so the caller must hear
        it rather than read a clean cancel."""
        monkeypatch.setattr(
            le.login,
            "start_device_login",
            lambda *a, **k: SimpleNamespace(
                already_logged_in=False,
                url="https://x/?user_code=A-1",
                code="A-1",
                ports=[],
                error="",
            ),
        )
        monkeypatch.setattr(le.login, "cancel_device_login", lambda *a, **k: False)
        handle = le.RealLaunchEngine().begin_signin(
            instance_id="i-0abc", profile="dev", region="us-east-1"
        )
        with caplog.at_level("WARNING"):
            assert handle.abort() is False
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_the_real_handle_reports_an_abort_that_raised(self, monkeypatch, caplog):
        monkeypatch.setattr(
            le.login,
            "start_device_login",
            lambda *a, **k: SimpleNamespace(
                already_logged_in=False,
                url="https://x/?user_code=A-1",
                code="A-1",
                ports=[],
                error="",
            ),
        )

        def _boom(*_a, **_k):
            raise RuntimeError("ssm unreachable")

        monkeypatch.setattr(le.login, "cancel_device_login", _boom)
        handle = le.RealLaunchEngine().begin_signin(
            instance_id="i-0abc", profile="dev", region="us-east-1"
        )
        with caplog.at_level("WARNING"):
            assert handle.abort() is False
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_a_launch_with_no_instance_is_refused(self, tmp_path):
        s = _store(tmp_path)
        job = s.create(profile="dev", region="us-east-1", size_key="balanced")
        with pytest.raises(ValueError):
            lj.run_signin_retry(job, s, TargetEngine())

    def test_a_restart_during_a_retry_keeps_the_crew_visible(self, tmp_path):
        """The launch already registered the crew, so "the stack may still exist,
        check before retrying" would be wrong — and failing the job would hide a
        working instance behind a red card."""
        s, job = self._registered_unsigned(tmp_path)
        job.step(lj.STEP_SIGNIN).state = lj.STEP_ACTIVE
        job.status = lj.AWAITING_SIGNIN
        job.signin = lj.SigninPrompt(url="https://x/?user_code=A", code="A")
        s.save(job)
        fresh = lj.LaunchJobStore(root=s.root)  # a new process: owns nothing
        assert fresh.reap_orphans() == [job.id]
        out = fresh.get(job.id)
        assert out.status == lj.DONE
        assert out.instance_id
        assert out.signin is None
        assert out.signin_detected is False
        assert "sign-in" in out.error

    def test_a_restart_after_the_sign_in_confirmed_keeps_it(self, tmp_path):
        """``run_launch`` saves the connect step BEFORE the terminal status.

        A restart landing in that window must not clear ``signin_detected``: a
        crew that IS signed in would be badged "Needs sign-in", and its "get a new
        code" would sign the box out of the session it has.
        """
        s = _store(tmp_path)
        job = s.create(profile="dev", region="us-east-1", size_key="balanced")
        lj.run_launch(job, s, TargetEngine(FakeHandle(already=True)))
        mid = s.get(job.id)
        assert mid.signin_detected is True
        mid.status = lj.RUNNING  # the pre-terminal save the window exposes
        s.save(mid)

        fresh = lj.LaunchJobStore(root=s.root)
        assert fresh.reap_orphans() == [job.id]
        out = fresh.get(job.id)
        assert out.status == lj.DONE
        assert out.signin_detected is True, "a confirmed sign-in must survive the reap"
        assert out.error == "", "and it is not an interruption the user must act on"

    def test_a_restart_before_the_crew_exists_still_fails_the_job(self, tmp_path):
        s = _store(tmp_path)
        job = s.create(profile="dev", region="us-east-1", size_key="balanced")
        job.step(lj.STEP_PROVISION).state = lj.STEP_ACTIVE
        job.status = lj.RUNNING
        s.save(job)
        fresh = lj.LaunchJobStore(root=s.root)
        assert fresh.reap_orphans() == [job.id]
        assert fresh.get(job.id).status == lj.FAILED


# ── handlers_cloud.py: the restart route ─────────────────────────────────────
def _state(tmp_path, engine=None):
    return SimpleNamespace(
        owner_id="owner-1",
        cloud_launch_sync=True,
        cloud_launch_engine=engine or TargetEngine(),
        cloud_launch_store=lj.LaunchJobStore(root=tmp_path / "launch-jobs"),
    )


def _req(method, path, *, state, body=None, match_info=None):
    app = web.Application()
    app["state"] = state
    req = make_mocked_request(method, path, app=app, match_info=match_info or {})
    req["user"] = "owner-1"
    req["app"] = ""
    if body is not None:

        async def _json():
            return body

        req.json = _json  # type: ignore[assignment]
    return req


def _read(resp):
    return json.loads(resp.body.decode("utf-8"))


BASE_BODY = {"profile": "dev", "region": "us-east-1", "size_key": "balanced"}


@pytest.fixture(autouse=True)
def _posix_host(monkeypatch):
    """These routes are POSIX-only by design; pin the platform so the host cannot
    turn a success-path assertion into a 400 that is nothing to do with the code."""
    monkeypatch.setattr(hc.sys, "platform", "linux")


@pytest.mark.asyncio
class TestSigninRestartRoute:
    async def _unsigned_job(self, tmp_path):
        state = _state(
            tmp_path, TargetEngine(FakeHandle(url="https://x/?user_code=A", signed=False))
        )
        resp = await hc.api_cloud_launch_create(
            _req("POST", "/api/cloud/launch", state=state, body=dict(BASE_BODY))
        )
        return state, _read(resp)["id"]

    async def test_it_starts_a_new_sign_in(self, tmp_path):
        state, job_id = await self._unsigned_job(tmp_path)
        state.cloud_launch_engine = TargetEngine(
            FakeHandle(url="https://x/?user_code=NEW-1", code="NEW-1", signed=True)
        )
        resp = await hc.api_cloud_launch_signin_restart(
            _req("POST", "/x", state=state, match_info={"id": job_id})
        )
        assert resp.status == 202
        assert state.cloud_launch_store.get(job_id).signin_detected is True

    async def test_the_route_never_re_provisions(self, tmp_path):
        state, job_id = await self._unsigned_job(tmp_path)
        engine = CountingEngine(FakeHandle(url="https://x/?user_code=B", signed=True))
        state.cloud_launch_engine = engine
        resp = await hc.api_cloud_launch_signin_restart(
            _req("POST", "/x", state=state, match_info={"id": job_id})
        )
        assert resp.status == 202
        assert engine.provisions == 0, "the restart route must not provision a second instance"
        assert engine.preflights == 0
        assert engine.signins == 1
        assert state.cloud_launch_store.get(job_id).instance_id == "i-0abc123456789def0"

    async def test_an_unknown_job_is_404(self, tmp_path):
        state, _ = await self._unsigned_job(tmp_path)
        resp = await hc.api_cloud_launch_signin_restart(
            _req("POST", "/x", state=state, match_info={"id": "b" * 12})
        )
        assert resp.status == 404

    async def test_a_launch_that_never_created_a_crew_is_400(self, tmp_path):
        state = _state(tmp_path)
        job = state.cloud_launch_store.create(
            profile="dev", region="us-east-1", size_key="balanced"
        )
        resp = await hc.api_cloud_launch_signin_restart(
            _req("POST", "/x", state=state, match_info={"id": job.id})
        )
        assert resp.status == 400
        assert _read(resp)["code"] == "launch_has_no_instance"

    async def test_the_job_is_claimed_before_the_worker_starts(self, tmp_path, monkeypatch):
        """The guard is only a guard if the state it reads is already written.

        The worker persists RUNNING itself, but it runs AFTER the lock is
        released — so a second restart arriving in that window read a terminal
        job, passed the "nothing active" check, and raced a second remote login
        with one cancel handle between them. Observed at the moment the worker is
        started, which is that window's first instant.
        """
        state, job_id = await self._unsigned_job(tmp_path)
        store = state.cloud_launch_store
        observed: list = []

        def fake_start(st, jb, eng):
            fresh = store.get(jb.id)
            observed.append((fresh.status, fresh.step(lj.STEP_SIGNIN).state))

        monkeypatch.setattr(hc, "_start_signin_worker", fake_start)
        resp = await hc.api_cloud_launch_signin_restart(
            _req("POST", "/x", state=state, match_info={"id": job_id})
        )
        assert resp.status == 202
        assert observed == [(lj.RUNNING, lj.STEP_ACTIVE)]

    async def test_a_second_restart_is_refused_while_the_first_runs(self, tmp_path):
        state, job_id = await self._unsigned_job(tmp_path)
        job = state.cloud_launch_store.get(job_id)
        hc._claim_signin(state, job)  # the claim the route makes under its lock
        resp = await hc.api_cloud_launch_signin_restart(
            _req("POST", "/x", state=state, match_info={"id": job_id})
        )
        assert resp.status == 409
        assert _read(resp)["code"] == "launch_already_running"

    async def test_it_refuses_while_another_launch_runs(self, tmp_path):
        """One live device code at a time, or the user types the wrong one."""
        state, job_id = await self._unsigned_job(tmp_path)
        other = state.cloud_launch_store.create(
            profile="dev", region="us-east-1", size_key="balanced"
        )
        other.status = lj.RUNNING
        state.cloud_launch_store.save(other)
        resp = await hc.api_cloud_launch_signin_restart(
            _req("POST", "/x", state=state, match_info={"id": job_id})
        )
        assert resp.status == 409
        assert _read(resp)["code"] == "launch_already_running"

    def test_the_sync_starter_does_no_store_io_when_the_thread_fails(self, tmp_path, monkeypatch):
        """`_start_signin_worker` is sync and is called from the event loop. Its
        failure path must not read or write the store -- that revert belongs in
        `_unclaim_signin`, which the async caller runs in an executor."""
        state = _state(tmp_path, TargetEngine(FakeHandle(url="", code="", signed=False)))
        state.cloud_launch_sync = False
        store = state.cloud_launch_store
        job = store.create(profile="dev", region="us-east-1", size_key="balanced")

        touched: list[str] = []
        monkeypatch.setattr(store, "get", lambda *a, **k: touched.append("get"))
        monkeypatch.setattr(store, "save", lambda *a, **k: touched.append("save"))

        class _NoThread:
            def __init__(self, *a, **k): ...

            def start(self):
                raise RuntimeError("can't start new thread")

        monkeypatch.setattr(hc.threading, "Thread", _NoThread)
        with pytest.raises(RuntimeError):
            hc._start_signin_worker(state, job, state.cloud_launch_engine)
        assert touched == [], f"store I/O on the event loop path: {touched}"

    async def test_a_failed_thread_start_does_not_wedge_launches(self, tmp_path, monkeypatch):
        """`_claim_signin` persists RUNNING before the worker thread exists. If
        `Thread.start` then raises, nothing would ever move the job off RUNNING --
        every later launch and restart answers 409 forever. The claim must be
        undone and the caller told 503, not 202."""
        state, job_id = await self._unsigned_job(tmp_path)
        state.cloud_launch_sync = False  # take the real thread path
        state.cloud_launch_engine = TargetEngine(
            FakeHandle(url="https://x/?user_code=N", code="N", signed=False)
        )

        class _NoThread:
            def __init__(self, *a, **k): ...

            def start(self):
                raise RuntimeError("can't start new thread")

        monkeypatch.setattr(hc.threading, "Thread", _NoThread)
        resp = await hc.api_cloud_launch_signin_restart(
            _req("POST", "/x", state=state, match_info={"id": job_id})
        )
        assert resp.status == 503
        assert _read(resp)["code"] == "signin_worker_unavailable"

        out = state.cloud_launch_store.get(job_id)
        assert out.status != lj.RUNNING, "the claim was left in place: launches are wedged"
        assert out.terminal
        assert "out of threads" in out.error

        # And a second attempt is admitted, not refused as launch_already_running.
        monkeypatch.undo()
        state.cloud_launch_sync = True
        resp2 = await hc.api_cloud_launch_signin_restart(
            _req("POST", "/x", state=state, match_info={"id": job_id})
        )
        assert resp2.status == 202

    async def test_it_is_owner_only(self, tmp_path):
        state, job_id = await self._unsigned_job(tmp_path)
        app = web.Application()
        app["state"] = state
        req = make_mocked_request(
            "POST", "/x", headers={"X-Session-Key": "slack:x"}, app=app, match_info={"id": job_id}
        )
        req["user"] = "owner-1"
        req["app"] = ""
        resp = await hc.api_cloud_launch_signin_restart(req)
        assert resp.status == 403

    async def test_the_route_is_registered(self):
        import inspect

        from kiro_crew.dashboard.routes import connections

        src = inspect.getsource(connections)
        assert "/api/cloud/launch/{id}/signin/restart" in src
        assert "api_cloud_launch_signin_restart" in src


@pytest.mark.asyncio
class TestSigninPromptReprobe:
    """A preserved code approved out of band must not badge the crew forever.

    A timed-out sign-in deliberately KEEPS its code and leaves the remote login
    polling, so the user can still approve it from the open browser tab. Nothing
    re-probed the box after the job went terminal, so an approval that landed
    signed the crew in while ``signin_detected`` stayed False — "Needs sign-in"
    forever, Connect held.
    """

    async def _preserved(self, tmp_path):
        """A terminal, registered, unsigned job still holding its device code."""
        state = _state(
            tmp_path, TargetEngine(FakeHandle(url="https://x/?user_code=A", code="A", signed=False))
        )
        resp = await hc.api_cloud_launch_create(
            _req("POST", "/api/cloud/launch", state=state, body=dict(BASE_BODY))
        )
        job = state.cloud_launch_store.get(_read(resp)["id"])
        assert job.terminal and job.signin is not None and job.signin_detected is False
        return state, job

    async def _fetch(self, state, job_id):
        return await hc.api_cloud_launch_signin(
            _req("POST", "/x", state=state, match_info={"id": job_id})
        )

    async def test_the_box_probe_itself_writes_nothing(self, tmp_path, monkeypatch):
        """The SSM query runs in an executor thread; the write must not.

        This is the property that makes the lock meaningful. While the mutation
        lived beside the query on the worker thread, nothing serialised it against
        a restart persisting RUNNING under `_launch_lock` -- so the write moved to
        `_record_probed_signin`, on the loop, under that lock. If a future change
        puts a `save` back into the query path, this fails.
        """
        state, job = await self._preserved(tmp_path)
        store = state.cloud_launch_store
        before = json.dumps(store.get(job.id).to_dict(), sort_keys=True)

        monkeypatch.setattr(hc.login_mod, "is_logged_in", lambda *a, **k: True)
        assert hc._probe_signin_on_box(state, job) is True

        after = json.dumps(store.get(job.id).to_dict(), sort_keys=True)
        assert after == before, "the probe must query only; the write is the caller's"

    async def test_the_recorded_write_is_a_coroutine_taking_the_lock(self) -> None:
        """`_record_probed_signin` has to be awaitable to take the async lock.

        A sync helper cannot `async with _launch_lock(...)`, which is exactly why
        the first attempt at this fix settled for a bare re-read.
        """
        import inspect

        assert inspect.iscoroutinefunction(hc._record_probed_signin)
        src = inspect.getsource(hc._record_probed_signin)
        assert "_launch_lock" in src, "the write must be serialised with _claim_signin"

    async def test_a_probe_result_is_dropped_when_the_job_moved_on(self, tmp_path, monkeypatch):
        """The probe holds a snapshot from BEFORE an SSM round trip of seconds.

        If a restart was admitted in that window it has already persisted RUNNING
        under the launch lock -- and that claim is the only thing stopping two
        remote logins racing one cancel handle. Writing this stale terminal
        snapshot over it would undo exactly that. The probe must re-read and drop
        its own result.
        """
        state, job = await self._preserved(tmp_path)
        store = state.cloud_launch_store

        def _logged_in(iid, profile, region, *, target=None):
            # Simulate the restart landing DURING the probe: by the time the box
            # answers, the job on disk is RUNNING with the sign-in step active.
            moved = store.get(job.id)
            moved.status = lj.RUNNING
            moved.step(lj.STEP_SIGNIN).state = lj.STEP_ACTIVE
            moved.signin = None
            moved.signin_detected = False
            store.save(moved)
            return True

        monkeypatch.setattr(hc.login_mod, "is_logged_in", _logged_in)
        await self._fetch(state, job.id)

        out = store.get(job.id)
        assert out.status == lj.RUNNING, "the restart's claim was overwritten"
        assert out.step(lj.STEP_SIGNIN).state == lj.STEP_ACTIVE
        assert out.signin_detected is False, "the stale probe result was written over a newer claim"

    async def test_an_approved_preserved_code_clears_the_unsigned_state(
        self, tmp_path, monkeypatch
    ):
        state, job = await self._preserved(tmp_path)
        probed: list = []

        def _logged_in(iid, profile, region, *, target=None):
            probed.append((iid, target))
            return True

        monkeypatch.setattr(hc.login_mod, "is_logged_in", _logged_in)
        resp = await self._fetch(state, job.id)
        assert resp.status == 409
        assert _read(resp)["code"] == "signin_already_complete"
        out = state.cloud_launch_store.get(job.id)
        assert out.signin_detected is True, "the badge would never clear"
        assert out.signin is None, "the code has been used"
        assert out.step(lj.STEP_SIGNIN).state == lj.STEP_DONE
        assert probed and probed[0][0] == out.instance_id

    async def test_the_probe_asks_about_the_jobs_own_identity(self, tmp_path, monkeypatch):
        """A session for a DIFFERENT identity is not this crew being signed in."""
        state = _state(
            tmp_path, TargetEngine(FakeHandle(url="https://x/?user_code=A", code="A", signed=False))
        )
        resp = await hc.api_cloud_launch_create(
            _req(
                "POST",
                "/api/cloud/launch",
                state=state,
                body=dict(
                    BASE_BODY,
                    login_target={
                        "license": "pro",
                        "start_url": "https://amzn.awsapps.com/start",
                        "region": "us-east-1",
                    },
                ),
            )
        )
        job = state.cloud_launch_store.get(_read(resp)["id"])
        seen: list = []
        monkeypatch.setattr(
            hc.login_mod,
            "is_logged_in",
            lambda iid, profile, region, *, target=None: seen.append(target) or False,
        )
        await self._fetch(state, job.id)
        assert seen == [COMPANY]

    async def test_a_box_that_is_still_unsigned_is_left_alone(self, tmp_path, monkeypatch):
        state, job = await self._preserved(tmp_path)
        monkeypatch.setattr(hc.login_mod, "is_logged_in", lambda *a, **k: False)
        resp = await self._fetch(state, job.id)
        assert resp.status == 409
        assert _read(resp)["code"] == "no_signin_pending"
        out = state.cloud_launch_store.get(job.id)
        assert out.signin_detected is False
        assert out.signin is not None, "the preserved code is still finishable"

    async def test_a_failing_probe_changes_nothing(self, tmp_path, monkeypatch):
        state, job = await self._preserved(tmp_path)

        def _boom(*_a, **_k):
            raise RuntimeError("ssm unreachable")

        monkeypatch.setattr(hc.login_mod, "is_logged_in", _boom)
        resp = await self._fetch(state, job.id)
        assert resp.status == 409
        assert state.cloud_launch_store.get(job.id).signin_detected is False

    async def test_a_non_terminal_job_is_never_probed(self, tmp_path, monkeypatch):
        """A COUNTER, not a raising double: the route must not reach AWS for a job a
        worker is still driving, and a raising probe would be indistinguishable
        from one that was never called."""
        state, job = await self._preserved(tmp_path)
        job.status = lj.RUNNING
        job.step(lj.STEP_SIGNIN).state = lj.STEP_ACTIVE
        state.cloud_launch_store.save(job)
        calls: list = []
        monkeypatch.setattr(
            hc.login_mod,
            "is_logged_in",
            lambda *a, **k: calls.append(1) or True,
        )
        resp = await self._fetch(state, job.id)
        assert resp.status == 409
        assert calls == [], "a job a worker owns must not be probed or rewritten"
        assert state.cloud_launch_store.get(job.id).signin_detected is False

    async def test_a_pending_prompt_is_returned_without_any_probe(self, tmp_path, monkeypatch):
        """The normal path: AWAITING_SIGNIN with a live code still just answers it."""
        state, job = await self._preserved(tmp_path)
        job.status = lj.AWAITING_SIGNIN
        state.cloud_launch_store.save(job)
        calls: list = []
        monkeypatch.setattr(hc.login_mod, "is_logged_in", lambda *a, **k: calls.append(1) or True)
        resp = await self._fetch(state, job.id)
        assert resp.status == 200
        assert _read(resp)["signin"]["code"] == "A"
        assert calls == []

    async def test_a_job_with_no_preserved_code_is_not_probed(self, tmp_path, monkeypatch):
        """Nothing is pending approval, so there is nothing that could have been
        approved — probing every terminal job would be the blanket poll this
        deliberately is not."""
        state, job = await self._preserved(tmp_path)
        job.signin = None
        state.cloud_launch_store.save(job)
        calls: list = []
        monkeypatch.setattr(hc.login_mod, "is_logged_in", lambda *a, **k: calls.append(1) or True)
        resp = await self._fetch(state, job.id)
        assert resp.status == 409
        assert _read(resp)["code"] == "no_signin_pending"
        assert calls == []

    async def test_an_already_signed_job_is_not_probed(self, tmp_path, monkeypatch):
        state, job = await self._preserved(tmp_path)
        job.signin_detected = True
        state.cloud_launch_store.save(job)
        calls: list = []
        monkeypatch.setattr(hc.login_mod, "is_logged_in", lambda *a, **k: calls.append(1) or True)
        await self._fetch(state, job.id)
        assert calls == []

    async def test_it_is_owner_only(self, tmp_path):
        state, job = await self._preserved(tmp_path)
        app = web.Application()
        app["state"] = state
        req = make_mocked_request(
            "POST", "/x", headers={"X-Session-Key": "slack:x"}, app=app, match_info={"id": job.id}
        )
        req["user"] = "owner-1"
        req["app"] = ""
        resp = await hc.api_cloud_launch_signin(req)
        assert resp.status == 403


def test_ssm_module_is_untouched_by_these_tests():
    """Guards against a stray monkeypatch leaking into the shared module."""
    assert callable(ssm.run_command)


class TestKillPatternMatchesWhatTheBoxLaunches:
    """The `pkill` pattern is only ever exercised on the remote box.

    The design lane's objection is exact: a pattern that matches nothing exits 1
    from `pkill` just as a successful kill does, so an assertion about our own
    command text proves nothing about whether the box's login would be found.

    Two guards stand in for a live run. This class is the static one -- it ties
    our killer to main's launcher, so a rename on EITHER side fails here instead
    of silently matching nothing over SSM. The runtime one is the `pgrep`
    re-probe, which reports UNCONFIRMED rather than a clean stop.
    """

    def test_the_pattern_is_a_substring_of_the_launched_command_line(self) -> None:
        # The real builder, with the identity flags a company-SSO launch adds --
        # the pattern must survive them being appended.
        cmd = login._device_login_command(
            replace_existing=False,
            identity_provider="https://amzn.awsapps.com/start",
            license_="pro",
            idp_region="us-east-1",
        )
        launch_lines = [ln for ln in cmd.splitlines() if "login --use-device-flow" in ln]
        assert launch_lines, f"no launch line found in:\n{cmd}"
        for ln in launch_lines:
            # `$KIRO` resolves to a path whose basename is `kiro-cli`, so the
            # rendered command line contains our fragment. Substituting the
            # resolved binary is what a real box does before exec.
            rendered = ln.replace('"$KIRO"', "/usr/local/bin/kiro-cli")
            assert login._LOGIN_PROCESS_PATTERN in rendered, (
                f"pkill pattern {login._LOGIN_PROCESS_PATTERN!r} would not match "
                f"the launched command line {rendered!r}"
            )

    def test_the_pattern_survives_a_stdbuf_wrapper(self) -> None:
        # main wraps the login in `stdbuf` when it is available; the wrapper's own
        # command line must still contain the fragment, or the wrapper outlives
        # the kill and keeps the pipe open.
        cmd = login._device_login_command(replace_existing=False)
        stdbuf_lines = [
            ln for ln in cmd.splitlines() if "stdbuf" in ln and "login --use-device-flow" in ln
        ]
        assert stdbuf_lines, "expected a stdbuf-wrapped launch line"
        for ln in stdbuf_lines:
            rendered = ln.replace('"$KIRO"', "/usr/local/bin/kiro-cli")
            assert login._LOGIN_PROCESS_PATTERN in rendered

    def test_the_cancel_reprobes_and_reports_a_login_that_outlived_the_kill(self) -> None:
        cmd = login._cancel_login_command()
        # The AVAILABILITY CHECK, not the bare word "pgrep": a disabled branch
        # (`if false; then pgrep ...`) still contains the word, so grepping for it
        # passes while the re-probe never runs.
        assert "command -v pgrep" in cmd, "the cancel must confirm the kill, not assume it"
        assert f'pgrep -u "$(id -u)" -f "{login._LOGIN_PROCESS_PATTERN}"' in cmd
        assert login._CANCEL_UNCONFIRMED_SENTINEL in cmd
        # The re-probe has to come AFTER the kill, or it confirms nothing.
        assert cmd.index("pkill") < cmd.index("command -v pgrep")

    def test_a_login_still_running_is_reported_as_not_stopped(self, monkeypatch, caplog):
        calls: list[str] = []

        def fake_run(instance_id, command, *_a, **_k):
            calls.append(command)
            return ssm.CommandResult("Failed", login._CANCEL_UNCONFIRMED_SENTINEL, "", 1)

        monkeypatch.setattr(ssm, "run_command", fake_run)
        with caplog.at_level("WARNING"):
            assert login.cancel_device_login("i-abc") is False
        assert calls, "the cancel must reach the box"
        # Named, because this is the dangerous outcome: the login outlived the kill.
        assert "still running after the cancel" in caplog.text

    def test_a_box_without_pkill_is_told_apart_from_a_surviving_login(self, monkeypatch, caplog):
        # Both answer False, but the operator's next step differs: install pkill,
        # versus find out why the pattern stopped matching what the box launched.
        def fake_run(instance_id, command, *_a, **_k):
            return ssm.CommandResult("Failed", login._CANCEL_NO_PKILL_SENTINEL, "", 1)

        monkeypatch.setattr(ssm, "run_command", fake_run)
        with caplog.at_level("WARNING"):
            assert login.cancel_device_login("i-abc") is False
        assert "has no pkill" in caplog.text
        assert "still running after the cancel" not in caplog.text

    def test_a_sentinel_on_stderr_is_read_too(self, monkeypatch, caplog):
        # A shell that dies mid-script can leave its last line on stderr.
        #
        # Asserting only the return value proves nothing here: `cancel_device_login`
        # ends in a broad `status != "Success"` arm that answers False as well, so a
        # version reading only stdout still "passes". The DISTINGUISHING evidence is
        # which warning it logged -- the UNCONFIRMED one is reachable only by having
        # actually read stderr.
        def fake_run(instance_id, command, *_a, **_k):
            return ssm.CommandResult("Failed", "", login._CANCEL_UNCONFIRMED_SENTINEL, 1)

        monkeypatch.setattr(ssm, "run_command", fake_run)
        with caplog.at_level("WARNING"):
            assert login.cancel_device_login("i-abc") is False
        assert "still running after the cancel" in caplog.text
        assert "did not complete" not in caplog.text
