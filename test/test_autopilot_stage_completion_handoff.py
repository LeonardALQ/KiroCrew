"""Autopilot must consume a stage's completion reports before advancing.

A stage turn can dispatch background agents and end while their terminal reports
are still entering the parent conversation.  The active LLM turn remains in
``slot.task`` so completion delivery waits for that turn, while the outer stage
controller waits for every terminal report before advancing.

These tests make that ordering deterministic: the agent is already done, its
terminal report is still pending, and Stage 2 may start only after that report
has queued and its completion turn has finished.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from chat_test_helpers import _make_state

from kiro_crew.dashboard.chat_utils import SUBAGENT_COMPLETION_KIND
from kiro_crew.subagent import SubagentManager


@pytest.fixture(autouse=True)
def _isolate_config_dir(tmp_path, monkeypatch):
    """Keep stage result files inside the test's temporary directory."""
    for module in ("state", "chat", "chat_orchestrator"):
        monkeypatch.setattr(f"kiro_crew.dashboard.{module}.config_dir", lambda: tmp_path)


class _StageDeliveryManager:
    """Minimal manager double with one parent report still in flight."""

    def __init__(self) -> None:
        self.report_task: asyncio.Task | None = None

    def running_agents_for(self, _parent: str) -> list[dict]:
        return []

    async def wait_for_parent_reports(self, _parent: str) -> None:
        if self.report_task is not None:
            await asyncio.shield(self.report_task)


@pytest.mark.asyncio
async def test_completion_turn_finishes_before_next_stage_starts(tmp_path, monkeypatch):
    """A done agent with an undelivered report must hold the stage boundary."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    manager = _StageDeliveryManager()
    state.subagents = manager
    slot = state.get_or_create_slot("completion-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True

    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **_kwargs):
        if message.startswith("[Subagent completion event]"):
            order.append("completion")
            _slot.append("assistant", "completion synthesized", "msg msg-a")
            return
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            _slot.append("assistant", "agents dispatched", "msg msg-a")

            async def _finish_report() -> None:
                await asyncio.sleep(0)
                _slot.queue_append(
                    "[Subagent completion event]\nall agents complete",
                    kind=SUBAGENT_COMPLETION_KIND,
                )

            manager.report_task = asyncio.create_task(_finish_report())
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            _slot.append("assistant", "verified", "msg msg-a")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    async def _start_system_turn(_state, _slot):
        index = next(
            (
                i
                for i, entry in enumerate(_slot._queue)
                if entry.get("kind") == SUBAGENT_COMPLETION_KIND
            ),
            None,
        )
        if index is None:
            return False
        entry = _slot._queue.pop(index)
        task = asyncio.create_task(_mock_run_chat(_state, _slot, entry["content"]))
        _slot.task = task
        return True

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._start_next_queued_turn",
        _start_system_turn,
    )

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)
    if slot.task is not None and slot.task is not asyncio.current_task():
        await asyncio.wait_for(slot.task, timeout=5)

    assert order == ["stage-1", "completion", "stage-2"]


@pytest.mark.asyncio
async def test_manager_waits_for_parent_terminal_reports():
    """The manager's parent-scoped wait follows report tasks, not ``info.done``."""
    manager = object.__new__(SubagentManager)
    release = asyncio.Event()
    entered = asyncio.Event()

    async def _report() -> None:
        entered.set()
        await release.wait()

    report = asyncio.create_task(_report())
    owner = SimpleNamespace(parent_session_key="dashboard:parent")
    manager._report_tasks = {report}
    manager._report_owners = {report: owner}

    def _forget(task: asyncio.Task) -> None:
        manager._report_tasks.discard(task)
        manager._report_owners.pop(task, None)

    report.add_done_callback(_forget)
    waiter = asyncio.create_task(manager.wait_for_parent_reports("dashboard:parent"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not waiter.done(), "the parent wait returned before terminal delivery finished"

    release.set()
    await asyncio.wait_for(waiter, timeout=1)


@pytest.mark.asyncio
async def test_live_stage_controller_counts_as_running_between_stages(tmp_path):
    """The slot stays busy while its outer controller owns the stage boundary."""
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("running-handoff", mode="orchestrator")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _controller() -> None:
        entered.set()
        await release.wait()

    controller = asyncio.create_task(_controller())
    slot.track_stage_controller(controller)
    await asyncio.wait_for(entered.wait(), timeout=1)
    slot.task = None

    try:
        assert slot.running is True
    finally:
        release.set()
        await asyncio.wait_for(controller, timeout=1)


class _KeyedStageDeliveryManager:
    """Record every parent key the stage controller asks about."""

    def __init__(self) -> None:
        self.running_keys: list[str] = []
        self.report_keys: list[str] = []

    def running_agents_for(self, parent: str) -> list[dict]:
        self.running_keys.append(parent)
        return []

    async def wait_for_parent_reports(self, parent: str) -> bool:
        self.report_keys.append(parent)
        return False


@pytest.mark.asyncio
async def test_channel_linked_stage_uses_effective_parent_key(tmp_path, monkeypatch):
    """Channel-born children and terminal reports live under the channel key."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    manager = _KeyedStageDeliveryManager()
    state.subagents = manager
    slot = state.get_or_create_slot("linked-handoff", mode="orchestrator")
    slot.linked_session_key = "slack:C123:1700000000.000001"
    slot._stage_titles = ["Only"]
    slot._plan_goal = "Use the linked session"
    slot._auto_run = True

    async def _mock_run_chat(_state, _slot, _message, **_kwargs):
        _slot.append("assistant", "done", "msg msg-a")

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    expected = "slack:C123:1700000000.000001"
    assert expected in manager.running_keys
    assert manager.report_keys
    assert set(manager.report_keys) == {expected}


@pytest.mark.asyncio
async def test_auth_refusal_restores_completion_and_pauses(tmp_path, monkeypatch):
    """A signed-out completion resumes its stage boundary before Stage 2."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    manager = _StageDeliveryManager()
    state.subagents = manager
    slot = state.get_or_create_slot("auth-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    order: list[str] = []
    refuse_auth = True

    async def _mock_run_chat(_state, _slot, message, **_kwargs):
        if message.startswith("[Subagent completion event]"):
            order.append("completion")
            _slot.append("assistant", "completion synthesized", "msg msg-a")
            return
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            _slot.append("assistant", "agents dispatched", "msg msg-a")

            async def _finish_report() -> None:
                await asyncio.sleep(0)
                _slot.queue_append(
                    "[Subagent completion event]\nall agents complete",
                    kind=SUBAGENT_COMPLETION_KIND,
                )

            manager.report_task = asyncio.create_task(_finish_report())
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected direct turn: {message[:80]}")

    async def _start_completion_turn(_state, _slot):
        index = next(
            i
            for i, entry in enumerate(_slot._queue)
            if entry.get("kind") == SUBAGENT_COMPLETION_KIND
        )
        entry = _slot._queue.pop(index)

        async def _completion_turn() -> None:
            if refuse_auth:
                _slot._last_turn_auth_required = True
                return
            await _mock_run_chat(_state, _slot, entry["content"])

        task = asyncio.create_task(_completion_turn())
        _slot.task = task
        return True

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._start_next_queued_turn",
        _start_completion_turn,
    )

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    assert order == ["stage-1"]
    assert any(
        entry.get("kind") == SUBAGENT_COMPLETION_KIND for entry in slot._queue
    ), "the completion must remain queued for post-login retry"
    text = "\n".join(
        str(message.get("content", ""))
        for message in slot.messages
        if message.get("role") == "assistant"
    )
    assert "completion event could not be processed" in text

    refuse_auth = False
    slot._last_turn_auth_required = False
    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    assert order == ["stage-1", "completion", "stage-2"]
    assert not any(entry.get("kind") == SUBAGENT_COMPLETION_KIND for entry in slot._queue)


@pytest.mark.asyncio
async def test_synthetic_recovery_finishes_before_next_stage(tmp_path, monkeypatch):
    """A recovery turn started by Stage 1 must finish before Stage 2."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("recovery-handoff", mode="orchestrator")
    slot._stage_titles = ["Recover", "Verify"]
    slot._plan_goal = "Recover then verify"
    slot._auto_run = True
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **_kwargs):
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")

            async def _recovery() -> None:
                order.append("recovery-start")
                await asyncio.sleep(0)
                order.append("recovery-end")
                _slot._synthetic_recovery_inflight -= 1

            _slot._synthetic_recovery_inflight += 1
            _slot.task = asyncio.create_task(_recovery())
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    assert order == ["stage-1", "recovery-start", "recovery-end", "stage-2"]


@pytest.mark.asyncio
async def test_closing_slot_cancels_controller_before_next_stage(tmp_path, monkeypatch):
    """Closing an active Autopilot slot prevents every remaining stage."""
    from kiro_crew.dashboard.chat_handlers import close_slot
    from kiro_crew.dashboard.chat_orchestrator import api_chat_plan_action

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("close-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    stage_one_started = asyncio.Event()
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **_kwargs):
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            stage_one_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # _run_chat owns cancellation cleanup and returns normally.
                return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    async def _retire_nudge(_slot_key):
        return None

    async def _save_slot(*_args, **_kwargs):
        return None

    class _PlanRequest:
        app = {"state": state}
        match_info = {"slot": slot.key}

        async def json(self):
            return {"action": "go all"}

        def get(self, _key, default=""):
            return default

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers._retire_slot_nudge_loop", _retire_nudge)
    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.save_slot_off_loop", _save_slot)

    response = await api_chat_plan_action(_PlanRequest())
    assert response.status == 200
    controller = slot.task
    assert controller is not None
    await asyncio.wait_for(stage_one_started.wait(), timeout=1)

    await close_slot(state, slot, slot.key)
    if not controller.done():
        await asyncio.wait_for(asyncio.shield(controller), timeout=1)

    assert controller.done()
    assert order == ["stage-1"]


@pytest.mark.asyncio
async def test_stop_between_stages_cancels_controller_before_next_stage(tmp_path, monkeypatch):
    """Session Stop cancels the live controller even when no child turn owns the slot."""
    from kiro_crew.dashboard.chat_handlers import stop_slot_turn
    from kiro_crew.dashboard.chat_orchestrator import api_chat_plan_action

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("stop-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    boundary_entered = asyncio.Event()
    release_boundary = asyncio.Event()
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **_kwargs):
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            _slot.append("assistant", "collected", "msg msg-a")
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    async def _hold_stage_boundary(_state, _slot, _tracker, stage_num, **_kwargs):
        if stage_num == 1:
            boundary_entered.set()
            await release_boundary.wait()
        return True

    async def _stop_turn(*_args, **_kwargs):
        return "idle"

    class _PlanRequest:
        app = {"state": state}
        match_info = {"slot": slot.key}

        async def json(self):
            return {"action": "go all"}

        def get(self, _key, default=""):
            return default

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._settle_stage_delivery",
        _hold_stage_boundary,
    )
    monkeypatch.setattr(state.sessions, "stop_turn", _stop_turn)

    response = await api_chat_plan_action(_PlanRequest())
    assert response.status == 200
    controller = slot._stage_controller_task
    assert controller is not None
    await asyncio.wait_for(boundary_entered.wait(), timeout=1)
    assert slot.task is None

    result = await stop_slot_turn(state, slot, source="session_control")
    release_boundary.set()
    await asyncio.gather(controller, return_exceptions=True)

    assert result["ok"] is True
    assert slot._auto_run is False
    assert controller.done()
    assert order == ["stage-1"]


@pytest.mark.asyncio
async def test_live_completion_turn_holds_stage_boundary(tmp_path, monkeypatch):
    """A live completion turn must finish before the next stage starts."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("live-completion-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    completion_started = asyncio.Event()
    stage_two_started = asyncio.Event()
    release_completion = asyncio.Event()
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **_kwargs):
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")

            async def _completion_turn() -> None:
                order.append("completion-start")
                completion_started.set()
                await release_completion.wait()
                order.append("completion-end")

            _slot.task = asyncio.create_task(_completion_turn())
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            stage_two_started.set()
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)

    controller = asyncio.create_task(_stage_loop(state, slot, auto_run=True))
    await asyncio.wait_for(completion_started.wait(), timeout=1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stage_two_started.wait(), timeout=0.1)

    release_completion.set()
    await asyncio.wait_for(controller, timeout=5)

    assert order == ["stage-1", "completion-start", "completion-end", "stage-2"]


@pytest.mark.asyncio
async def test_final_controller_await_does_not_strand_a_late_queue_entry(tmp_path, monkeypatch):
    """A message queued during the final payload await must start before release."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("late-queue-handoff", mode="orchestrator")
    slot._stage_titles = ["Only"]
    slot._plan_goal = "Finish once"
    slot._auto_run = True
    started: list[str] = []

    async def _mock_run_chat(_state, _slot, _message, **_kwargs):
        _slot.append("assistant", "stage complete", "msg msg-a")

    async def _start_queued(_state, _slot):
        if not _slot._queue:
            return False
        entry = _slot.queue_pop(0)
        started.append(entry["content"])
        _slot.task = asyncio.create_task(asyncio.sleep(0))
        return True

    async def _late_done_payload(_state, _slot, **_kwargs):
        await asyncio.sleep(0)
        _slot.queue_append("queued during done payload")
        return {"slot": _slot.key, "continuing": True, "needs_input": False}

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._start_next_queued_turn",
        _start_queued,
    )
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator.chat_done_payload",
        _late_done_payload,
    )

    controller = asyncio.create_task(_stage_loop(state, slot, auto_run=True))
    slot.track_stage_controller(controller)
    slot.task = controller
    await asyncio.wait_for(controller, timeout=5)
    if slot.task is not None and slot.task is not controller:
        await asyncio.wait_for(slot.task, timeout=1)

    assert started == ["queued during done payload"]
    assert slot._queue == []


class _ReboundStageDeliveryManager:
    """Hold the terminal report under the stage turn's original parent key."""

    def __init__(self, original_key: str) -> None:
        self.original_key = original_key
        self.release = asyncio.Event()
        self.wait_started = asyncio.Event()
        self.waited = False

    def running_agents_for(self, _parent: str) -> list[dict]:
        return []

    async def wait_for_parent_reports(self, parent: str) -> bool:
        if parent != self.original_key or self.waited:
            return False
        self.waited = True
        self.wait_started.set()
        await self.release.wait()
        return True


@pytest.mark.asyncio
async def test_stage_rebind_still_waits_for_the_original_parent_reports(tmp_path, monkeypatch):
    """A mid-stage link cannot move the boundary away from existing children."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("rebind-handoff", mode="orchestrator")
    original_key = f"dashboard:{slot.key}"
    manager = _ReboundStageDeliveryManager(original_key)
    state.subagents = manager
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    stage_two_started = asyncio.Event()
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **_kwargs):
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            _slot.linked_session_key = "cron:rebound-parent"
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            stage_two_started.set()
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)

    controller = asyncio.create_task(_stage_loop(state, slot, auto_run=True))
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(stage_two_started.wait(), timeout=0.1)
    finally:
        manager.release.set()
    await asyncio.wait_for(controller, timeout=5)

    assert manager.wait_started.is_set()
    assert order == ["stage-1", "stage-2"]


@pytest.mark.asyncio
async def test_failed_completion_turn_requeues_its_unconsumed_entry(tmp_path, monkeypatch):
    """A pre-consumption turn failure must leave the exact completion retryable."""
    from kiro_crew.dashboard.chat_runner import _start_next_queued_turn

    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("failed-completion-handoff", mode="orchestrator")
    slot._in_stage_execution = True
    content = "[Subagent completion event]\nresult still owed"
    consumption: list[bool] = []
    slot.queue_insert(
        0,
        content,
        kind=SUBAGENT_COMPLETION_KIND,
        on_consumed=consumption.append,
    )

    async def _failing_turn(*_args, **_kwargs):
        raise RuntimeError("completion turn failed before consumption")

    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _failing_turn)

    assert await _start_next_queued_turn(state, slot) is True
    task = slot.task
    assert task is not None
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    retries = [
        entry
        for entry in slot._queue
        if entry.get("kind") == SUBAGENT_COMPLETION_KIND and entry.get("content") == content
    ]
    assert len(retries) == 1
    assert callable(retries[0].get("_on_consumed"))
    assert consumption == []
