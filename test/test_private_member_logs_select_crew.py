"""Private-member fixes for kiro_cli_logs (C) and select_crew (D).

Both bugs came from resolving a private member's identity through code that
READS the member's transcript, which the "Private member view" sandbox hides —
so the read failed and each tool degraded the wrong way. The fixes resolve the
member from its readable binding (C) and treat an unreadable/absent transcript
mode as unknown (D).
"""

from __future__ import annotations

import json
import unittest.mock
from pathlib import Path

import kiro_crew.mcp_core as mcp_core
from kiro_crew.history import ConversationLog


class TestKiroCliLogsPrivateMember:
    def _bind_member(self, home, monkeypatch, *, session="dashboard:alice", store="member-alice"):
        from member_memory_helpers import forget_declared_stores, write_member_home

        from kiro_crew import memory_stores
        from kiro_crew.vector_memory import VectorMemoryStore

        monkeypatch.setenv("KIROCREW_HOME", str(home))
        write_member_home(home, store.removeprefix("member-"))
        forget_declared_stores(monkeypatch)
        tier = VectorMemoryStore(
            db_path=home / "memory_stores" / store / memory_stores.MEMORY_DB_FILE
        )
        tier.init()
        self._tier = tier
        from kiro_crew.member_memory_auth import bind_private_session_store

        bind_private_session_store(session, store)
        monkeypatch.setattr(mcp_core, "_resolve_session_key_strict", lambda: session)

    def test_private_member_gets_accurate_refusal_not_identity_error(self, tmp_path, monkeypatch):
        # A private member (binding present, transcripts hidden) must get the
        # INTENDED "unavailable to private members" refusal — not the old
        # "protected memory identity is unavailable", which the transcript-
        # reading mcp_memory_scope produced by raising.
        self._bind_member(tmp_path, monkeypatch)
        captured: dict = {}

        class _Sel:
            def log_tool_invocation(self, **kw):
                captured.update(kw)

        monkeypatch.setattr(mcp_core, "sel", lambda: _Sel())
        out = mcp_core._call_tool_inner("kiro_cli_logs", {})
        assert out == "Error: shared kiro-cli logs are unavailable to private members."
        # And the audit outcome is the deliberate one, not the error fallthrough.
        assert captured.get("outcome") == "denied_memory_scope"

    def test_global_v1_caller_reads_logs(self, tmp_path, monkeypatch):
        # No binding, no boundaries -> the original Global caller contract: the
        # tool proceeds to read logs (stubbed here) rather than refusing.
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
        from kiro_crew.mcp_tools import logs as logs_tool

        monkeypatch.setattr(
            logs_tool.diagnostics, "read_kiro_cli_logs", lambda tail=None, since=None: "LOG BODY"
        )
        monkeypatch.setattr(mcp_core, "sel", lambda: unittest.mock.MagicMock())
        out = mcp_core._call_tool_inner("kiro_cli_logs", {})
        assert "unavailable to private members" not in out
        assert "LOG BODY" in out


def _write_cfg(tmp_path: Path) -> Path:
    data = {
        "agents": {
            "default": {
                "kiro_agent": "kirocrew",
                "workspace": "default",
                "memory_store": "default",
            },
            "oncall": {
                "kiro_agent": "oncall-agent",
                "workspace": "oncall-ws",
                "memory_store": "oncall-mem",
                "triggers": "incident, prod outage",
            },
        },
        "default_agent": "default",
        "workspaces": {"default": {"dir": "workspace"}, "oncall-ws": {"dir": "oncall"}},
        "memory_stores": {
            "default": {},
            "oncall-mem": {"owner_member": "oncall", "memory_version": 2},
        },
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class TestSelectCrewMemoryModeDegrade:
    """select_crew's routing log: fail closed to 'incognito' for a bound member
    whose mode can't be read, but keep the legacy 'persistent' default for an
    ordinary Global session's header — never record mode ''."""

    def _run_and_capture_mode(self, tmp_path, monkeypatch, *, metadata, is_member):
        p = _write_cfg(tmp_path)
        captured: dict = {}

        def _record(member, session_key, memory_mode, **kw):
            captured["mode"] = memory_mode
            return True

        monkeypatch.setattr(mcp_core, "record_activity", _record)
        monkeypatch.setattr(mcp_core, "_resolve_session_key", lambda: "dashboard:alice")
        # Simulate the transcript this process would see: absent/hidden -> {}
        # readable-empty, a legacy header, or a real mode line.
        monkeypatch.setattr(ConversationLog, "get_metadata", lambda self, key: metadata)
        # Whether the caller is a bound private member (binding readable) decides
        # the fail-closed direction; stub it rather than laying down a binding.
        monkeypatch.setattr(mcp_core, "_select_crew_is_private_member", lambda _sk: is_member)
        with unittest.mock.patch("kiro_crew.config.loader.config_path", return_value=p):
            mcp_core._do_select_crew("oncall")
        return captured.get("mode")

    def test_private_member_hidden_transcript_degrades_to_incognito(self, tmp_path, monkeypatch):
        # A bound private member whose transcript is hidden reads as {} (NOT an
        # exception); the old `.get("memory_mode", "")` recorded "" and durably
        # defeated the mode. It must fail closed to incognito.
        mode = self._run_and_capture_mode(tmp_path, monkeypatch, metadata={}, is_member=True)
        assert mode == "incognito"

    def test_legacy_global_header_without_mode_stays_persistent(self, tmp_path, monkeypatch):
        # A Global (non-member) session whose valid header predates the
        # memory_mode field is legacy persistent, NOT a no-trace session — it
        # must not be downgraded to incognito (Codex/GPT lane P2).
        mode = self._run_and_capture_mode(
            tmp_path, monkeypatch, metadata={"title": "something"}, is_member=False
        )
        assert mode == "persistent"

    def test_global_empty_metadata_stays_persistent(self, tmp_path, monkeypatch):
        # No metadata at all on a non-member session is still the legacy Global
        # persistent case, never "" and never incognito.
        mode = self._run_and_capture_mode(tmp_path, monkeypatch, metadata={}, is_member=False)
        assert mode == "persistent"

    def test_real_persistent_mode_is_preserved(self, tmp_path, monkeypatch):
        # A positively-read mode is recorded as-is (the degrade only fires when
        # the mode cannot be read), for a member or a Global session alike.
        mode = self._run_and_capture_mode(
            tmp_path, monkeypatch, metadata={"memory_mode": "persistent"}, is_member=False
        )
        assert mode == "persistent"

    def test_member_explicit_incognito_mode_preserved(self, tmp_path, monkeypatch):
        # A member whose header DOES carry the mode records it verbatim.
        mode = self._run_and_capture_mode(
            tmp_path, monkeypatch, metadata={"memory_mode": "incognito"}, is_member=True
        )
        assert mode == "incognito"
