"""Private-member (V2 memory) chat-history MCP tools.

A private member's kiro-cli + mcp-core run inside the "Private member view"
sandbox, which HIDES the transcript directory (``sessions/``). The history
tools therefore cannot read transcripts in-process, exactly like
``memory_recall`` cannot, and both must relay over HTTP to the gateway (which
CAN see the transcripts and resolves the member's bound store itself).

These tests simulate that view the way the brief prescribes: a valid session
binding at ``member-memory-bindings/sessions/<sha256>/memory.json`` exists,
while the transcript directory is absent — so the in-process store resolution
fails closed. They assert:

* the tools RELAY instead of returning the old misleading "private memory is
  unavailable" message (regression: the whole point of the fix);
* the gateway route authenticates the member, resolves the store, and renders
  the same output the direct path renders, scoped to the member's own store;
* an unverified caller is refused, and the member sees only its own transcripts.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from kiro_crew import mcp_core
from kiro_crew.history import ConversationLog
from kiro_crew.mcp_tools import sessions as session_tools


def _declare_members(home, monkeypatch, *members):
    """Declare one V2 store per member under *home*, with its DB initialised.

    ``write_member_home`` writes the config + manifest, but ``require_memory_store``
    (reached from ``bind_private_session_store``) also needs the store's on-disk
    ``memory.db``, so each store's vector tier is opened once here (as the
    ``env`` fixture does). Returns the opened tiers so the caller can close them.
    """
    from member_memory_helpers import forget_declared_stores, write_member_home

    from kiro_crew import memory_stores
    from kiro_crew.vector_memory import VectorMemoryStore

    monkeypatch.setenv("KIROCREW_HOME", str(home))
    write_member_home(home, *members)
    forget_declared_stores(monkeypatch)
    tiers = []
    for member in members:
        db = home / "memory_stores" / f"member-{member}" / memory_stores.MEMORY_DB_FILE
        tier = VectorMemoryStore(db_path=db)
        tier.init()
        tiers.append(tier)
    return tiers


def _boundaries_and_binding(home, monkeypatch, *, store="member-alice", session="dashboard:alice"):
    """Point KIROCREW_HOME at *home*, activate boundaries, and bind *session*.

    Writes the gateway-published session binding (the leaf the private-member
    sandbox keeps READABLE) without laying down any transcript — the shape the
    fix must handle. Returns nothing; call it before invoking a tool.
    """
    _declare_members(home, monkeypatch, store.removeprefix("member-"))
    from kiro_crew.member_memory_auth import bind_private_session_store

    bind_private_session_store(session, store)


class TestPrivateMemberStoreDetection:
    def test_binding_resolves_store_without_reading_transcript(self, tmp_path, monkeypatch):
        # The member store is resolved from the binding leaf alone. No sessions/
        # dir exists here (the private view hides it), so a resolution that
        # touched the transcript would fail — this must NOT.
        _boundaries_and_binding(tmp_path, monkeypatch)
        monkeypatch.setattr(mcp_core, "_resolve_session_key_strict", lambda: "dashboard:alice")
        store, refusal = session_tools._private_member_store()
        assert refusal == ""
        assert store == "member-alice"

    def test_global_caller_reads_in_process(self, tmp_path, monkeypatch):
        # No binding, no boundaries -> ordinary Global V1 caller reads in-process.
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
        store, refusal = session_tools._private_member_store()
        assert store == ""
        assert refusal == ""


class TestPrivateMemberRelay:
    """The tool relays to the gateway for a private member instead of failing closed."""

    def _relay_capture(self, monkeypatch):
        seen: dict = {}

        def _fake_post(path, body=None, *, timeout=30, session_key=None):
            seen["path"] = path
            seen["body"] = body
            seen["session_key"] = session_key
            return {"output": "RELAYED-OUTPUT"}

        monkeypatch.setattr(mcp_core, "_post", _fake_post)
        return seen

    @pytest.mark.parametrize(
        "tool,args",
        [
            ("search_chat_history", {"query": "redis"}),
            ("get_chat_session", {"session_key": "dashboard_chat-1"}),
            ("list_sessions", {}),
        ],
    )
    def test_member_relays_and_gets_output(self, tmp_path, monkeypatch, tool, args):
        _boundaries_and_binding(tmp_path, monkeypatch)
        monkeypatch.setattr(mcp_core, "_resolve_session_key_strict", lambda: "dashboard:alice")
        seen = self._relay_capture(monkeypatch)
        out = mcp_core._call_tool_inner(tool, args)
        # The old misleading refusal is GONE...
        assert "private memory is unavailable" not in out
        # ...and the tool relayed to the member-history route with the op + args.
        assert out == "RELAYED-OUTPUT"
        assert seen["path"] == "/api/sessions/member-history"
        assert seen["body"]["op"] == tool
        # validate_tool_args fills defaults, so the relayed args are a SUPERSET
        # of what was passed; the caller's own fields must survive verbatim.
        for k, v in args.items():
            assert seen["body"]["args"][k] == v
        # The wire session key is the strict-verified identity, not a re-resolved one.
        assert seen["session_key"] == "dashboard:alice"

    def test_relay_surfaces_gateway_error(self, tmp_path, monkeypatch):
        _boundaries_and_binding(tmp_path, monkeypatch)
        monkeypatch.setattr(mcp_core, "_resolve_session_key_strict", lambda: "dashboard:alice")

        def _fake_post(path, body=None, *, timeout=30, session_key=None):
            return {"error": "the member session could not be verified"}

        monkeypatch.setattr(mcp_core, "_post", _fake_post)
        out = mcp_core._call_tool_inner("search_chat_history", {"query": "x"})
        assert "could not be verified" in out
        assert "private memory is unavailable" not in out

    def test_member_relays_when_transcript_dir_absent(self, tmp_path, monkeypatch):
        # Branch 1: a bound member whose sandbox view HIDES sessions/ (the dir
        # does not exist here) cannot read transcripts in-process, so the tool
        # relays to the gateway instead of returning an empty in-process read.
        _boundaries_and_binding(tmp_path, monkeypatch)
        assert not (tmp_path / "sessions").exists()
        monkeypatch.setattr(mcp_core, "_resolve_session_key_strict", lambda: "dashboard:alice")
        relayed: dict = {}

        def _fake_post(path, body=None, *, timeout=30, session_key=None):
            relayed["hit"] = path
            return {"output": "RELAYED-OUTPUT"}

        monkeypatch.setattr(mcp_core, "_post", _fake_post)
        out = mcp_core._call_tool_inner("list_sessions", {})
        assert out == "RELAYED-OUTPUT"
        assert relayed["hit"] == "/api/sessions/member-history"

    def test_member_reads_in_process_when_transcript_dir_present(self, tmp_path, monkeypatch):
        # Branch 2: a bound member whose transcripts ARE readable in this process
        # (sessions/ exists — the gateway / an unsandboxed reader) reads
        # in-process, scoped to its own store, and does NOT relay. A relay here
        # would be a bug.
        _declare_members(tmp_path, monkeypatch, "alice", "bob")
        sessions = tmp_path / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        cl = ConversationLog(base_dir=sessions)
        cl.append("dashboard:alice", "user", "the alice widget in redis")
        cl.update_metadata("dashboard:alice", {"memory_store": "member-alice"})
        cl.append("dashboard:bob", "user", "the bob widget in redis")
        cl.update_metadata("dashboard:bob", {"memory_store": "member-bob"})
        from kiro_crew.member_memory_auth import bind_private_session_store

        bind_private_session_store("dashboard:alice", "member-alice")
        bind_private_session_store("dashboard:bob", "member-bob")
        monkeypatch.setattr(mcp_core, "_resolve_session_key_strict", lambda: "dashboard:alice")
        monkeypatch.setattr(mcp_core, "_resolve_session_key", lambda: "dashboard:alice")

        def _fail_post(path, body=None, *, timeout=30, session_key=None):
            raise AssertionError(f"must not relay when transcripts are readable: {path}")

        monkeypatch.setattr(mcp_core, "_post", _fail_post)
        out = mcp_core._call_tool_inner(
            "search_chat_history", {"query": "widget", "all_workspaces": True}
        )
        # Scoped to alice's store: alice's transcript shows, bob's never does.
        assert "dashboard_alice" in out
        assert "dashboard_bob" not in out
        assert "bob widget" not in out


class TestHistoryCoreScoping:
    """The scope-agnostic core, exercised the way the gateway relay calls it."""

    def _seed(self, home, monkeypatch):
        _declare_members(home, monkeypatch, "alice", "bob")
        sessions = home / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        cl = ConversationLog(base_dir=sessions)
        # A member-alice transcript and a member-bob one, each bound to its store.
        cl.append("dashboard:alice", "user", "the alice widget in redis")
        cl.update_metadata("dashboard:alice", {"memory_store": "member-alice"})
        cl.append("dashboard:bob", "user", "the bob widget in redis")
        cl.update_metadata("dashboard:bob", {"memory_store": "member-bob"})
        from kiro_crew.member_memory_auth import bind_private_session_store

        bind_private_session_store("dashboard:alice", "member-alice")
        bind_private_session_store("dashboard:bob", "member-bob")
        return cl

    def test_search_core_scopes_to_member_store(self, tmp_path, monkeypatch):
        self._seed(tmp_path, monkeypatch)
        from kiro_crew.member_memory_auth import private_history_session_index

        index = private_history_session_index()
        # Alice's store sees only alice's transcript. search_sessions reports
        # filename STEMS ("dashboard_alice"), not the colon key.
        out = session_tools._search_chat_history_core(
            {"query": "widget", "all_workspaces": True}, "member-alice", index, "dashboard:alice"
        )
        assert "dashboard_alice" in out
        assert "dashboard_bob" not in out
        assert "bob widget" not in out

    def test_get_core_denies_cross_member(self, tmp_path, monkeypatch):
        self._seed(tmp_path, monkeypatch)
        from kiro_crew.member_memory_auth import private_history_session_index

        index = private_history_session_index()
        # Alice's store cannot read bob's transcript.
        out = session_tools._get_chat_session_core(
            {"session_key": "dashboard:bob", "all_workspaces": True},
            "member-alice",
            index,
            "dashboard:alice",
        )
        assert "different memory store" in out
        assert "bob widget" not in out


class TestMemberHistoryGatewayRoute:
    """The MCP-only gateway relay route: auth, store resolution, rendering, refusal."""

    def _state(self, home):
        cl = ConversationLog(base_dir=home / "sessions")
        return SimpleNamespace(conversation_log=cl)

    async def _seed_and_call(self, tmp_path, monkeypatch, *, op, args, verified_store):
        from member_memory_helpers import make_request

        _declare_members(tmp_path, monkeypatch, "alice", "bob")
        sessions = tmp_path / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        cl = ConversationLog(base_dir=sessions)
        cl.append("dashboard:alice", "user", "the alice widget in redis")
        cl.update_metadata("dashboard:alice", {"memory_store": "member-alice"})
        cl.append("dashboard:bob", "user", "the bob widget in redis")
        cl.update_metadata("dashboard:bob", {"memory_store": "member-bob"})
        from kiro_crew.member_memory_auth import bind_private_session_store

        bind_private_session_store("dashboard:alice", "member-alice")
        bind_private_session_store("dashboard:bob", "member-bob")

        from kiro_crew.dashboard.handlers import sessions as gw

        # internal_memory_scope is the auth boundary; stub it to the verified
        # member store (its own contract is covered by test_shared/auth tests).
        async def _fake_scope(request, operation, *, claimed_session=None):
            return verified_store, None

        monkeypatch.setattr(gw, "internal_memory_scope", _fake_scope)
        monkeypatch.setattr(
            "kiro_crew.member_memory_auth.private_memory_boundaries_active", lambda: True
        )
        state = SimpleNamespace(conversation_log=cl)
        request = make_request(
            state,
            "/api/sessions/member-history",
            body={"op": op, "args": args},
            internal=True,
            session="dashboard:alice",
        )
        return await gw.api_sessions_member_history(request)

    def test_route_renders_member_scoped_search(self, tmp_path, monkeypatch):
        resp = asyncio.run(
            self._seed_and_call(
                tmp_path,
                monkeypatch,
                op="search_chat_history",
                args={"query": "widget", "all_workspaces": True},
                verified_store="member-alice",
            )
        )
        assert resp.status == 200
        payload = json.loads(resp.text)
        assert "dashboard_alice" in payload["output"]
        assert "dashboard_bob" not in payload["output"]  # scoped to alice's store

    def test_route_refuses_unverified_member(self, tmp_path, monkeypatch):
        from aiohttp import web

        resp = asyncio.run(
            self._seed_and_call_refusal(
                tmp_path,
                monkeypatch,
                refusal=web.json_response(
                    {"error": "unverified", "code": "member_session_unverified"}, status=403
                ),
            )
        )
        assert resp.status == 403

    async def _seed_and_call_refusal(self, tmp_path, monkeypatch, *, refusal):
        from member_memory_helpers import make_request

        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
        from kiro_crew.dashboard.handlers import sessions as gw

        async def _fake_scope(request, operation, *, claimed_session=None):
            return None, refusal

        monkeypatch.setattr(gw, "internal_memory_scope", _fake_scope)
        monkeypatch.setattr(
            "kiro_crew.member_memory_auth.private_memory_boundaries_active", lambda: True
        )
        state = SimpleNamespace(conversation_log=ConversationLog(base_dir=tmp_path / "sessions"))
        request = make_request(
            state,
            "/api/sessions/member-history",
            body={"op": "search_chat_history", "args": {"query": "x"}},
            internal=True,
            session="dashboard:alice",
        )
        return await gw.api_sessions_member_history(request)

    def test_route_rejects_unknown_op(self, tmp_path, monkeypatch):
        resp = asyncio.run(
            self._seed_and_call(
                tmp_path,
                monkeypatch,
                op="delete_everything",
                args={},
                verified_store="member-alice",
            )
        )
        assert resp.status == 400
