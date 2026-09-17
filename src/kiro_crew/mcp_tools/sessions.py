"""The reading this workspace's own chat history tools: what they advertise and what they do.

``schemas()`` returns the ADVERTISEMENT half of each tool -- its name, the
model-facing description, and the JSON Schema a call is validated against.
``HANDLERS`` maps each of those names to the function that runs it. Both halves
of a tool live here so its contract and its behavior are read together, and
``test_mcp_tool_registry`` fails if one arrives without the other.

Handlers reach this server's shared plumbing as attributes of ``mcp_core`` --
``mcp_core._post``, the identity resolvers, the governance vets. That is
deliberate rather than untidy: an attribute lookup resolves at CALL time, so a
test that rebinds one on the module still intercepts the handler. Importing
those names directly here would bind them at import time and silently escape
every existing patch site.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from kiro_crew import mcp_core
from kiro_crew.context import RECALL_ROLES
from kiro_crew.history import ConversationLog
from kiro_crew.validation import (
    GET_CHAT_SESSION_SCHEMA,
    LIST_SESSIONS_SCHEMA,
    SEARCH_CHAT_HISTORY_SCHEMA,
    validate_tool_args,
)


def schemas() -> list[dict[str, Any]]:
    """Descriptors for the sessions tools."""
    return [
        {
            "name": "search_chat_history",
            "description": (
                "Search your own past conversation transcripts (chat history) by "
                "keyword and get back ranked, snippet-level hits. Use this to "
                "recover the exact words of a past conversation — 'the error message "
                "from that debugging session', a name/number/path mentioned earlier, "
                "the verbatim evidence behind a conclusion memory_recall gave you. "
                "For what was decided or learned, call memory_recall first: it "
                "searches the memory store bound to this session by meaning. "
                "Search like a human: try a query, read the snippets, then re-search "
                "with different keywords if the first hit isn't right. Returns "
                "metadata + a short snippet per session (NOT full transcripts) — "
                "call get_chat_session with a returned session_key to read the full "
                "thread once a hit looks promising. Scoped to your current workspace "
                "by default. This is a READ — it never modifies memory or history."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword(s) to search for in past conversations.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10, max 50).",
                        "default": 10,
                    },
                    "before": {
                        "type": "string",
                        "description": "Optional ISO date (YYYY-MM-DD); only sessions modified before this day.",
                    },
                    "after": {
                        "type": "string",
                        "description": "Optional ISO date (YYYY-MM-DD); only sessions modified on/after this day.",
                    },
                    "all_workspaces": {
                        "type": "boolean",
                        "description": "Search across all workspaces instead of just the current one (default false).",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_chat_session",
            "description": (
                "Read the full message transcript of one past conversation, "
                "identified by a session_key returned from search_chat_history. "
                "Returns the messages as role/content pairs, tail-capped at "
                "max_messages. Use after search_chat_history when a snippet hit "
                "looks like the thread you need. Refuses incognito/temporary "
                "sessions. This is a READ — it never modifies memory or history."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_key": {
                        "type": "string",
                        "description": "The session_key from a search_chat_history result.",
                    },
                    "max_messages": {
                        "type": "integer",
                        "description": "Max (most recent) messages to return (default 50, max 200).",
                        "default": 50,
                    },
                    "all_workspaces": {
                        "type": "boolean",
                        "description": "Allow reading a session from a different workspace than the caller's (default false — deny cross-workspace).",
                        "default": False,
                    },
                },
                "required": ["session_key"],
            },
        },
        {
            "name": "list_sessions",
            "description": (
                "List your recent conversation sessions in this workspace so you "
                "can see the work in flight and what you've been doing — titles, "
                "owning agent, message volume, and last-activity time, newest "
                "first. Use this when the user asks 'what are you working on?', "
                "'what sessions are open?', 'what have we been doing?', or when you "
                "need a bird's-eye view of your own workspace before acting. This "
                "is a READ — it never modifies memory or history. It complements "
                "search_chat_history (which finds a specific past thread by "
                "keyword): list_sessions is the browse/overview, search is the "
                "lookup. Scoped to your current workspace by default; "
                "incognito/temporary sessions are never listed."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max sessions to return, newest first (default 20, max 100).",
                        "default": 20,
                    },
                    "all_workspaces": {
                        "type": "boolean",
                        "description": "List sessions across all workspaces instead of just the current one (default false).",
                        "default": False,
                    },
                    "summarize": {
                        "type": "boolean",
                        "description": (
                            "When true, generate a fresh one-line LLM summary for the top "
                            "sessions (bounded, best-effort — costs tokens + latency, so it's "
                            "opt-in). When false (default), the existing session title is used "
                            "with zero cost."
                        ),
                        "default": False,
                    },
                },
            },
        },
    ]


def _private_member_store() -> tuple[str, str]:
    """``(store, refusal)`` when the caller is a bound private member.

    Reads the caller's own gateway-published session binding
    (``member-memory-bindings/sessions/<sha256>/memory.json``), which the
    private-member sandbox view keeps READABLE — unlike the transcript
    directory it hides. So this answers "am I a private member, and to which
    store?" WITHOUT the transcript read that
    ``member_memory_auth.mcp_memory_scope`` performs and that fails closed
    inside the member view (the read that produced the old, misleading "private
    memory is unavailable" for every history tool). A non-empty store means the
    history tools must relay to the gateway, which can see the transcripts;
    ``("", "")`` means an ordinary Global V1 / owner caller that reads
    in-process.
    """
    from kiro_crew.member_memory_auth import (
        private_memory_boundaries_active,
        read_private_session_store,
    )

    if not private_memory_boundaries_active():
        return "", ""
    session, error = mcp_core.require_strict_session_key(
        "Error: chat history requires a verified session."
    )
    if not session:
        return "", error
    try:
        # ONLY the binding, never ``mcp_memory_scope``: that resolves the store
        # by reading the caller's transcript, which the private-member sandbox
        # view HIDES — so calling it here would raise for exactly the members
        # this exists to route, reintroducing the false "private memory is
        # unavailable". The binding leaf stays readable in the member view, and
        # the gateway re-verifies the caller's member proof and process identity
        # (``internal_memory_scope``) before serving anything, so the authority
        # checks ``mcp_memory_scope`` would run are not lost — they move to the
        # side that can actually see the transcripts.
        return read_private_session_store(session) or "", ""
    except (OSError, ValueError):
        return "", "Error: this session's private memory is unavailable."


def _in_process_history_scope(
    member_store: str,
) -> tuple[str | None, dict[str, tuple[str, ...]], str]:
    """Visibility scope for an in-process history read (no gateway relay).

    A bound member whose transcripts are readable in this process reads with its
    own bound ``member_store`` as the scope, so the ``_history_memory_visible``
    gate confines it to its own store and never falls back to an unscoped read;
    every other caller defers to :func:`_history_memory_scope`.
    """
    if member_store:
        from kiro_crew.member_memory_auth import private_history_session_index

        try:
            return member_store, private_history_session_index(), ""
        except (OSError, ValueError):
            return None, {}, "Error: this session's private memory is unavailable."
    return _history_memory_scope()


def _transcripts_readable_in_process() -> bool:
    """True when this process can read the transcript tree in-process.

    The private-member view sandbox hides ``<config_dir>/sessions/``, so a
    member process sees no such directory while the gateway (and any
    unsandboxed reader) does; a bound member relays its history tools only when
    this is False, and reads in-process with its own store as the scope when it
    is True.
    """
    import os

    from kiro_crew.history import ConversationLog

    try:
        sessions_dir = ConversationLog()._dir
        return sessions_dir.is_dir() and os.access(sessions_dir, os.R_OK | os.X_OK)
    except (OSError, ValueError):
        return False


def _history_memory_scope() -> tuple[str | None, dict[str, tuple[str, ...]], str]:
    """In-process visibility scope for a Global V1 / owner caller.

    Returns ``(None, {}, "")`` — an unrestricted in-process read — while private
    memory boundaries are off. With boundaries on, the caller's store is resolved
    the way ``member_memory_auth.mcp_memory_scope`` always did for an in-process
    reader, so a Global V1 caller gets scope ``""`` and a bound private member
    whose transcripts are visible in this process gets its own store, and the
    ``_history_memory_visible`` gate keeps every other store's transcript out of
    the results. A bound member whose sandbox view hides the transcripts relays
    to the gateway before reaching here, so the transcript read below runs only
    where the transcripts are visible.
    """
    from kiro_crew.member_memory_auth import (
        mcp_memory_scope,
        private_history_session_index,
        private_memory_boundaries_active,
    )

    if not private_memory_boundaries_active():
        return None, {}, ""
    session, error = mcp_core.require_strict_session_key(
        "Error: chat history requires a verified session."
    )
    if not session:
        return None, {}, error
    try:
        scope = mcp_memory_scope(session)
        return scope, private_history_session_index(), ""
    except (OSError, ValueError):
        return None, {}, "Error: this session's private memory is unavailable."


def _history_memory_visible(key: str, scope: str | None, index: dict[str, tuple[str, ...]]) -> bool:
    if scope is None:
        return True
    from kiro_crew.history import transcript_stems
    from kiro_crew.member_memory_auth import private_memory_store_for_session

    try:
        # An unsigned transcript cannot supply its own canonical identity.
        # Every candidate comes from the protected snapshot and is re-read
        # against its current store and transcript before any content is shown.
        candidates = set(index.get(key, ()))
        for stem in transcript_stems(key):
            candidates.update(index.get(stem, ()))
        return all(
            private_memory_store_for_session(candidate) == scope
            for candidate in candidates or {key}
        )
    except (OSError, ValueError):
        return False


def _relay_history_tool(op: str, args: dict[str, Any]) -> str:
    """Run a history tool on the gateway, for a private member that cannot read
    transcripts in its own sandbox view.

    The member process runs inside the "Private member view" sandbox, which
    hides ``sessions/`` (every transcript) and ``snapshots/`` — so the
    in-process path sees zero transcripts and the store-resolution read fails
    closed. ``memory_recall`` already works from a member by relaying over HTTP
    to the gateway (``/api/memory/recall``), because the GATEWAY process can see
    the transcripts and resolves the member's store itself. This does the same
    for the three history tools: it POSTs the op + validated args to a
    member-scoped gateway route, which authenticates the caller's member proof,
    resolves its store, applies the same store-visibility and incognito rules,
    and returns the fully-rendered tool text. Global V1 / owner callers never
    reach here — they read in-process.

    The session key sent on the wire is the STRICT identity this tool already
    verified (``_private_member_store`` gated on it), not a re-resolved one, so
    the gateway authorizes the same session that was checked.
    """
    session, error = mcp_core.require_strict_session_key(
        "Error: chat history requires a verified session."
    )
    if not session:
        return error
    resp = mcp_core._post(
        "/api/sessions/member-history",
        {"op": op, "args": args},
        session_key=session,
    )
    if not isinstance(resp, dict):
        return "Error: chat history is unavailable (unexpected gateway response)."
    err = resp.get("error")
    if err:
        # The gateway already redacts its own error bodies at the trust
        # boundary (``_http_error_body``); surface it so a member sees the real
        # cause (an unverifiable session, a disabled store) rather than the old
        # blanket "private memory is unavailable".
        return f"Error: {err}"
    output = resp.get("output")
    return output if isinstance(output, str) else "Error: chat history returned no result."


def search_chat_history(name: str, args: dict[str, Any]) -> str:
    args = validate_tool_args(args, SEARCH_CHAT_HISTORY_SCHEMA)
    member_store, member_refusal = _private_member_store()
    if member_refusal:
        return member_refusal
    if member_store and not _transcripts_readable_in_process():
        return _relay_history_tool("search_chat_history", args)
    memory_scope, memory_index, refusal = _in_process_history_scope(member_store)
    if refusal:
        return refusal
    return _search_chat_history_core(
        args, memory_scope, memory_index, mcp_core._resolve_session_key()
    )


def _search_chat_history_core(
    args: dict[str, Any],
    memory_scope: str | None,
    memory_index: dict[str, tuple[str, ...]],
    session_key: str,
) -> str:
    """The scope-agnostic body of ``search_chat_history``.

    Shared by the in-process handler (Global V1 / owner, ``memory_scope`` None)
    and the gateway member-history route (a private member's store, resolved
    gateway-side). Splitting it here keeps ONE copy of the incognito / workspace
    / date filtering and the snippet rendering, so the member relay path cannot
    drift from the direct path.
    """
    query = args["query"]
    limit = args.get("limit", 10)
    all_workspaces = args.get("all_workspaces", False)
    # A supplied-but-unparseable date (one that passes the regex but names no
    # real calendar day, like Feb 30) must ERROR, not be silently dropped — a silent
    # drop would return the UNFILTERED set and mislead the caller.
    after_epoch = before_epoch = None
    if args.get("after"):
        after_epoch = mcp_core._parse_iso_date_epoch(args["after"])
        if after_epoch is None:
            return "Invalid 'after' date — use a real calendar date (YYYY-MM-DD)."
    if args.get("before"):
        before_epoch = mcp_core._parse_iso_date_epoch(args["before"])
        if before_epoch is None:
            return "Invalid 'before' date — use a real calendar date (YYYY-MM-DD)."

    cl = ConversationLog()
    # Default scoping: confine to the caller's workspace (fail-closed — unset
    # buckets to "default"). all_workspaces opts out.
    current_ws: str | None = None if all_workspaces else mcp_core._caller_workspace(cl, session_key)

    # Fetch the FULL ranked match set (bounded by the backend's scan window),
    # not a fixed limit*3 over-fetch: heavy incognito/workspace/date drops on
    # the first page could otherwise starve a caller whose real matches rank
    # lower, returning "no results" while hits exist.
    ranked: list[dict] = cl.search_sessions(query, limit=mcp_core._SEARCH_HISTORY_SCAN)

    results: list[dict] = []
    for meta in ranked:
        key = meta.get("key", "")
        if not key:
            continue
        if not _history_memory_visible(key, memory_scope, memory_index):
            continue
        # TOCTOU: the file may be unlinked (clear-sessions, rotation, concurrent
        # process) between the ranked snapshot and this read. has_log is the
        # existence gate so we never emit a ghost row for a session the read
        # tool cannot retrieve. Do NOT additionally require non-empty
        # metadata: a legacy session whose file predates the metadata line
        # returns {} here yet get_chat_session serves it fine, so rejecting {}
        # would hide those sessions from search while they remain readable.
        if not cl.has_log(key):
            continue
        full_meta = cl.get_metadata(key)
        if mcp_core._history_is_incognito(full_meta) or mcp_core._history_is_incognito(meta):
            continue  # EB-5: incognito/temporary never surface
        if current_ws is not None and mcp_core._ws_bucket(full_meta.get("workspace")) != current_ws:
            continue  # EB-cc3: workspace scoping (fail-closed; normalizes non-str)
        modified = meta.get("modified", 0) or 0
        if after_epoch is not None and modified < after_epoch:
            continue
        if before_epoch is not None and modified >= before_epoch:
            continue

        snippet = mcp_core._extract_history_snippet(cl.read_messages(key), query)
        results.append(
            {
                "session_key": key,
                "title": meta.get("title") or key,
                "date": meta.get("created") or "",
                "snippet": snippet,
            }
        )
        if len(results) >= limit:
            break

    if not results:
        mcp_core.sel().log_tool_invocation(
            session_key=session_key,
            source="mcp",
            tool_name="search_chat_history",
            outcome="no_results",
            metadata={"query_len": len(query)},
        )
        return "No matching conversations found. Try different keywords."

    lines = [
        "\U0001f50e Chat history matches "
        "(snippets only — use get_chat_session to read a full thread):"
    ]
    for r in results:
        lines.append("\n---")
        lines.append(f"**{r['title']}**  ·  `{r['session_key']}`")
        if r["date"]:
            lines.append(f"_{r['date']}_")
        if r["snippet"]:
            lines.append(f"\n{r['snippet']}")

    output = "\n".join(lines)
    # EB-6: redact secrets/exfil URLs from snippets before returning.
    output = mcp_core._redact_history_output(output)
    mcp_core.sel().log_tool_invocation(
        session_key=session_key,
        source="mcp",
        tool_name="search_chat_history",
        outcome="success",
        metadata={"query_len": len(query), "result_count": len(results)},
    )
    return output


def get_chat_session(name: str, args: dict[str, Any]) -> str:
    args = validate_tool_args(args, GET_CHAT_SESSION_SCHEMA)
    member_store, member_refusal = _private_member_store()
    if member_refusal:
        return member_refusal
    if member_store and not _transcripts_readable_in_process():
        return _relay_history_tool("get_chat_session", args)
    memory_scope, memory_index, refusal = _in_process_history_scope(member_store)
    if refusal:
        return refusal
    return _get_chat_session_core(args, memory_scope, memory_index, mcp_core._resolve_session_key())


def _get_chat_session_core(
    args: dict[str, Any],
    memory_scope: str | None,
    memory_index: dict[str, tuple[str, ...]],
    session_key: str,
) -> str:
    """The scope-agnostic body of ``get_chat_session`` (see
    :func:`_search_chat_history_core` for why this split exists)."""
    key = args["session_key"]
    max_messages = args.get("max_messages", 50)
    all_workspaces = args.get("all_workspaces", False)
    if not _history_memory_visible(key, memory_scope, memory_index):
        mcp_core.sel().log_tool_invocation(
            session_key=session_key,
            source="mcp",
            tool_name="get_chat_session",
            outcome="denied_memory_scope",
        )
        return "Access denied: that conversation belongs to a different memory store."

    # Defense-in-depth on a path-bearing identifier: ConversationLog._safe_key
    # already neutralizes separators. Reject path separators outright, and ".."
    # only as a STANDALONE component — not as a substring — so legitimate keys
    # like "dashboard_chat-2..3" round-trip between search and read. (A strict
    # allowlist regex is avoided: real keys legitimately contain ':' and '.')
    if "/" in key or "\\" in key or key in ("..", "."):
        mcp_core.sel().log_tool_invocation(
            session_key=session_key,
            source="mcp",
            tool_name="get_chat_session",
            outcome="rejected_bad_key",
        )
        return "Invalid session_key."

    cl = ConversationLog()
    if not cl.has_log(key):
        mcp_core.sel().log_tool_invocation(
            session_key=session_key,
            source="mcp",
            tool_name="get_chat_session",
            outcome="not_found",
        )
        # Do NOT echo the raw caller-supplied key: the dashboard renders it as
        # live markdown, so a crafted key (e.g. "[x](https://evil/)") would be a
        # reflected phishing/prompt-injection payload. Return a stable
        # fingerprint instead — enough to correlate, safe to render. (Not a
        # security signature — just a display-safe correlation id — but use
        # sha256 anyway so no weak-hash scanner flags this egress path.)
        fp = hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()[:12]
        return f"No conversation found for that session_key (fp:{fp})."

    meta = cl.get_metadata(key)
    if mcp_core._history_is_incognito(meta):
        # EB-7b: no bypass of incognito exclusion via direct fetch.
        mcp_core.sel().log_tool_invocation(
            session_key=session_key,
            source="mcp",
            tool_name="get_chat_session",
            outcome="refused_incognito",
        )
        return "That conversation is private (incognito/temporary) and cannot be read."

    # Deny-by-default workspace isolation: mirror search_chat_history's
    # fail-closed scoping so a caller can't bypass it by fetching a session
    # from another workspace directly. Unset/non-string workspaces bucket as
    # "default" via _ws_bucket.
    if not all_workspaces:
        caller_ws = mcp_core._caller_workspace(cl, session_key)
        if mcp_core._ws_bucket(meta.get("workspace")) != caller_ws:
            mcp_core.sel().log_tool_invocation(
                session_key=session_key,
                source="mcp",
                tool_name="get_chat_session",
                outcome="denied_cross_workspace",
            )
            return "Access denied: that conversation belongs to a different workspace."

    # RECALL_ROLES rather than a literal, because this is the one surface whose
    # whole purpose is reading a past session: a breadcrumb appended with
    # role="inject" (a /note, a cron result) is precisely a message meant to
    # survive the session boundary being crossed here, and a hardcoded
    # {"user", "assistant"} dropped it. The constant already governs replay and
    # compression in context.py, so sharing it keeps the fetch from drifting
    # from them. Note it is narrowING as well as widening: "system" is absent
    # from RECALL_ROLES, so passing no roles at all would not be equivalent --
    # recent() treats a falsy roles as "no filter" and would admit internal
    # rows here.
    messages = cl.recent(key, max_messages=max_messages, roles=RECALL_ROLES)
    if not messages:
        mcp_core.sel().log_tool_invocation(
            session_key=session_key,
            source="mcp",
            tool_name="get_chat_session",
            outcome="empty",
        )
        return mcp_core._redact_history_output(f"Conversation `{key}` has no readable messages.")

    title = meta.get("title") or key
    lines = [f"\U0001f4dc Conversation: **{title}**  ·  `{key}`", ""]
    for m in messages:
        role = str(m.get("role", "?")).title()
        lines.append(f"**{role}:** {m.get('content', '')}")
        lines.append("")

    output = mcp_core._redact_history_output("\n".join(lines))
    mcp_core.sel().log_tool_invocation(
        session_key=session_key,
        source="mcp",
        tool_name="get_chat_session",
        outcome="success",
        metadata={"message_count": len(messages)},
    )
    return output


def list_sessions(name: str, args: dict[str, Any]) -> str:
    args = validate_tool_args(args, LIST_SESSIONS_SCHEMA)
    member_store, member_refusal = _private_member_store()
    if member_refusal:
        return member_refusal
    if member_store and not _transcripts_readable_in_process():
        return _relay_history_tool("list_sessions", args)
    memory_scope, memory_index, refusal = _in_process_history_scope(member_store)
    if refusal:
        return refusal
    return _list_sessions_core(args, memory_scope, memory_index, mcp_core._resolve_session_key())


def _list_sessions_core(
    args: dict[str, Any],
    memory_scope: str | None,
    memory_index: dict[str, tuple[str, ...]],
    session_key: str,
) -> str:
    """The scope-agnostic body of ``list_sessions`` (see
    :func:`_search_chat_history_core` for why this split exists)."""
    limit = args.get("limit", 20)
    all_workspaces = args.get("all_workspaces", False)
    summarize = args.get("summarize", False)
    cl = ConversationLog()
    list_ws: str | None = None if all_workspaces else mcp_core._caller_workspace(cl, session_key)

    rows: list[dict] = []
    for meta in cl.list_sessions():
        key = meta.get("key", "")
        if not key:
            continue
        if not _history_memory_visible(key, memory_scope, memory_index):
            continue
        if mcp_core._history_is_incognito(meta):
            continue  # incognito/temporary never surface
        if list_ws is not None:
            # list_sessions() rows omit `workspace`, so scope off the full
            # metadata line (mirrors search_chat_history). Runs in the MCP
            # process, not the gateway loop, so the extra read is fine.
            if mcp_core._ws_bucket(cl.get_metadata(key).get("workspace")) != list_ws:
                continue  # fail-closed workspace scoping
        rows.append(meta)
        if len(rows) >= limit:
            break

    if not rows:
        mcp_core.sel().log_tool_invocation(
            session_key=session_key,
            source="mcp",
            tool_name="list_sessions",
            outcome="no_results",
        )
        return "No sessions found in this workspace yet."

    # Opt-in: ask the gateway (which owns the LLM background session) to
    # generate fresh one-line summaries for the returned keys. Best-effort —
    # any failure falls back to titles, so the list is always returned.
    summaries: dict[str, str] = {}
    if summarize:
        resp = mcp_core._post(
            "/api/sessions/summarize",
            {"keys": [r["key"] for r in rows]},
            timeout=120,
        )
        if isinstance(resp, dict) and isinstance(resp.get("summaries"), dict):
            summaries = {str(k): str(v) for k, v in resp["summaries"].items() if v}

    scope_label = "across all workspaces" if all_workspaces else "in this workspace"
    lines = [f"\U0001f5c2\ufe0f Sessions {scope_label} ({len(rows)}, newest first):"]
    for r in rows:
        key = r["key"]
        title = r.get("title") or key
        agent = r.get("agent")
        msgs = r.get("messages", 0)
        created = r.get("created", "")
        meta_bits = []
        if agent:
            meta_bits.append(f"agent={agent}")
        meta_bits.append(f"~{msgs} msgs")
        if created:
            meta_bits.append(str(created)[:16])
        lines.append("\n---")
        lines.append(f"**{title}**  ·  `{key}`")
        lines.append(f"_{'  ·  '.join(meta_bits)}_")
        summary = summaries.get(key)
        if summary:
            lines.append(f"\n{summary}")

    output = mcp_core._redact_history_output("\n".join(lines))
    mcp_core.sel().log_tool_invocation(
        session_key=session_key,
        source="mcp",
        tool_name="list_sessions",
        outcome="success",
        metadata={"result_count": len(rows), "summarized": len(summaries)},
    )
    return output


HANDLERS: dict[str, Callable[[str, dict[str, Any]], str]] = {
    "search_chat_history": search_chat_history,
    "get_chat_session": get_chat_session,
    "list_sessions": list_sessions,
}
