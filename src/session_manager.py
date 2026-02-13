from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def flush_state(state: dict[str, Any]) -> None:
    state_path = Path(state["state_path"])  # required
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def init_session_store(workspace: str) -> dict[str, Any]:
    docs_dir = Path(workspace) / "docs"
    base_dir = docs_dir / "sessions"
    base_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "workspace": str(Path(workspace).resolve()),
        "base_dir": str(base_dir.resolve()),
        "state_path": str((base_dir / "state.json").resolve()),
        "sessions": {},
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    flush_state(state)
    return state


def open_session(state: dict[str, Any], session_name: str = "default") -> dict[str, Any]:
    sessions = state.setdefault("sessions", {})
    if session_name in sessions:
        return sessions[session_name]

    base_dir = Path(state["base_dir"])
    session_dir = base_dir / session_name
    responses_dir = session_dir / "responses"
    requests_log = session_dir / "requests.jsonl"
    cookie_jar = session_dir / "cookiejar.txt"

    responses_dir.mkdir(parents=True, exist_ok=True)
    requests_log.touch(exist_ok=True)
    cookie_jar.touch(exist_ok=True)

    session = {
        "name": session_name,
        "session_dir": str(session_dir.resolve()),
        "responses_dir": str(responses_dir.resolve()),
        "requests_log": str(requests_log.resolve()),
        "cookie_jar": str(cookie_jar.resolve()),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    sessions[session_name] = session
    state["updated_at"] = _now_iso()
    flush_state(state)
    return session


def get_cookie_jar_path(state: dict[str, Any], session_name: str) -> str:
    session = open_session(state, session_name=session_name)
    return session["cookie_jar"]


def record_request(state: dict[str, Any], session_name: str, request_data: dict[str, Any]) -> str:
    session = open_session(state, session_name=session_name)
    request_id = request_data.get("request_id") or uuid.uuid4().hex[:16]
    entry = {
        "event": "request",
        "timestamp": _now_iso(),
        "session": session_name,
        "request_id": request_id,
        **request_data,
    }
    _append_jsonl(Path(session["requests_log"]), entry)
    session["updated_at"] = _now_iso()
    state["updated_at"] = _now_iso()
    flush_state(state)
    return request_id


def record_response(state: dict[str, Any], session_name: str, response_data: dict[str, Any]) -> str:
    session = open_session(state, session_name=session_name)
    request_id = response_data.get("request_id") or uuid.uuid4().hex[:16]
    entry = {
        "event": "response",
        "timestamp": _now_iso(),
        "session": session_name,
        "request_id": request_id,
        **response_data,
    }
    _append_jsonl(Path(session["requests_log"]), entry)
    session["updated_at"] = _now_iso()
    state["updated_at"] = _now_iso()
    flush_state(state)
    return request_id
