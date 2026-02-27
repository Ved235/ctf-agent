from __future__ import annotations

import json
from pathlib import Path

from rev.rev_events import read_events
from rev.rev_types import JSONValue, RevBlackboardState, RevEventRecord, RevHypothesis
from solver_types import SolverContext

def _challenge_metadata(ctx: SolverContext, challenge_ctx: dict[str, JSONValue]) -> dict[str, JSONValue]:
    paths = challenge_ctx.get("paths") if isinstance(challenge_ctx.get("paths"), dict) else {}
    challenge = challenge_ctx.get("challenge") if isinstance(challenge_ctx.get("challenge"), dict) else {}
    src_path = paths.get("source_path") or paths.get("source_dir")
    return {
        "name": challenge.get("name"),
        "description": challenge.get("description"),
        "category": challenge.get("category"),
        "host": challenge.get("host"),
        "port": challenge.get("port"),
        "flag_format": challenge.get("flag_format"),
        "source_dir": src_path,
        "workspace": ctx.workspace,
    }


def _new_state(challenge_metadata: dict[str, JSONValue]) -> dict[str, JSONValue]:
    return {
        "challenge_metadata": challenge_metadata,
        "facts": [],
        "hypotheses": [],
        "attempts": [],
        "static_summaries": [],
        "function_summaries": [],
        "binary_metadata": {},
        "analysis_progress_state": {"phase": "init"},
        "confidence_score": 0.0,
        "state": {
            "status": "initialized",
            "step_counter": 0,
            "last_event_id": "",
        },
    }


def apply_event_to_state(state: dict[str, JSONValue], event: RevEventRecord) -> None:
    payload = event.payload or {}

    facts = state.setdefault("facts", [])
    fact_set = {str(item) for item in facts if isinstance(item, str)}
    incoming_facts = payload.get("facts")
    if isinstance(incoming_facts, list):
        for fact in incoming_facts:
            text = str(fact).strip()
            if text and text not in fact_set:
                facts.append(text)
                fact_set.add(text)

    hyps = state.setdefault("hypotheses", [])
    hyp_keys: set[tuple[str, str, str]] = set()
    for item in hyps:
        if isinstance(item, dict):
            hyp_keys.add((str(item.get("type", "")), str(item.get("status", "")), str(item.get("rationale", ""))))

    def _add_hyp(raw_list: object) -> None:
        if not isinstance(raw_list, list):
            return
        for item in raw_list:
            try:
                hyp = RevHypothesis.model_validate(item)
            except Exception:
                continue
            key = (hyp.type, hyp.status, hyp.rationale)
            if key in hyp_keys:
                continue
            hyp_keys.add(key)
            hyps.append(hyp.model_dump())

    _add_hyp(payload.get("hypotheses"))
    _add_hyp(payload.get("hypothesis_updates"))

    static_summaries = state.setdefault("static_summaries", [])
    static_summary = str(payload.get("static_summary", "")).strip()
    if static_summary and static_summary not in static_summaries:
        static_summaries.append(static_summary)

    function_summaries = state.setdefault("function_summaries", [])
    fn_keys: dict[tuple[str, str], int] = {}
    for idx, item in enumerate(function_summaries):
        if isinstance(item, dict):
            key = (str(item.get("name", "")), str(item.get("addr", "")))
            fn_keys[key] = idx
    incoming_functions = payload.get("interesting_functions")
    if isinstance(incoming_functions, list):
        for item in incoming_functions:
            try:
                fn = item if isinstance(item, dict) else {}
                name = str(fn.get("name", "")).strip()
                addr = str(fn.get("addr", "")).strip()
                role = str(fn.get("role", "")).strip()
                function_summary = str(fn.get("function_summary", "")).strip()
            except Exception:
                continue
            if not name and not addr:
                continue
            key = (name, addr)
            if key in fn_keys:
                idx = fn_keys[key]
                existing = function_summaries[idx]
                if not isinstance(existing, dict):
                    continue
                if role:
                    existing["role"] = role
                if function_summary:
                    existing["function_summary"] = function_summary
                continue
            fn_keys[key] = len(function_summaries)
            function_summaries.append(
                {
                    "name": name,
                    "addr": addr,
                    "role": role,
                    "function_summary": function_summary,
                }
            )

    static_key_facts = payload.get("static_key_facts")
    if isinstance(static_key_facts, list):
        for item in static_key_facts:
            text = str(item).strip()
            if text and text not in fact_set:
                facts.append(text)
                fact_set.add(text)

    attempts = state.setdefault("attempts", [])
    attempt = payload.get("attempt")
    if isinstance(attempt, dict):
        compact_attempt: dict[str, JSONValue] = {"event_type": event.event_type}
        for key in ("stage", "status", "reason", "summary", "next_action"):
            if key in attempt:
                compact_attempt[key] = attempt[key]
        if len(compact_attempt) == 1:
            compact_attempt["stage"] = str(attempt.get("stage", event.event_type))
        attempts.append(compact_attempt)
    elif payload.get("summary"):
        attempts.append(
            {
                "event_type": event.event_type,
                "summary": str(payload.get("summary", "")),
                "next_action": str(payload.get("next_action", "")),
            }
        )

    binary_metadata = payload.get("binary_metadata")
    if isinstance(binary_metadata, dict):
        state.setdefault("binary_metadata", {}).update(binary_metadata)

    phase = str(payload.get("phase", "")).strip()
    progress = state.setdefault("analysis_progress_state", {})
    if isinstance(payload.get("analysis_progress_state"), dict):
        progress.update(payload.get("analysis_progress_state"))
    if phase:
        progress["phase"] = phase

    confidence = payload.get("confidence_score")
    if isinstance(confidence, (int, float)):
        state["confidence_score"] = float(confidence)

    st = state.setdefault("state", {})
    st["status"] = "running"
    st["step_counter"] = event.step_index
    st["last_event_id"] = event.event_id


def init_rev_blackboard(ctx: SolverContext, challenge_ctx: dict[str, JSONValue]) -> RevBlackboardState:
    state_dict = _new_state(_challenge_metadata(ctx, challenge_ctx))
    state = RevBlackboardState.model_validate(state_dict)
    _persist(ctx, state)
    ctx.blackboard_state = state.model_dump()
    ctx.step_counter = 0
    return state


def rebuild_blackboard(ctx: SolverContext, challenge_ctx: dict[str, JSONValue]) -> RevBlackboardState:
    state_dict = _new_state(_challenge_metadata(ctx, challenge_ctx))
    events = read_events(ctx)
    for ev in events:
        apply_event_to_state(state_dict, ev)
    state = RevBlackboardState.model_validate(state_dict)
    _persist(ctx, state)
    ctx.blackboard_state = state.model_dump()
    ctx.step_counter = len(events)
    return state


def fold_single_event_into_state(ctx: SolverContext, event: RevEventRecord) -> None:
    bb = ctx.blackboard_state
    if not isinstance(bb, dict):
        bb = _new_state({"workspace": ctx.workspace})
    apply_event_to_state(bb, event)
    state = RevBlackboardState.model_validate(bb)
    ctx.blackboard_state = state.model_dump()
    _persist(ctx, state)


def _persist(ctx: SolverContext, state: RevBlackboardState) -> None:
    bb_path = Path(ctx.blackboard_path)
    bb_path.parent.mkdir(parents=True, exist_ok=True)
    bb_path.write_text(json.dumps(state.model_dump(), indent=2), encoding="utf-8")
