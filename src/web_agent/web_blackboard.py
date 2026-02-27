from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from solver_types import SolverContext, SurfaceMapperReport


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_blackboard(challenge_ctx: dict[str, Any], ctx: SolverContext) -> dict[str, Any]:
    source_dir = challenge_ctx.get("paths", {}).get("source_dir")
    challenge = challenge_ctx.get("challenge", {})
    state = {
        "challenge_metadata": {
            "name": challenge.get("name"),
            "description": challenge.get("description"),
            "category": challenge.get("category"),
            "host": challenge.get("host"),
            "port": challenge.get("port"),
            "flag_format": challenge.get("flag_format"),
            "source_dir": source_dir,
        },
        "artifacts": [],
        "findings": [],
        "hypotheses": [],
        "events": [],
        "state": {
            "status": "initialized",
            "current_phase": "surface_mapping",
            "last_updated_ts": _now_iso(),
            "step_counter": 0,
        },
    }
    ctx.blackboard_state = state
    persist_blackboard(ctx)
    return state


def append_event(state: dict[str, Any], actor: str, action: str, details: dict[str, Any]) -> None:
    events = state.setdefault("events", [])
    events.append(
        {
            "timestamp": _now_iso(),
            "actor": actor,
            "action": action,
            "details": details,
        }
    )
    state.setdefault("state", {})["last_updated_ts"] = _now_iso()


def merge_surface_report(state: dict[str, Any], report: SurfaceMapperReport | dict[str, Any]) -> None:
    parsed = report if isinstance(report, SurfaceMapperReport) else SurfaceMapperReport.model_validate(report)

    artifacts = state.setdefault("artifacts", [])
    existing_artifact_ids = {a.get("id") for a in artifacts}
    for artifact in parsed.artifact_refs:
        if artifact.id not in existing_artifact_ids:
            artifacts.append(artifact.model_dump())
            existing_artifact_ids.add(artifact.id)

    findings = state.setdefault("findings", [])
    existing_endpoint_keys = {
        (f.get("url"), f.get("method"), f.get("response_artifact_id"))
        for f in findings
        if f.get("kind") == "endpoint"
    }
    for endpoint in parsed.endpoints:
        key = (endpoint.url, endpoint.method, endpoint.response_artifact_id)
        if key in existing_endpoint_keys:
            continue
        findings.append(
            {
                "kind": "endpoint",
                "url": endpoint.url,
                "method": endpoint.method,
                "status_code": endpoint.status_code,
                "content_type": endpoint.content_type,
                "response_artifact_id": endpoint.response_artifact_id,
                "discovered_from": endpoint.discovered_from,
            }
        )
        existing_endpoint_keys.add(key)

    for header_item in parsed.headers_observed:
        findings.append({"kind": "headers_observed", **header_item.model_dump()})

    hypotheses = state.setdefault("hypotheses", [])
    existing_hypothesis_titles = {h.get("title") for h in hypotheses}
    for hypothesis in parsed.hypotheses:
        if hypothesis.title in existing_hypothesis_titles:
            continue
        hypotheses.append(hypothesis.model_dump())
        existing_hypothesis_titles.add(hypothesis.title)

    if parsed.errors:
        findings.append({"kind": "errors", "messages": parsed.errors})

    blackboard_state = state.setdefault("state", {})
    blackboard_state["status"] = "surface_mapped"
    blackboard_state["last_updated_ts"] = _now_iso()


def persist_blackboard(ctx: SolverContext) -> None:
    blackboard_path = Path(ctx.blackboard_path)
    blackboard_path.parent.mkdir(parents=True, exist_ok=True)

    ctx.blackboard_state.setdefault("state", {})["step_counter"] = ctx.step_counter
    ctx.blackboard_state.setdefault("state", {})["last_updated_ts"] = _now_iso()

    blackboard_path.write_text(
        json.dumps(ctx.blackboard_state, indent=2),
        encoding="utf-8",
    )
