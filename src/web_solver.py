from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cai.sdk.agents import Runner

from blackboard import append_event, init_blackboard, persist_blackboard
from session_manager import init_session_store, open_session
from solver_types import SolverContext, WebManagerOutput
from web_manager_agent import build_web_manager_agent

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

RESULT_FILENAME = "agent_result.json"


def _base_url(challenge: dict[str, Any]) -> str:
    host = challenge.get("host")
    port = challenge.get("port")
    if port in (80, "80", None):
        return f"http://{host}"
    return f"http://{host}:{port}"


def _build_context(challenge_ctx: dict[str, Any]) -> SolverContext:
    workspace = Path(challenge_ctx["paths"]["workspace"]).resolve()
    docs_dir = workspace / "docs"
    artifacts_dir = workspace / "artifacts"

    docs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    blackboard_path = docs_dir / "blackboard.json"
    surface_report_path = docs_dir / "surface_report.json"

    return SolverContext(
        challenge=challenge_ctx["challenge"],
        workspace=str(workspace),
        docs_dir=str(docs_dir),
        artifacts_dir=str(artifacts_dir),
        blackboard_path=str(blackboard_path),
        session_state_path=str((docs_dir / "sessions" / "state.json")),
        surface_report_path=str(surface_report_path),
    )


def run_web_solver(challenge_ctx: dict[str, Any]) -> dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv(override=True)

    model = os.environ.get("CAI_MODEL")
    if not model:
        raise RuntimeError("CAI_MODEL is not set.")

    ctx = _build_context(challenge_ctx)

    ctx.session_state = init_session_store(ctx.workspace)
    open_session(ctx.session_state, "default")

    init_blackboard(challenge_ctx, ctx)
    append_event(ctx.blackboard_state, "web_solver", "initialized", {"workspace": ctx.workspace})
    persist_blackboard(ctx)

    manager = build_web_manager_agent(model=model, ctx=ctx)
    base_url = _base_url(ctx.challenge)

    manager_prompt = (
        "Run the surface mapping cycle using your tools and finalize with WebManagerOutput.\n"
        f"Challenge: {json.dumps(ctx.challenge)}\n"
        f"Base URL: {base_url}\n"
        "Use session_name='default' and include seed_paths ['/','/robots.txt','/sitemap.xml']."
    )

    run_result = Runner.run_sync(
        starting_agent=manager,
        input=manager_prompt,
        context=ctx,
        max_turns=12,
    )

    final_output = run_result.final_output
    if isinstance(final_output, WebManagerOutput):
        manager_output = final_output
    elif isinstance(final_output, str):
        manager_output = WebManagerOutput.model_validate_json(final_output)
    elif isinstance(final_output, dict):
        manager_output = WebManagerOutput.model_validate(final_output)
    else:
        manager_output = WebManagerOutput.model_validate(final_output)

    output_file = Path(ctx.workspace) / RESULT_FILENAME
    session_log_path = Path(ctx.docs_dir) / "sessions" / "default" / "requests.jsonl"

    artifact_payload = {
        "status": manager_output.status,
        "summary": manager_output.summary,
        "blackboard_path": manager_output.blackboard_path,
        "surface_report_path": manager_output.surface_report_path,
        "session_log_path": manager_output.session_log_path,
        "next_actions": manager_output.next_actions,
    }
    output_file.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")

    exists = output_file.is_file()
    return {
        "status": "ok" if exists else "error",
        "output_file": str(output_file.resolve()),
        "artifact_exists": exists,
        "final_output": manager_output.model_dump_json(),
        "manager_status": manager_output.status,
        "blackboard_path": str(Path(ctx.blackboard_path).resolve()),
        "session_log_path": str(session_log_path.resolve()),
        "surface_report_path": str(Path(ctx.surface_report_path).resolve()),
    }
