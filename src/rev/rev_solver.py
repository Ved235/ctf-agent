from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from cai.sdk.agents import Runner

from rev.rev_blackboard import fold_single_event_into_state, init_rev_blackboard, rebuild_blackboard
from rev.rev_events import append_event, ensure_event_store
from rev.rev_manager_agent import build_rev_manager_agent
from rev.rev_mcp_runtime import bootstrap_loader, cleanup_rev_runtime, init_rev_runtime
from rev.rev_types import RevManagerOutput
from rev.utils import int_env
from solver_types import SolverContext, parse_agent_output

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

RESULT_FILENAME = "agent_result.json"


def _build_context(challenge_ctx: dict[str, Any]) -> SolverContext:
    workspace = Path(challenge_ctx["paths"]["workspace"]).resolve()
    docs_dir = workspace / "docs" / "rev"
    artifacts_dir = workspace / "artifacts"
    docs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return SolverContext(
        challenge=challenge_ctx["challenge"],
        workspace=str(workspace),
        docs_dir=str(docs_dir),
        artifacts_dir=str(artifacts_dir),
        blackboard_path=str((docs_dir / "blackboard.json").resolve()),
        session_state_path=str((workspace / "docs" / "sessions" / "state.json").resolve()),
        events_path=str((docs_dir / "events.jsonl").resolve()),
    )


def _resolve_binary_path(challenge_ctx: dict[str, Any]) -> str:
    paths = challenge_ctx.get("paths", {})
    source_path = paths.get("source_path") or paths.get("source_dir")
    if not source_path:
        raise RuntimeError("REV source path missing in challenge context paths.source_path")
    binary_path = Path(source_path).resolve()
    if not binary_path.is_file():
        raise RuntimeError(f"REV source is not a file: {binary_path}")
    return str(binary_path)


async def _run_rev_solver_async(challenge_ctx: dict[str, Any]) -> dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv(override=True)

    model = os.environ.get("CAI_MODEL")
    if not model:
        raise RuntimeError("CAI_MODEL is not set.")

    ctx = _build_context(challenge_ctx)
    output_file = Path(ctx.workspace) / RESULT_FILENAME
    runtime_started = False
    tool_map_path = ""

    try:
        binary_path = _resolve_binary_path(challenge_ctx)
        ctx.runtime["binary_path"] = binary_path
        ctx.runtime["binary_basename"] = Path(binary_path).name

        ensure_event_store(ctx)
        init_rev_blackboard(ctx, challenge_ctx)
        if os.environ.get("REV_REBUILD_ON_START", "0").lower() in {"1", "true", "yes"}:
            rebuild_blackboard(ctx, challenge_ctx)

        rec = append_event(
            ctx,
            actor="rev_solver",
            event_type="solver_initialized",
            payload={
                "analysis_progress_state": {"phase": "bootstrap"},
                "binary_metadata": {
                    "binary_name": Path(binary_path).name,
                    "binary_path": binary_path,
                    "size_bytes": Path(binary_path).stat().st_size,
                },
                "attempt": {"stage": "init"},
            },
        )
        fold_single_event_into_state(ctx, rec)

        loader_result = await bootstrap_loader(Path(binary_path).name)
        rec = append_event(
            ctx,
            actor="rev_solver",
            event_type="loader_bootstrap",
            payload={
                "facts": ["loader_bootstrap_ok"],
                "analysis_progress_state": {"phase": "mcp_connect"},
                "attempt": {"stage": "loader_bootstrap", "status_code": loader_result.get("status_code")},
            },
        )
        fold_single_event_into_state(ctx, rec)

        runtime = await init_rev_runtime(ctx, binary_path)
        runtime_started = True
        tool_map_path = runtime.tool_map_path
        rec = append_event(
            ctx,
            actor="rev_solver",
            event_type="mcp_connected",
            payload={
                "facts": ["ida_mcp_connected", "pwndbg_mcp_connected"],
                "analysis_progress_state": {"phase": "manager_run"},
                "binary_metadata": {"binary_name": runtime.binary_basename},
                "attempt": {
                    "stage": "mcp_connected",
                    "ida_tool_count": len(runtime.ida_tools),
                    "dbg_tool_count": len(runtime.dbg_tools),
                    "ida_tools_sample": runtime.ida_tools[:20],
                    "dbg_tools_sample": runtime.dbg_tools[:20],
                    "tool_map_path": runtime.tool_map_path,
                },
            },
        )
        fold_single_event_into_state(ctx, rec)

        manager = build_rev_manager_agent(model=model)
        manager_prompt = (
            f"Run one REV solve cycle for binary_path={binary_path} "
            f"with max_steps={int_env('REV_MAX_STEPS', 200)}."
        )
        run_result = await Runner.run(
            starting_agent=manager,
            input=manager_prompt,
            context=ctx,
            max_turns=int_env("REV_MANAGER_MAX_TURNS", 20),
        )

        manager_output = parse_agent_output(RevManagerOutput, run_result.final_output)

        # Keep canonical paths deterministic even if model returns wrong values.
        manager_output.blackboard_path = str(Path(ctx.blackboard_path).resolve())
        manager_output.events_path = str(Path(ctx.events_path).resolve())
        payload = manager_output.model_dump()
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        exists = output_file.is_file()
        return {
            "status": "ok" if exists else "error",
            "output_file": str(output_file.resolve()),
            "artifact_exists": exists,
            "final_output": manager_output.model_dump_json(),
            "manager_status": manager_output.status,
            "blackboard_path": str(Path(ctx.blackboard_path).resolve()),
            "events_path": str(Path(ctx.events_path).resolve()),
            "mcp_tool_map_path": tool_map_path,
            "failure_reason": manager_output.failure_reason,
        }
    except Exception as e:
        rec = append_event(
            ctx,
            actor="rev_solver",
            event_type="solve_failed",
            payload={
                "analysis_progress_state": {"phase": "failed"},
                "attempt": {"stage": "fatal", "reason": str(e)},
            },
        )
        fold_single_event_into_state(ctx, rec)
        err_payload = {
            "status": "error",
            "summary": str(e),
            "blackboard_path": str(Path(ctx.blackboard_path).resolve()),
            "events_path": str(Path(ctx.events_path).resolve()),
            "next_actions": ["Inspect events and MCP connectivity diagnostics."],
            "verified_flag": None,
            "failure_reason": "runtime_error",
        }
        output_file.write_text(json.dumps(err_payload, indent=2), encoding="utf-8")
        return {
            "status": "error",
            "output_file": str(output_file.resolve()),
            "artifact_exists": output_file.is_file(),
            "final_output": json.dumps(err_payload),
            "manager_status": "error",
            "blackboard_path": str(Path(ctx.blackboard_path).resolve()),
            "events_path": str(Path(ctx.events_path).resolve()),
            "mcp_tool_map_path": tool_map_path,
            "failure_reason": "runtime_error",
        }
    finally:
        if runtime_started:
            try:
                await cleanup_rev_runtime(ctx)
            except Exception:
                pass


def run_rev_solver(challenge_ctx: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(_run_rev_solver_async(challenge_ctx))
