from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cai.sdk.agents import Agent, ModelSettings, RunContextWrapper, function_tool

from blackboard import append_event, merge_surface_report, persist_blackboard
from solver_types import SolverContext, SurfaceMapperReport, WebManagerOutput
from specialist_runner import SPECIALIST_REGISTRY, run_specialist_agent_tool


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-") or "checkpoint"


def build_web_manager_agent(model: str, ctx: SolverContext) -> Agent:
    @function_tool(strict_mode=False)
    async def invoke_surface_mapper(
        run_ctx: RunContextWrapper[SolverContext],
        task_json: Any = None,
        base_url: str = "",
        session_name: str = "default",
        seed_paths: list[str] | None = None,
        objective: str = "",
    ) -> dict[str, Any]:
        manager_ctx = run_ctx.context
        if manager_ctx is None:
            return {"status": "error", "error": "missing solver context"}

        if task_json is None:
            task_payload = {}
        elif isinstance(task_json, str):
            try:
                task_payload = json.loads(task_json)
            except json.JSONDecodeError:
                task_payload = {}
        elif isinstance(task_json, dict):
            task_payload = dict(task_json)
        else:
            task_payload = {}
        if not isinstance(task_payload, dict):
            task_payload = {}
        if base_url:
            task_payload["base_url"] = base_url
        if session_name:
            task_payload["session_name"] = session_name
        if seed_paths:
            task_payload["seed_paths"] = seed_paths
        if objective:
            task_payload["objective"] = objective
        builder = SPECIALIST_REGISTRY.get("surface_mapper")
        if builder is None:
            return {"status": "error", "error": "surface_mapper not registered"}

        surface_agent = builder(model)
        output = await run_specialist_agent_tool(
            specialist_name="surface_mapper",
            agent=surface_agent,
            ctx=manager_ctx,
            task_payload=task_payload,
        )

        append_event(
            manager_ctx.blackboard_state,
            actor="web_manager",
            action="invoke_surface_mapper",
            details={
                "task": task_payload,
                "status": output.get("status"),
            },
        )
        manager_ctx.step_counter += 1
        persist_blackboard(manager_ctx)

        return output

    @function_tool(strict_mode=False)
    async def manager_commit_surface_report(
        run_ctx: RunContextWrapper[SolverContext],
        report_json: Any = None,
    ) -> dict[str, Any]:
        manager_ctx = run_ctx.context
        if manager_ctx is None:
            return {"status": "error", "error": "missing solver context"}

        if isinstance(report_json, str) and report_json:
            parsed = SurfaceMapperReport.model_validate_json(report_json)
        elif isinstance(report_json, dict):
            parsed = SurfaceMapperReport.model_validate(report_json)
        else:
            return {"status": "error", "error": "report_json or report is required"}
        merge_surface_report(manager_ctx.blackboard_state, parsed)

        report_path = Path(manager_ctx.surface_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(parsed.model_dump(), indent=2), encoding="utf-8")

        append_event(
            manager_ctx.blackboard_state,
            actor="web_manager",
            action="commit_surface_report",
            details={
                "endpoint_count": len(parsed.endpoints),
                "hypothesis_count": len(parsed.hypotheses),
                "error_count": len(parsed.errors),
                "surface_report_path": str(report_path.resolve()),
            },
        )
        manager_ctx.step_counter += 1
        persist_blackboard(manager_ctx)

        return {
            "status": "ok",
            "endpoint_count": len(parsed.endpoints),
            "hypothesis_count": len(parsed.hypotheses),
            "surface_report_path": str(report_path.resolve()),
        }

    @function_tool(strict_mode=False)
    async def manager_checkpoint(
        run_ctx: RunContextWrapper[SolverContext],
        note: str = "checkpoint",
    ) -> str:
        manager_ctx = run_ctx.context
        if manager_ctx is None:
            return "error: missing solver context"

        checkpoint_dir = Path(manager_ctx.docs_dir) / "blackboard_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        manager_ctx.step_counter += 1
        append_event(
            manager_ctx.blackboard_state,
            actor="web_manager",
            action="checkpoint",
            details={"note": note},
        )
        persist_blackboard(manager_ctx)

        checkpoint_path = checkpoint_dir / f"{manager_ctx.step_counter:04d}_{_slug(note)}.json"
        checkpoint_path.write_text(
            json.dumps(manager_ctx.blackboard_state, indent=2),
            encoding="utf-8",
        )
        return str(checkpoint_path.resolve())

    instructions = (
        "You are WebManager, the orchestrator and only blackboard state owner. "
        "You must call tools in this order: "
        "1) manager_checkpoint(note='start'); "
        "2) invoke_surface_mapper(task_json=JSON string with base_url/session_name/seed_paths/objective); "
        "3) manager_commit_surface_report(report_json=<exact JSON from invoke_surface_mapper.report_json>); "
        "4) manager_checkpoint(note='after-surface-mapper'). "
        "After tools, return strict JSON matching WebManagerOutput. "
        "Never invent tool output."
    )

    return Agent(
        name="WebManager",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(temperature=0),
        tools=[
            invoke_surface_mapper,
            manager_commit_surface_report,
            manager_checkpoint,
        ],
        output_type=WebManagerOutput,
    )
