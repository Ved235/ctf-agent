from __future__ import annotations

from pathlib import Path
from typing import Any

from cai.sdk.agents import Agent, ModelSettings, RunContextWrapper, function_tool

from web_agent.web_blackboard import append_event, merge_surface_report, persist_blackboard
from solver_types import SolverContext, SurfaceMapperReport, WebManagerOutput
from specialist_runner import SPECIALIST_REGISTRY, run_specialist_agent_tool


def _base_url_from_challenge(challenge: dict[str, Any]) -> str:
    host = challenge.get("host")
    port = challenge.get("port")
    if port in (80, "80", None):
        return f"http://{host}"
    return f"http://{host}:{port}"


def build_web_manager_agent(model: str) -> Agent:
    @function_tool(strict_mode=False)
    async def run_surface_mapping_and_commit(
        run_ctx: RunContextWrapper[SolverContext],
        base_url: str = "",
        session_name: str = "default",
        seed_paths: list[str] = [],
        objective: str = "",
        timeout_s: int = 10,
        max_turns: int = 20,
    ) -> dict[str, Any]:
        manager_ctx = run_ctx.context
        if manager_ctx is None:
            return WebManagerOutput(
                status="error",
                summary="missing solver context",
                blackboard_path="",
                session_log_path="",
                next_actions=["Fix solver context wiring."],
            ).model_dump()

        session_log_path = str((Path(manager_ctx.docs_dir) / "sessions" / session_name / "requests.jsonl").resolve())
        builder = SPECIALIST_REGISTRY.get("surface_mapper")
        if builder is None:
            return WebManagerOutput(
                status="error",
                summary="surface_mapper not registered",
                blackboard_path=str(Path(manager_ctx.blackboard_path).resolve()),
                session_log_path=session_log_path,
                next_actions=["Register surface_mapper specialist."],
            ).model_dump()

        if not base_url:
            base_url = _base_url_from_challenge(manager_ctx.challenge)

        # Defaults: start with a few common entrypoints; specialist is free to crawl beyond.
        if not seed_paths:
            seed_paths = ["/", "/robots.txt", "/sitemap.xml"]
        seed_paths = list(seed_paths)

        task_payload = {
            "base_url": base_url,
            "session_name": session_name,
            "seed_paths": seed_paths,
            "objective": objective,
            "timeout_s": timeout_s,
            "max_turns": max_turns,
        }

        surface_agent = builder(model)
        output = await run_specialist_agent_tool(
            specialist_name="surface_mapper",
            agent=surface_agent,
            ctx=manager_ctx,
            task_payload=task_payload,
        )

        report_obj = output.get("report") if isinstance(output, dict) else None
        try:
            report = SurfaceMapperReport.model_validate(report_obj)
        except Exception as e:
            append_event(
                manager_ctx.blackboard_state,
                actor="web_manager",
                action="surface_mapper_invalid_report",
                details={"error": str(e)},
            )
            persist_blackboard(manager_ctx)
            return WebManagerOutput(
                status="error",
                summary="surface mapper returned invalid report",
                blackboard_path=str(Path(manager_ctx.blackboard_path).resolve()),
                session_log_path=session_log_path,
                next_actions=["Inspect specialist output and fix schema adherence."],
            ).model_dump()

        merge_surface_report(manager_ctx.blackboard_state, report)
        append_event(
            manager_ctx.blackboard_state,
            actor="web_manager",
            action="surface_mapping_committed",
            details={
                "base_url": base_url,
                "endpoint_count": len(report.endpoints),
                "hypothesis_count": len(report.hypotheses),
                "error_count": len(report.errors),
            },
        )
        manager_ctx.step_counter += 1
        persist_blackboard(manager_ctx)

        return WebManagerOutput(
            status="completed",
            summary=report.summary,
            blackboard_path=str(Path(manager_ctx.blackboard_path).resolve()),
            session_log_path=session_log_path,
            next_actions=[
                "Use findings/hypotheses in blackboard to proceed to vulnerability testing and exploit dev.",
            ],
        ).model_dump()

    instructions = (
        "You are WebManager, the orchestrator and only blackboard state owner.\n"
        "Call run_surface_mapping_and_commit exactly once.\n"
        "Return ONLY the tool output.\n"
        "Never invent results."
    )

    return Agent(
        name="WebManager",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(),
        tools=[run_surface_mapping_and_commit],
        output_type=WebManagerOutput,
        tool_use_behavior="stop_on_first_tool",
    )
