from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cai.sdk.agents import Agent, ModelSettings, RunContextWrapper, function_tool

from rev.analysis_agent import build_analysis_agent
from rev.debugger_agent import build_debugger_agent
from rev.ida_analysis_agent import build_ida_analysis_agent
from rev.rev_blackboard import fold_single_event_into_state
from rev.rev_events import append_event
from rev.rev_types import (
    AnalysisInput,
    DebuggerAnalysisInput,
    IDAAnalysisInput,
    ManagerCommitEventInput,
    RevDynamicSummary,
    RevHypothesis,
    RevHypothesisUpdate,
    RevManagerOutput,
    RevStaticSummary,
)
from rev.utils import int_env, truncate
from solver_types import SolverContext
from specialist_runner import run_specialist_agent_tool


def _compact_previous(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    static = state.get("last_static")
    if isinstance(static, dict):
        out["static"] = {
            "summary": static.get("summary"),
            "key_facts": list(static.get("key_facts", [])),
            "interesting_functions": list(static.get("interesting_functions", [])),
            "errors": list(static.get("errors", []))[:5],
        }

    dynamic = state.get("last_dynamic")
    if isinstance(dynamic, dict):
        out["dynamic"] = {
            "summary": dynamic.get("summary"),
            "key_observations": list(dynamic.get("key_observations", []))[:8],
            "commands_run": list(dynamic.get("commands_run", []))[:8],
            "errors": list(dynamic.get("errors", []))[:5],
        }

    analysis = state.get("last_analysis")
    if isinstance(analysis, dict):
        out["analysis"] = {
            "next_action": analysis.get("next_action"),
            "facts": list(analysis.get("facts", []))[:8],
            "updated_hypotheses": list(analysis.get("updated_hypotheses", []))[:8],
            "is_done": bool(analysis.get("is_done")),
            "is_blocked": bool(analysis.get("is_blocked")),
            "stop_reason": str(analysis.get("stop_reason") or ""),
        }

    return out


def _blackboard_digest(sctx: SolverContext, max_items: int = 20) -> dict[str, Any]:
    bb = sctx.blackboard_state if isinstance(sctx.blackboard_state, dict) else {}
    function_summaries = bb.get("function_summaries", [])
    if not isinstance(function_summaries, list):
        function_summaries = []
    return {
        "facts": list(bb.get("facts", []))[-max_items:],
        "hypotheses": list(bb.get("hypotheses", []))[-max_items:],
        "attempts": list(bb.get("attempts", []))[-max_items:],
        "static_summaries": list(bb.get("static_summaries", []))[-5:],
        "function_summaries": function_summaries[-max_items:],
        "analysis_progress_state": bb.get("analysis_progress_state", {}),
    }


def _manager_state(sctx: SolverContext, max_steps: int) -> dict[str, Any]:
    st = sctx.runtime.get("rev_manager_state")
    if not isinstance(st, dict):
        st = {
            "max_steps": max(1, int(max_steps)),
            "steps_used": 0,
            "last_static": {},
            "last_dynamic": {},
            "last_analysis": {},
        }
        sctx.runtime["rev_manager_state"] = st
    else:
        st["max_steps"] = max(1, int(st.get("max_steps", max_steps)))
        st["steps_used"] = int(st.get("steps_used", 0))
    return st


def _parse_hypothesis_updates(raw_updates: list[dict]) -> list[RevHypothesis]:
    out: list[RevHypothesis] = []
    for item in raw_updates:
        if not isinstance(item, dict):
            continue
        h_type = str(item.get("type", "")).strip()
        if not h_type:
            continue
        score_raw = str(item.get("score", "0.5")).strip() or "0.5"
        try:
            score = float(score_raw)
        except Exception:
            score = 0.5
        try:
            hyp = RevHypothesis(
                type=h_type,
                score=max(0.0, min(1.0, score)),
                rationale=str(item.get("rationale", "")),
                status=str(item.get("status", "candidate")) or "candidate",
            )
        except Exception:
            continue
        out.append(hyp)
    return out


def _truncate_text_list(items: list[Any], max_items: int, max_chars: int) -> list[str]:
    out: list[str] = []
    for item in items[: max(0, max_items)]:
        text = str(item).strip()
        if text:
            out.append(truncate(text, max_chars))
    return out


def _limit_interesting_functions(items: list[Any], summary_chars: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "name": truncate(str(item.get("name", "")).strip(), 200),
                "addr": truncate(str(item.get("addr", "")).strip(), 120),
                "role": truncate(str(item.get("role", "")).strip(), 300),
                "function_summary": truncate(str(item.get("function_summary", "")).strip(), summary_chars),
            }
        )
    return out


def _flag_format(sctx: SolverContext) -> str:
    challenge = sctx.challenge if isinstance(sctx.challenge, dict) else {}
    return str(challenge.get("flag_format", "") or "")


def build_rev_manager_agent(model: str) -> Agent:
    @function_tool(strict_mode=False)
    async def manager_load_blackboard_digest(
        run_ctx: RunContextWrapper[SolverContext],
        max_items: int = 20,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        sctx = run_ctx.context
        if sctx is None:
            return {"status": "error", "error": "missing solver context"}
        budget = int(max_steps) if max_steps else int_env("REV_MAX_STEPS", 200)
        state = _manager_state(sctx, max_steps=budget)
        remaining = max(0, int(state["max_steps"]) - int(state["steps_used"]))
        return {
            "status": "ok",
            "digest": _blackboard_digest(sctx, max_items=max(1, int(max_items))),
            "latest": _compact_previous(state),
            "steps_used": int(state["steps_used"]),
            "steps_remaining": remaining,
            "max_steps": int(state["max_steps"]),
        }

    @function_tool(strict_mode=False)
    async def invoke_ida_analysis(
        run_ctx: RunContextWrapper[SolverContext],
        task: str,
        focus_functions: list[str] = [],
        known_facts: list[str] = [],
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        sctx = run_ctx.context
        if sctx is None:
            return {"status": "error", "error": "missing solver context"}

        budget = int(max_steps) if max_steps else int_env("REV_MAX_STEPS", 200)
        state = _manager_state(sctx, max_steps=budget)
        if int(state["steps_used"]) >= int(state["max_steps"]):
            return {"status": "error", "error": "budget_exhausted"}

        runtime_binary = str(sctx.runtime.get("binary_path") or "")
        flag_format = _flag_format(sctx)
        digest = _blackboard_digest(sctx, max_items=20)

        try:
            payload = IDAAnalysisInput(
                binary_path=runtime_binary,
                flag_format=flag_format,
                task=str(task or "Perform focused static reverse engineering analysis on this binary."),
                focus_functions=list(focus_functions)[:15],
                known_facts=(list(known_facts)[:15] if known_facts else list(digest.get("facts", []))[:15]),
                previous_findings=_compact_previous(state),
            ).model_dump()
        except ValidationError as e:
            return {
                "status": "error",
                "error": "invalid_input",
                "details": e.errors(),
            }

        out = await run_specialist_agent_tool(
            specialist_name="ida_analysis",
            agent=build_ida_analysis_agent(model),
            ctx=sctx,
            task_payload=payload,
        )
        report = RevStaticSummary.model_validate(out.get("report", {}))

        state["last_static"] = report.model_dump()
        state["steps_used"] = int(state["steps_used"]) + 1

        return {
            "status": "ok",
            "report": report.model_dump(),
            "steps_used": int(state["steps_used"]),
            "steps_remaining": max(0, int(state["max_steps"]) - int(state["steps_used"])),
        }

    @function_tool(strict_mode=False)
    async def invoke_debugger_analysis(
        run_ctx: RunContextWrapper[SolverContext],
        task: str,
        focus_functions: list[str] = [],
        known_facts: list[str] = [],
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        sctx = run_ctx.context
        if sctx is None:
            return {"status": "error", "error": "missing solver context"}

        budget = int(max_steps) if max_steps else int_env("REV_MAX_STEPS", 200)
        state = _manager_state(sctx, max_steps=budget)
        if int(state["steps_used"]) >= int(state["max_steps"]):
            return {"status": "error", "error": "budget_exhausted"}

        runtime_binary = str(sctx.runtime.get("binary_path") or "")
        flag_format = _flag_format(sctx)
        digest = _blackboard_digest(sctx, max_items=20)

        try:
            payload = DebuggerAnalysisInput(
                binary_path=runtime_binary,
                flag_format=flag_format,
                task=str(task or "Run focused dynamic checks to validate current hypotheses."),
                focus_functions=list(focus_functions)[:15],
                known_facts=(list(known_facts)[:15] if known_facts else list(digest.get("facts", []))[:15]),
                previous_findings=_compact_previous(state),
            ).model_dump()
        except ValidationError as e:
            return {
                "status": "error",
                "error": "invalid_input",
                "details": e.errors(),
            }

        out = await run_specialist_agent_tool(
            specialist_name="debugger",
            agent=build_debugger_agent(model),
            ctx=sctx,
            task_payload=payload,
        )
        report = RevDynamicSummary.model_validate(out.get("report", {}))

        state["last_dynamic"] = report.model_dump()
        state["steps_used"] = int(state["steps_used"]) + 1

        return {
            "status": "ok",
            "report": report.model_dump(),
            "steps_used": int(state["steps_used"]),
            "steps_remaining": max(0, int(state["max_steps"]) - int(state["steps_used"])),
        }

    @function_tool(strict_mode=False)
    async def invoke_analysis(
        run_ctx: RunContextWrapper[SolverContext],
        task: str,
        known_facts: list[str] = [],
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        sctx = run_ctx.context
        if sctx is None:
            return {"status": "error", "error": "missing solver context"}

        budget = int(max_steps) if max_steps else int_env("REV_MAX_STEPS", 200)
        state = _manager_state(sctx, max_steps=budget)
        if int(state["steps_used"]) >= int(state["max_steps"]):
            return {"status": "error", "error": "budget_exhausted"}

        runtime_binary = str(sctx.runtime.get("binary_path") or "")
        flag_format = _flag_format(sctx)
        digest = _blackboard_digest(sctx, max_items=20)
        task_max_chars = max(128, int_env("REV_ANALYSIS_TASK_MAX_CHARS", 3000))
        fact_max_chars = max(64, int_env("REV_ANALYSIS_FACT_MAX_CHARS", 1000))
        fn_summary_max_chars = max(128, int_env("REV_ANALYSIS_FUNCTION_SUMMARY_MAX_CHARS", 3000))
        bounded_task = truncate(
            str(task or "Synthesize static and dynamic findings into hypotheses and next action."),
            task_max_chars,
        )
        previous = _compact_previous(state)
        if "static_summary" not in previous and isinstance(state.get("last_static"), dict):
            previous["static_summary"] = state.get("last_static")
        if "dynamic_summary" not in previous and isinstance(state.get("last_dynamic"), dict):
            previous["dynamic_summary"] = state.get("last_dynamic")
        bb = sctx.blackboard_state if isinstance(sctx.blackboard_state, dict) else {}
        bb_static_summaries = bb.get("static_summaries", [])
        bb_function_summaries = bb.get("function_summaries", [])
        bb_facts = bb.get("facts", [])
        static_from_runtime = state.get("last_static")
        if not isinstance(previous.get("static"), dict):
            previous["static"] = {}
        previous_static = previous["static"]
        if isinstance(static_from_runtime, dict):
            previous_static["summary"] = truncate(str(static_from_runtime.get("summary", "") or ""), task_max_chars)
            previous_static["key_facts"] = _truncate_text_list(
                list(static_from_runtime.get("key_facts", [])),
                max_items=20000,
                max_chars=fact_max_chars,
            )
            previous_static["interesting_functions"] = _limit_interesting_functions(
                list(static_from_runtime.get("interesting_functions", [])),
                summary_chars=fn_summary_max_chars,
            )
        else:
            previous_static["summary"] = (
                truncate(str(bb_static_summaries[-1]), task_max_chars)
                if isinstance(bb_static_summaries, list) and bb_static_summaries
                else ""
            )
            previous_static["key_facts"] = (
                _truncate_text_list(list(bb_facts), max_items=20000, max_chars=fact_max_chars)
                if isinstance(bb_facts, list)
                else []
            )
            previous_static["interesting_functions"] = (
                _limit_interesting_functions(list(bb_function_summaries), summary_chars=fn_summary_max_chars)
                if isinstance(bb_function_summaries, list)
                else []
            )
        if not previous_static.get("summary") and not previous_static.get("interesting_functions"):
            previous.setdefault(
                "manager_guidance",
                "No static context is available yet. Invoke IDA analysis before analysis synthesis.",
            )

        try:
            payload = AnalysisInput(
                binary_path=runtime_binary,
                flag_format=flag_format,
                task=bounded_task,
                known_facts=(
                    _truncate_text_list(list(known_facts), max_items=15, max_chars=fact_max_chars)
                    if known_facts
                    else _truncate_text_list(list(digest.get("facts", [])), max_items=15, max_chars=fact_max_chars)
                ),
                previous_findings=previous,
            ).model_dump()
        except ValidationError as e:
            return {
                "status": "error",
                "error": "invalid_input",
                "details": e.errors(),
            }

        out = await run_specialist_agent_tool(
            specialist_name="analysis",
            agent=build_analysis_agent(model),
            ctx=sctx,
            task_payload=payload,
        )
        report = RevHypothesisUpdate.model_validate(out.get("report", {}))

        state["last_analysis"] = report.model_dump()
        state["steps_used"] = int(state["steps_used"]) + 1

        return {
            "status": "ok",
            "report": report.model_dump(),
            "steps_used": int(state["steps_used"]),
            "steps_remaining": max(0, int(state["max_steps"]) - int(state["steps_used"])),
        }

    @function_tool(strict_mode=False)
    async def manager_commit_event(
        run_ctx: RunContextWrapper[SolverContext],
        event_type: str,
        summary: str = "",
        facts: list[str] = [],
        next_action: str = "",
        phase: str = "",
        hypothesis_updates: list[dict] = [],
        errors: list[str] = [],
    ) -> dict[str, Any]:
        sctx = run_ctx.context
        if sctx is None:
            return {"status": "error", "error": "missing solver context"}

        parsed_hyps = _parse_hypothesis_updates(hypothesis_updates)
        try:
            commit = ManagerCommitEventInput(
                event_type=str(event_type).strip(),
                summary=str(summary),
                facts=[str(item) for item in facts][:20],
                next_action=str(next_action),
                phase=str(phase),
                hypothesis_updates=parsed_hyps,
                errors=[str(item) for item in errors][:20],
            )
        except ValidationError as e:
            return {
                "status": "error",
                "error": "invalid_event_input",
                "details": e.errors(),
            }

        if not commit.event_type:
            return {"status": "error", "error": "event_type is required"}

        payload: dict[str, Any] = {}
        if commit.summary:
            payload["summary"] = commit.summary
        if commit.facts:
            payload["facts"] = commit.facts
        if commit.next_action:
            payload["next_action"] = commit.next_action
        if commit.phase:
            payload["phase"] = commit.phase
            payload["analysis_progress_state"] = {"phase": commit.phase}
        if commit.hypothesis_updates:
            payload["hypothesis_updates"] = [item.model_dump() for item in commit.hypothesis_updates]
        if commit.errors:
            payload["errors"] = commit.errors

        state = _manager_state(sctx, max_steps=int_env("REV_MAX_STEPS", 200))
        if "ida" in commit.event_type.lower():
            last_static = state.get("last_static")
            if isinstance(last_static, dict):
                static_summary = str(last_static.get("summary", "")).strip()
                if static_summary:
                    payload["static_summary"] = static_summary
                static_key_facts = last_static.get("key_facts", [])
                if isinstance(static_key_facts, list) and static_key_facts:
                    payload["static_key_facts"] = [str(item) for item in static_key_facts[:20]]
                interesting_functions = last_static.get("interesting_functions", [])
                if isinstance(interesting_functions, list) and interesting_functions:
                    payload["interesting_functions"] = interesting_functions

        rec = append_event(
            sctx,
            actor="rev_manager",
            event_type=commit.event_type,
            payload=payload,
        )
        fold_single_event_into_state(sctx, rec)

        return {
            "status": "ok",
            "event_id": rec.event_id,
            "step_index": rec.step_index,
            "blackboard_path": str(Path(sctx.blackboard_path).resolve()),
        }

    instructions = (
        "You are RevManager, the orchestrator of a reverse engineering system designed to solve CTF REV challenges and find the flag. You do not analyze binaries yourself.\n"
        "You are the only agent with full visibility of the blackboard. Every other agent is stateless and sees only what you give them. You are responsible for:\n"
        "- Deciding which specialist agent runs next\n"
        "- Deciding exactly what context to pass each agent (minimum necessary)\n"
        "- Updating the blackboard after each agent returns.\n"
        "Workflow:\n"
        "- Run ida analysis agent when:\n"
        "  - static context is missing or stale\n"
        "  - a new function needs to be understood\n"
        "  - a hypothesis requires structural validation\n"
        "- Run analysis agent when:\n"
        "  - You believe you have enough information to solve challenge. Provide it with the key facts and all important context.\n"
        "- After each agent completes, update the blackboard with new findings and hypotheses.\n"
        "- Repeat this loop until the challenge is solved (flag is found), blocked, or budget is exhausted.\n"
        "Rules:\n"
        "- Keep calls compact; do not include raw dumps in events.\n"
        "- Never rely on artifact IDs/paths for reasoning; use structured summaries.\n"
        "- If blocked, set status='error' and failure_reason='blocked'.\n"
        "- If budget exhausted, set status='error' and failure_reason='budget_exhausted'.\n"
        "- There is only one file unless otherwise provided in context."
    )

    return Agent(
        name="RevManager",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(parallel_tool_calls=False),
        tools=[
            invoke_ida_analysis,
            invoke_analysis,
            manager_commit_event,
            manager_load_blackboard_digest,
        ],
        output_type=RevManagerOutput,
        tool_use_behavior="run_llm_again",
    )
