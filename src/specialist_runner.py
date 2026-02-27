from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import BaseModel

from cai.sdk.agents import Agent, Runner
from cai.sdk.agents.exceptions import MaxTurnsExceeded

from rev.analysis_agent import build_analysis_agent
from rev.debugger_agent import build_debugger_agent
from rev.ida_analysis_agent import build_ida_analysis_agent
from rev.rev_types import (
    AnalysisInput,
    DebuggerAnalysisInput,
    IDAAnalysisInput,
    RevDynamicSummary,
    RevHypothesisUpdate,
    RevStaticSummary,
)
from rev.utils import int_env, truncate
from solver_types import SolverContext, SurfaceMapperReport, parse_agent_output
from web_agent.surface_mapper_agent import build_surface_mapper_agent


SPECIALIST_REGISTRY: dict[str, Callable[[str], Agent]] = {
    "surface_mapper": build_surface_mapper_agent,
    "ida_analysis": build_ida_analysis_agent,
    "debugger": build_debugger_agent,
    "analysis": build_analysis_agent,
}

SPECIALIST_IO_MAP: dict[str, tuple[type[BaseModel] | None, type[BaseModel]]] = {
    "surface_mapper": (None, SurfaceMapperReport),
    "ida_analysis": (IDAAnalysisInput, RevStaticSummary),
    "debugger": (DebuggerAnalysisInput, RevDynamicSummary),
    "analysis": (AnalysisInput, RevHypothesisUpdate),
}


def _collect_tool_outputs(new_items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in new_items:
        tool_output = getattr(item, "output", None)
        if isinstance(tool_output, dict):
            out.append(tool_output)
            continue
        if isinstance(tool_output, str):
            try:
                parsed = json.loads(tool_output)
            except Exception:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
    return out


def _fallback_report(
    *,
    output_model: type[BaseModel],
    payload: dict[str, Any],
    tool_outputs: list[dict[str, Any]],
    max_turns: int,
) -> dict[str, Any]:
    if output_model is RevStaticSummary:
        facts: list[str] = []
        functions: list[dict[str, str]] = []
        errors: list[str] = []
        seen_fn: set[tuple[str, str]] = set()
        fn_summaries: dict[str, str] = {}
        for item in tool_outputs:
            err = item.get("error")
            if isinstance(err, str) and err.strip():
                errors.append(err.strip())
            tool_name = str(item.get("tool_name", ""))
            key_data = item.get("key_data")
            if not isinstance(key_data, dict):
                continue
            if tool_name == "list_funcs":
                rows = key_data.get("functions")
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        name = str(row.get("name", ""))
                        addr = str(row.get("addr", ""))
                        key = (name, addr)
                        if key in seen_fn:
                            continue
                        seen_fn.add(key)
                        functions.append({"name": name, "addr": addr, "role": "", "function_summary": ""})
            if tool_name == "decompile":
                addr = str(key_data.get("addr", "")).strip()
                if addr:
                    facts.append(f"Decompiled function {addr}.")
                    text = str(key_data.get("decompilation", "")).strip()
                    if text:
                        fn_summaries[addr] = text[:500]
            if tool_name == "disasm":
                addr = str(key_data.get("addr", "")).strip()
                if addr and addr not in fn_summaries:
                    text = str(key_data.get("disassembly", "")).strip()
                    if text:
                        fn_summaries[addr] = text[:500]
        for fn in functions:
            addr = str(fn.get("addr", "")).strip()
            if addr and addr in fn_summaries:
                fn["function_summary"] = fn_summaries[addr]
        if not facts and tool_outputs:
            facts.append("Collected partial static analysis outputs before max turns.")
        report = {
            "summary": f"Partial static analysis returned because max_turns={max_turns} was reached.",
            "key_facts": facts[:20],
            "interesting_functions": functions[:20],
            "errors": errors[:20],
        }
        return output_model.model_validate(report).model_dump()

    if output_model is RevDynamicSummary:
        observations: list[str] = []
        commands: list[str] = []
        errors: list[str] = []
        for item in tool_outputs:
            err = item.get("error")
            if isinstance(err, str) and err.strip():
                errors.append(err.strip())
            key_data = item.get("key_data")
            if isinstance(key_data, dict):
                cmd = key_data.get("command")
                if isinstance(cmd, str) and cmd.strip():
                    commands.append(cmd.strip())
                results = key_data.get("results")
                if isinstance(results, list):
                    for result in results[:3]:
                        text = str(result).strip()
                        if text:
                            observations.append(text)
        report = {
            "summary": f"Partial dynamic analysis returned because max_turns={max_turns} was reached.",
            "key_observations": observations[:20],
            "breakpoints_hit": [],
            "commands_run": commands[:20],
            "errors": errors[:20],
        }
        return output_model.model_validate(report).model_dump()

    if output_model is RevHypothesisUpdate:
        known = payload.get("known_facts")
        known_facts = [str(x) for x in known] if isinstance(known, list) else []
        report = {
            "updated_hypotheses": [],
            "next_action": "Continue analysis with a higher max_turns budget using current findings.",
            "facts": known_facts[:20],
            "requires_script": False,
            "is_done": False,
            "is_blocked": False,
            "stop_reason": "",
        }
        limited = _limit_analysis_output(report)
        return output_model.model_validate(limited).model_dump()

    if output_model is SurfaceMapperReport:
        report = {
            "summary": f"Partial surface mapping returned because max_turns={max_turns} was reached.",
            "base_url": str(payload.get("base_url", "")),
            "endpoints": [],
            "headers_observed": [],
            "tech_stack": [],
            "hypotheses": [],
            "artifact_refs": [],
            "errors": ["max_turns_exceeded"],
        }
        return output_model.model_validate(report).model_dump()

    raise RuntimeError(f"No partial fallback defined for {output_model.__name__}")


def _limit_analysis_output(report: dict[str, Any]) -> dict[str, Any]:
    max_hypotheses = max(1, int_env("REV_ANALYSIS_OUTPUT_MAX_HYPOTHESES", 12))
    max_facts = max(1, int_env("REV_ANALYSIS_OUTPUT_MAX_FACTS", 30))
    max_text = max(64, int_env("REV_ANALYSIS_OUTPUT_MAX_TEXT_CHARS", 800))

    out: dict[str, Any] = {}
    out["next_action"] = truncate(str(report.get("next_action", "")), max_text)
    out["facts"] = [truncate(str(v), max_text) for v in list(report.get("facts", []))[:max_facts]]
    out["requires_script"] = bool(report.get("requires_script", False))
    out["is_done"] = bool(report.get("is_done", False))
    out["is_blocked"] = bool(report.get("is_blocked", False))
    out["stop_reason"] = truncate(str(report.get("stop_reason", "")), max_text)
    try:
        out["confidence_delta"] = float(report.get("confidence_delta", 0.0))
    except Exception:
        out["confidence_delta"] = 0.0

    bounded_hypotheses: list[dict[str, Any]] = []
    for item in list(report.get("updated_hypotheses", []))[:max_hypotheses]:
        if not isinstance(item, dict):
            continue
        score_raw = item.get("score", 0.5)
        try:
            score = float(score_raw)
        except Exception:
            score = 0.5
        score = max(0.0, min(1.0, score))
        bounded_hypotheses.append(
            {
                "type": truncate(str(item.get("type", "")), max_text),
                "score": score,
                "rationale": truncate(str(item.get("rationale", "")), max_text),
                "status": truncate(str(item.get("status", "candidate")), max_text),
            }
        )
    out["updated_hypotheses"] = bounded_hypotheses
    return out


async def run_specialist_agent_tool(
    specialist_name: str,
    agent: Agent,
    ctx: SolverContext,
    task_payload: dict,
) -> dict:
    io_spec = SPECIALIST_IO_MAP.get(specialist_name)
    if io_spec is None:
        raise RuntimeError(f"Unknown specialist IO schema for {specialist_name}")

    input_model, output_model = io_spec
    if input_model is None:
        payload = dict(task_payload)
    else:
        payload = input_model.model_validate(task_payload).model_dump()

    prompt = "Task payload:\n" + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)

    max_turns_raw = payload.get("max_turns", 12)
    try:
        max_turns = int(max_turns_raw)
    except Exception:
        max_turns = 12
    max_turns = max(1, min(max_turns, 50))

    streamed = Runner.run_streamed(
        starting_agent=agent,
        input=prompt,
        context=ctx,
        max_turns=max_turns,
    )

    max_turns_exceeded = False
    try:
        async for _ in streamed.stream_events():
            pass
    except MaxTurnsExceeded:
        max_turns_exceeded = True

    final_output = streamed.final_output
    if final_output is not None:
        try:
            parsed = parse_agent_output(output_model, final_output)
            parsed_dump = parsed.model_dump()
            if output_model is RevHypothesisUpdate:
                parsed_dump = RevHypothesisUpdate.model_validate(_limit_analysis_output(parsed_dump)).model_dump()
            return {
                "status": "partial" if max_turns_exceeded else "ok",
                "specialist": specialist_name,
                "report": parsed_dump,
                "partial_reason": "max_turns_exceeded" if max_turns_exceeded else "",
            }
        except Exception:
            if not max_turns_exceeded:
                raise

    tool_outputs = _collect_tool_outputs(list(getattr(streamed, "new_items", [])))
    partial_report = _fallback_report(
        output_model=output_model,
        payload=payload,
        tool_outputs=tool_outputs,
        max_turns=max_turns,
    )
    return {
        "status": "partial",
        "specialist": specialist_name,
        "report": partial_report,
        "partial_reason": "max_turns_exceeded",
    }
