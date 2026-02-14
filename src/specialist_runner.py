from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import BaseModel

from cai.sdk.agents import Agent, Runner

from solver_types import SolverContext, SurfaceMapperReport
from web_agent.surface_mapper_agent import build_surface_mapper_agent


SPECIALIST_REGISTRY: dict[str, Callable[[str], Agent]] = {
    "surface_mapper": build_surface_mapper_agent,
}


async def run_specialist_agent_tool(
    specialist_name: str,
    agent: Agent,
    ctx: SolverContext,
    task_payload: dict[str, Any],
) -> dict[str, Any]:
    # Keep prompts compact: store full evidence in artifacts/session logs, not in LLM context.
    prompt = "Task payload:\n" + json.dumps(task_payload, separators=(",", ":"), ensure_ascii=True)
   
    max_turns_raw = task_payload.get("max_turns", 12)
    try:
        max_turns = int(max_turns_raw)
    except Exception:
        max_turns = 12
    max_turns = max(1, min(max_turns, 50))

    # Keep specialist loops bounded; artifacts hold details.
    result = await Runner.run(
        starting_agent=agent,
        input=prompt,
        context=ctx,
        max_turns=max_turns,
    )

    final_output = result.final_output

    parsed: BaseModel
    if isinstance(final_output, BaseModel):
        parsed = final_output
    else:
        # Currently only surface mapper is implemented.
        expected_cls = SurfaceMapperReport if specialist_name == "surface_mapper" else None
        if expected_cls is None:
            raise RuntimeError(f"Unknown specialist output parser for {specialist_name}")

        if isinstance(final_output, str):
            parsed = expected_cls.model_validate_json(final_output)
        elif isinstance(final_output, dict):
            parsed = expected_cls.model_validate(final_output)
        else:
            parsed = expected_cls.model_validate(final_output)

    payload = parsed.model_dump()
    return {
        "status": "ok",
        "specialist": specialist_name,
        "report": payload,
    }
