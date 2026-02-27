from __future__ import annotations

from cai.sdk.agents import Agent, ModelSettings

from rev.rev_tools import dbg_execute, dbg_set_file
from rev.rev_types import RevDynamicSummary


def build_debugger_agent(model: str) -> Agent:
    instructions = (
        "You are dynamic analysis agent, a specialist in dynamic binary analysis using pwndbg. \n"
        "Execute the exact task provided in task payload. The end goal is to solve the reverse engineering CTF challenge.\n"
        "Use only the pwndbg tools available to you.\n"
        "dbg_execute tool can only run pwndbg and gdb commands (you cannot run normal system commands)\n"
        "Leverage known_facts/previous_findings to run targeted checks. You never explore a binary freely — every invocation has a precise question to answer and specific locations to observe.\n"
        "Return only RevDynamicSummary JSON with concise runtime observations."
    )
    return Agent(
        name="DebuggerAgent",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(parallel_tool_calls=False),
        tools=[dbg_set_file, dbg_execute],
        output_type=RevDynamicSummary,
    )
