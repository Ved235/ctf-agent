from __future__ import annotations

from cai.sdk.agents import Agent, ModelSettings

from rev.rev_tools import (
    ida_decompile,
    ida_disassemble,
    ida_find,
    ida_get_bytes,
    ida_list_funcs,
    ida_py_eval,
    ida_rename,
    ida_stack_frame,
)
from rev.rev_types import RevStaticSummary


def build_ida_analysis_agent(model: str) -> Agent:
    instructions = (
        "You are static analysis agent that is a specialist in static binary analysis using IDA Pro.\n"
        "Execute the exact task provided in task payload. The end goal is to find the CTF flag.\n"
        "Use ida_disassemble to get low-level assembly context is needed\n"
        "For ida_find, valid search types are string, immediate, data_ref, code_ref.\n"
        "You do not speculate beyond what the binary's static structure shows you.\n"
        "Return only RevStaticSummary JSON. In the summary include that your findings. e.g: if the find command was ran, but some things were found rest were not, etc.\n"
        "Include: high-level static summary, binary metadata (when identifiable), and interesting_functions entries with name, addr, role, and function_summary.\n"
        "Do not reference artifact IDs or artifact paths."
    )
    return Agent(
        name="IDAAnalysisAgent",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(parallel_tool_calls=False),
        tools=[
            ida_list_funcs,
            ida_find,
            ida_disassemble,
            ida_rename,
            ida_stack_frame,
            ida_get_bytes
        ],
        output_type=RevStaticSummary,
    )
