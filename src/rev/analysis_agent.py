from __future__ import annotations

from cai.sdk.agents import Agent, ModelSettings

from rev.rev_types import RevHypothesisUpdate
from rev.rev_tools import ida_py_eval

def build_analysis_agent(model: str) -> Agent:
    instructions = (
        "You are analysis agent. You receive structured findings from static analysis and construct a complete, actionable model of the binary's logic.\n"
        "The goal of your analysis is to find the flag to solve the CTF challenge. CTF use a lot of different techniques to obfuscate and hide flags. ROT, XOR, custom virtual machines these are just few examples.\n"
        "Use flag_format from task payload when available and align candidate reasoning with that format.\n"
        "Use previous_findings.static.summary and previous_findings.static.interesting_functions as your primary context.\n"
        "Only use ida_py_eval tool when absolute necessary to run python code inside ida.\n"
        "Return only RevHypothesisUpdate JSON.\n"
        "Set is_done=true only when analysis is sufficient for manager completion.\n"
        "Set is_blocked=true when you cannot proceed without missing prerequisites.\n"
        "When is_done or is_blocked, provide a concrete stop_reason.\n"
        "Scores must be calibrated [0,1]."
    )
    return Agent(
        name="AnalysisAgent",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(),
        tools=[ida_py_eval],
        output_type=RevHypothesisUpdate,
    )
