from __future__ import annotations

from cai.sdk.agents import Agent, ModelSettings

from solver_types import SurfaceMapperReport
from web_recon_tools import (
    http_request_with_session,
    load_http_response,
    wappalyzer_lookup,
)


def build_surface_mapper_agent(model: str) -> Agent:
    instructions = (
        "SurfaceMapper.\n"
        "Goal: perform an agentic crawl-style recon to map attack surface (no brute-force/wordlists).\n"
        "Rules:\n"
        "- Maintain a frontier (queue/stack) and visited set; avoid repeats.\n"
        "- Use http_request_with_session. Its tool output contains parsed headers plus body_parser (links/forms/scripts/hints + summaries).\n"
        "- Raw headers/body are stored in artifacts; use load_http_response(request_id) when needed.\n"
        "- Decide what to explore next based on evidence (tech stack, cookies, redirects, forms, errors, APIs).\n"
        "- Call wappalyzer_lookup once early on base_url.\n"
        "- If you need full raw content for a page, call load_http_response(request_id).\n"
        "- Never paste full bodies into the report; use response_artifact_id (request_id) as evidence pointer.\n"
        "Return ONLY SurfaceMapperReport JSON."
    )

    return Agent(
        name="SurfaceMapper",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(temperature=0, parallel_tool_calls=False),
        tools=[
            http_request_with_session,
            load_http_response,
            wappalyzer_lookup,
        ],
        output_type=SurfaceMapperReport,
    )
