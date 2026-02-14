from __future__ import annotations

from cai.sdk.agents import Agent, ModelSettings

from solver_types import SurfaceMapperReport
from web_agent.web_recon_tools import (
    http_request_with_session,
    load_http_response,
    wappalyzer_lookup,
)


def build_surface_mapper_agent(model: str) -> Agent:
    instructions = (
        "SurfaceMapper.\n"
        "Goal: perform a crawl style recon to map attack surface (no brute-force/wordlists).\n"
        "Rules:\n"
        "- Maintain a frontier (queue/stack) and visited set; avoid repeats.\n"
        "- Use http_request_with_session. Its tool output contains parsed headers plus body_parser.\n"
        "- Raw headers/body are stored in artifacts; use load_http_response(request_id) only when needed.\n"
        "- Decide what to explore next based on evidence (tech stack, cookies, redirects, forms, errors, APIs).\n"
        "- Call wappalyzer_lookup once early on base_url to know the technology stack.\n"
        "- Never paste full bodies into the report; use response_artifact_id (request_id) as evidence pointer.\n"
        "Return ONLY SurfaceMapperReport JSON."
    )

    return Agent(
        name="SurfaceMapper",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(parallel_tool_calls=False),
        tools=[
            http_request_with_session,
            load_http_response,
            wappalyzer_lookup,
        ],
        output_type=SurfaceMapperReport,
    )
