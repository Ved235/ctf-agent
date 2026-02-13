from __future__ import annotations

from cai.sdk.agents import Agent, ModelSettings

from solver_types import SurfaceMapperReport
from web_recon_tools import (
    extract_links_from_artifact,
    http_request_with_session,
    wappalyzer_lookup,
)


def build_surface_mapper_agent(model: str) -> Agent:
    instructions = (
        "You are SurfaceMapper, a web recon and crawl specialist for CTF challenges."
        " Use your tools to map reachable endpoints and collect response evidence."
        " While crawling if you discover new endpoints, add them to the list of URLs to crawl, but avoid crawling the same URL more than once."
        " Always include observed response headers and artifact references."
        " Use wappalyzer_lookup for stack hints."
        " Return ONLY strict JSON that matches SurfaceMapperReport."
    )

    return Agent(
        name="SurfaceMapper",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(temperature=0),
        tools=[
            http_request_with_session,
            extract_links_from_artifact,
            wappalyzer_lookup,
        ],
        output_type=SurfaceMapperReport,
    )
