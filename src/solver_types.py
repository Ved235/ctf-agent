from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class SolverContext:
    challenge: dict[str, Any]
    workspace: str
    docs_dir: str
    artifacts_dir: str
    blackboard_path: str
    blackboard_state: dict[str, Any] = field(default_factory=dict)
    session_state_path: str = ""
    step_counter: int = 0
    session_state: dict[str, Any] = field(default_factory=dict)
    surface_report_path: str = ""


class ArtifactRef(BaseModel):
    id: str
    kind: str
    path: str
    note: str


class SurfaceEndpoint(BaseModel):
    url: str
    method: str
    status_code: int | None = None
    content_type: str | None = None
    response_artifact_id: str | None = None
    discovered_from: str


class Hypothesis(BaseModel):
    title: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: str


class HeaderEntry(BaseModel):
    name: str
    value: str


class HeaderObservation(BaseModel):
    url: str
    status_line: str | None = None
    headers: list[HeaderEntry]


class SurfaceMapperReport(BaseModel):
    summary: str
    base_url: str
    endpoints: list[SurfaceEndpoint]
    headers_observed: list[HeaderObservation]
    tech_stack: list[str]
    hypotheses: list[Hypothesis]
    artifact_refs: list[ArtifactRef]
    errors: list[str]


class WebManagerOutput(BaseModel):
    status: str
    summary: str
    blackboard_path: str
    surface_report_path: str
    session_log_path: str
    next_actions: list[str]
