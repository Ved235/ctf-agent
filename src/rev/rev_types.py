from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue

JSONValue = JsonValue


class RevBinaryMetadata(BaseModel):
    binary_name: str
    binary_path: str
    size_bytes: int | None = None
    sha256: str | None = None
    arch: str | None = None
    file_type: str | None = None


class RevHypothesis(BaseModel):
    type: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    status: str = "candidate"


class RevHypothesisUpdate(BaseModel):
    updated_hypotheses: list[RevHypothesis] = Field(default_factory=list)
    next_action: str = ""
    confidence_delta: float = 0.0
    facts: list[str] = Field(default_factory=list)
    requires_script: bool = False
    is_done: bool = False
    is_blocked: bool = False
    stop_reason: str = ""


class RevInterestingFunction(BaseModel):
    name: str
    addr: str
    role: str
    function_summary: str


class RevStaticSummary(BaseModel):
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    interesting_functions: list[RevInterestingFunction] = Field(default_factory=list)
    binary_metadata: RevBinaryMetadata | None = None
    errors: list[str] = Field(default_factory=list)


class RevDynamicSummary(BaseModel):
    summary: str
    key_observations: list[str] = Field(default_factory=list)
    breakpoints_hit: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class IDAAnalysisInput(BaseModel):
    binary_path: str
    task: str
    focus_functions: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    previous_findings: dict[str, JSONValue] = Field(default_factory=dict)
    max_turns: int = Field(default=12, ge=1, le=50)


class DebuggerAnalysisInput(BaseModel):
    binary_path: str
    task: str
    focus_functions: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    previous_findings: dict[str, JSONValue] = Field(default_factory=dict)
    max_turns: int = Field(default=12, ge=1, le=50)


class AnalysisInput(BaseModel):
    binary_path: str
    task: str
    known_facts: list[str] = Field(default_factory=list)
    previous_findings: dict[str, JSONValue] = Field(default_factory=dict)
    max_turns: int = Field(default=12, ge=1, le=50)


class ManagerCommitEventInput(BaseModel):
    event_type: str
    summary: str = ""
    facts: list[str] = Field(default_factory=list)
    next_action: str = ""
    phase: str = ""
    hypothesis_updates: list[RevHypothesis] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RevManagerOutput(BaseModel):
    status: str
    summary: str
    blackboard_path: str
    events_path: str
    next_actions: list[str]
    verified_flag: str | None = None
    failure_reason: str | None = None


class RevEventRecord(BaseModel):
    event_id: str
    ts: str
    actor: str
    event_type: str
    payload: dict[str, JSONValue] = Field(default_factory=dict)
    step_index: int


class RevBlackboardState(BaseModel):
    challenge_metadata: dict[str, JSONValue]
    facts: list[str] = Field(default_factory=list)
    hypotheses: list[RevHypothesis] = Field(default_factory=list)
    attempts: list[dict[str, JSONValue]] = Field(default_factory=list)
    static_summaries: list[str] = Field(default_factory=list)
    function_summaries: list[RevInterestingFunction] = Field(default_factory=list)
    binary_metadata: dict[str, JSONValue] = Field(default_factory=dict)
    analysis_progress_state: dict[str, JSONValue] = Field(default_factory=dict)
    confidence_score: float = 0.0
    state: dict[str, JSONValue] = Field(default_factory=dict)
