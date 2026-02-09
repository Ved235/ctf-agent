import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cai.sdk.agents import Agent, RunContextWrapper, Runner, function_tool
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

RESULT_FILENAME = "agent_result.json"


@dataclass
class AgentContext:
    challenge_ctx: dict[str, Any]


def _workspace_path_from_context(ctx: RunContextWrapper[AgentContext]) -> Path:
    workspace = ctx.context.challenge_ctx["paths"]["workspace"]
    return Path(workspace).resolve()


@function_tool
def list_source_tree(ctx: RunContextWrapper[AgentContext], source_dir: str | None = None) -> dict:
    """
    List entries in the copied challenge source directory.
    Returns a sorted list of relative paths.
    """
    workspace = _workspace_path_from_context(ctx)
    target_value = source_dir or ctx.context.challenge_ctx["paths"].get("source_dir")
    if not target_value:
        return {"ok": False, "entries": [], "reason": "No source directory configured."}

    target = Path(target_value).resolve()
    if not target.is_dir():
        return {"ok": False, "entries": [], "reason": f"Source directory does not exist: {target}"}

    if workspace not in [target, *target.parents]:
        return {"ok": False, "entries": [], "reason": "Source directory is outside workspace."}

    entries = sorted(
        str(path.relative_to(target))
        for path in target.rglob("*")
    )
    return {"ok": True, "entries": entries, "reason": None}


@function_tool
def write_workspace_json(ctx: RunContextWrapper[AgentContext], filename: str, payload_json: str) -> dict:
    """
    Write a JSON payload into the workspace using a fixed output filename.
    """
    if filename != RESULT_FILENAME:
        return {"ok": False, "path": None, "reason": f"Only {RESULT_FILENAME} is allowed."}

    workspace = _workspace_path_from_context(ctx)
    target = (workspace / filename).resolve()
    if workspace not in [target, *target.parents]:
        return {"ok": False, "path": None, "reason": "Write target is outside workspace."}

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        return {"ok": False, "path": None, "reason": f"Invalid JSON payload: {exc}"}

    if not isinstance(payload, dict):
        return {"ok": False, "path": None, "reason": "Payload must be a JSON object."}

    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"ok": True, "path": str(target), "reason": None}


def run(challenge_ctx: dict) -> dict:
    if load_dotenv is not None:
        load_dotenv(override=True)
    model = os.environ.get("CAI_MODEL")
    if not model:
        raise RuntimeError("CAI_MODEL is not set.")

    context = AgentContext(challenge_ctx=challenge_ctx)
    challenge = challenge_ctx["challenge"]
    paths = challenge_ctx["paths"]

    prompt = (
        "Create one JSON artifact for this challenge.\n"
        f"- Workspace: {paths['workspace']}\n"
        f"- Challenge name: {challenge['name']}\n"
        f"- Category: {challenge['category']}\n"
        f"- Source dir: {paths.get('source_dir')}\n\n"
        f"Use tool list_source_tree first, then write_workspace_json with filename '{RESULT_FILENAME}'.\n"
        "For write_workspace_json, pass payload_json as a stringified JSON object.\n"
        "Payload must include: status, challenge_name, workspace, source_listing, notes.\n"
        "Do not create any other files."
    )

    agent = Agent[AgentContext](
        name="CTF Test Agent",
        instructions=(
            "You can only use provided tools. "
            "Always write exactly one output JSON file in the workspace, with no extra files."
        ),
        model=model,
        tools=[write_workspace_json, list_source_tree],
    )

    result = Runner.run_sync(agent, prompt, context=context)

    output_file = str((Path(paths["workspace"]).resolve() / RESULT_FILENAME))
    exists = Path(output_file).is_file()
    return {
        "status": "ok" if exists else "error",
        "output_file": output_file,
        "artifact_exists": exists,
        "final_output": str(result.final_output),
    }
