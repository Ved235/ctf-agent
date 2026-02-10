import os
from pathlib import Path

from cai.sdk.agents import Agent, Runner
from cai.tools.reconnaissance.generic_linux_command import generic_linux_command

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

RESULT_FILENAME = "agent_result.json"


def run(challenge_ctx: dict) -> dict:
    if load_dotenv is not None:
        load_dotenv(override=True)

    model = os.environ.get("CAI_MODEL")
    if not model:
        raise RuntimeError("CAI_MODEL is not set.")

    workspace_parent = os.environ.get("CAI_WORKSPACE_DIR")
    workspace_name = os.environ.get("CAI_WORKSPACE")
    if not workspace_parent or not workspace_name:
        raise RuntimeError("CAI_WORKSPACE_DIR and CAI_WORKSPACE must be set before running the agent.")

    challenge = challenge_ctx["challenge"]
    expected_workspace = Path(workspace_parent).resolve() / workspace_name

    prompt = (
        "You have exactly one tool: generic_linux_command.\n"
        "Run shell commands to do all work and create exactly one file named agent_result.json.\n"
        f"Expected CAI workspace path: {expected_workspace}\n"
        f"Challenge name: {challenge['name']}\n"
        f"Challenge category: {challenge['category']}\n\n"
        "Steps:\n"
        "1) Run `pwd` and verify it matches expected workspace.\n"
        "2) If `challenge_source` exists, gather sorted relative entries with find.\n"
        "3) If missing, use empty source_listing and a note saying source is not configured.\n"
        "4) Write agent_result.json using embedded python (`python - <<'PY' ... PY`) with keys:\n"
        "   status, challenge_name, workspace, source_listing, notes\n"
        "5) Return a short completion message."
    )

    agent = Agent(
        name="CTF Test Agent",
        instructions=(
            "Do not invent tool output."
        ),
        model=model,
        tools=[generic_linux_command],
    )

    result = Runner.run_sync(agent, prompt)

    output_file = str((Path(workspace_parent).resolve() / workspace_name / RESULT_FILENAME))
    exists = Path(output_file).is_file()
    return {
        "status": "ok" if exists else "error",
        "output_file": output_file,
        "artifact_exists": exists,
        "final_output": str(result.final_output),
    }
