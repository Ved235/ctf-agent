import argparse
import json
import os
import re
import shutil
from pathlib import Path

from test_agent import run as run_test_agent
from web_solver import run_web_solver

REQUIRED_FIELDS = ("name", "description", "category", "host", "port", "flag_format")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "challenge"


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Challenge JSON not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Challenge JSON missing required field(s): {missing_list}")
    return data


def _copy_source_if_present(manifest: dict, manifest_path: Path, workspace: Path) -> tuple[str | None, dict]:
    source_value = manifest.get("source")
    if not source_value:
        return None, {"configured": False, "copied": False, "error": None}

    source_path = Path(source_value)
    if not source_path.is_absolute():
        source_path = (manifest_path.parent / source_path).resolve()
    else:
        source_path = source_path.resolve()

    if not source_path.is_dir():
        return None, {
            "configured": True,
            "copied": False,
            "error": f"Configured source path is not a directory: {source_path}",
        }

    target = (workspace / "challenge_source").resolve()
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_path, target)

    return str(target), {"configured": True, "copied": True, "error": None}


def build_workspace_and_context(challenge_json_path: str, workspace_root: str | None = None) -> dict:
    manifest_path = Path(challenge_json_path).resolve()
    manifest = _load_manifest(manifest_path)

    repo_root = Path(__file__).resolve().parents[1]
    root = Path(workspace_root).resolve() if workspace_root else (repo_root / "artifacts").resolve()

    workspace = (root / "workspaces" / str(manifest["category"]) / _slugify(str(manifest["name"]))).resolve()
    scripts_dir = workspace / "scripts"
    docs_dir = workspace / "docs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    source_dir, source_meta = _copy_source_if_present(manifest, manifest_path, workspace)

    return {
        "challenge": {
            "name": manifest["name"],
            "description": manifest["description"],
            "category": manifest["category"],
            "host": manifest["host"],
            "port": manifest["port"],
            "flag_format": manifest["flag_format"],
        },
        "paths": {
            "workspace": str(workspace),
            "scripts_dir": str(scripts_dir),
            "docs_dir": str(docs_dir),
            "source_dir": source_dir,
        },
        "source": source_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a challenge workspace and run the test CAI agent.")
    parser.add_argument("challenge_json_path", help="Path to challenge JSON file (manifest-like schema).")
    parser.add_argument(
        "--workspace-root",
        default=None,
        help="Override workspace root. Default: <repo>/artifacts",
    )
    args = parser.parse_args()

    challenge_ctx = build_workspace_and_context(args.challenge_json_path, args.workspace_root)
    workspace_path = Path(challenge_ctx["paths"]["workspace"]).resolve()
    os.environ["CAI_WORKSPACE_DIR"] = str(workspace_path.parent)
    os.environ["CAI_WORKSPACE"] = workspace_path.name
    print(f"Workspace parent set to: {workspace_path.parent}")
    print(f"Workspace name set to: {workspace_path.name}")
    if challenge_ctx["challenge"]["category"] == "web":
        result = run_web_solver(challenge_ctx)
    else:
        result = run_test_agent(challenge_ctx)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
