from __future__ import annotations

import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from cai.sdk.agents import RunContextWrapper, function_tool

from session_manager import (
    get_cookie_jar_path,
    open_session,
    record_request,
    record_response,
)
from solver_types import SolverContext


_HREF_RE = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
_ACTION_RE = re.compile(r"action=[\"']([^\"']+)[\"']", re.IGNORECASE)
_INPUT_RE = re.compile(r"<input[^>]*name=[\"']([^\"']+)[\"']", re.IGNORECASE)


def _append_query(url: str, params: dict[str, Any]) -> str:
    parsed = urlparse(url)
    existing = dict(parse_qsl(parsed.query, keep_blank_values=True))
    merged = {**existing, **{k: str(v) for k, v in params.items()}}
    return urlunparse(parsed._replace(query=urlencode(merged, doseq=True)))


def _parse_headers(header_text: str) -> tuple[str | None, dict[str, str], list[dict[str, Any]]]:
    blocks = [b for b in re.split(r"\r?\n\r?\n", header_text.strip()) if b.strip()]
    parsed_blocks: list[dict[str, Any]] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines or not lines[0].startswith("HTTP/"):
            continue
        block_headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            block_headers[key.strip()] = val.strip()
        parsed_blocks.append({"status_line": lines[0].strip(), "headers": block_headers})

    if not parsed_blocks:
        return None, {}, []

    last = parsed_blocks[-1]
    return last["status_line"], last["headers"], parsed_blocks


def _safe_json_load(text: str, default: Any) -> Any:
    try:
        return json.loads(text) if text else default
    except json.JSONDecodeError:
        return default


@function_tool
async def http_request_with_session(
    ctx: RunContextWrapper[SolverContext],
    session_name: str = "default",
    method: str = "GET",
    url: str = "",
    headers_json: str = "{}",
    params_json: str = "{}",
    data: str = "",
    follow_redirects: bool = True,
    timeout_s: int = 15,
) -> dict[str, Any]:
    if not url:
        return {"status": "error", "error": "url is required"}

    solver_ctx = ctx.context
    if solver_ctx is None:
        return {"status": "error", "error": "missing solver context"}

    session_state = solver_ctx.session_state
    session = open_session(session_state, session_name=session_name)
    cookie_jar = get_cookie_jar_path(session_state, session_name=session_name)

    headers = _safe_json_load(headers_json, {})
    params = _safe_json_load(params_json, {})
    if not isinstance(headers, dict):
        return {"status": "error", "error": "headers_json must decode to object"}
    if not isinstance(params, dict):
        return {"status": "error", "error": "params_json must decode to object"}

    request_id = uuid.uuid4().hex[:16]
    artifacts_dir = Path(solver_ctx.artifacts_dir) / "http" / session_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    header_path = artifacts_dir / f"{request_id}.headers.txt"
    body_path = artifacts_dir / f"{request_id}.body.bin"
    raw_http_path = artifacts_dir / f"{request_id}.http"
    metadata_path = artifacts_dir / f"{request_id}.json"

    final_url = _append_query(url, params) if params else url

    command = [
        "curl",
        "-sS",
        "-X",
        method.upper(),
        "-D",
        str(header_path),
        "-o",
        str(body_path),
        "-b",
        cookie_jar,
        "-c",
        cookie_jar,
        "--max-time",
        str(timeout_s),
        "-w",
        "%{http_code}\\n%{url_effective}\\n%{content_type}",
    ]
    if follow_redirects:
        command.append("-L")
    for key, value in headers.items():
        command.extend(["-H", f"{key}: {value}"])
    if data:
        command.extend(["--data-raw", data])
    command.append(final_url)

    record_request(
        session_state,
        session_name=session_name,
        request_data={
            "request_id": request_id,
            "method": method.upper(),
            "url": url,
            "final_url": final_url,
            "headers": headers,
            "params": params,
            "data": data,
            "follow_redirects": follow_redirects,
            "timeout_s": timeout_s,
            "command": command,
        },
    )

    proc = subprocess.run(command, capture_output=True, text=True)
    stdout_lines = (proc.stdout or "").splitlines()
    status_code = int(stdout_lines[0]) if stdout_lines and stdout_lines[0].isdigit() else None
    url_effective = stdout_lines[1] if len(stdout_lines) > 1 else final_url
    content_type = stdout_lines[2] if len(stdout_lines) > 2 else None

    header_text = header_path.read_text(encoding="utf-8", errors="replace") if header_path.exists() else ""
    body_bytes = body_path.read_bytes() if body_path.exists() else b""

    with raw_http_path.open("wb") as f:
        f.write(header_text.encode("utf-8", errors="replace"))
        f.write(b"\n")
        f.write(body_bytes)

    status_line, parsed_headers, header_blocks = _parse_headers(header_text)
    body_preview = body_bytes.decode("utf-8", errors="replace")[:2000]

    metadata = {
        "request_id": request_id,
        "session_name": session_name,
        "status_code": status_code,
        "status_line": status_line,
        "url_requested": url,
        "url_effective": url_effective,
        "method": method.upper(),
        "content_type": content_type,
        "headers": parsed_headers,
        "header_blocks": header_blocks,
        "body_path": str(body_path.resolve()),
        "raw_http_path": str(raw_http_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "body_length": len(body_bytes),
        "stderr": proc.stderr,
        "return_code": proc.returncode,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    record_response(
        session_state,
        session_name=session_name,
        response_data=metadata,
    )

    error = None
    if proc.returncode != 0:
        error = f"curl exited with {proc.returncode}"

    return {
        "status": "ok" if error is None else "error",
        "error": error,
        "request_id": request_id,
        "response_artifact_path": str(raw_http_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "status_code": status_code,
        "status_line": status_line,
        "headers": parsed_headers,
        "body_preview": body_preview,
        "content_type": content_type,
        "session_log": session["requests_log"],
    }


@function_tool
async def extract_links_from_artifact(
    ctx: RunContextWrapper[SolverContext],
    session_name: str = "default",
    request_id: str = "",
    max_links: int = 1000,
) -> dict[str, Any]:
    if not request_id:
        return {"status": "error", "error": "request_id is required"}

    solver_ctx = ctx.context
    if solver_ctx is None:
        return {"status": "error", "error": "missing solver context"}

    artifact_dir = Path(solver_ctx.artifacts_dir) / "http" / session_name
    metadata_path = artifact_dir / f"{request_id}.json"
    if not metadata_path.exists():
        return {"status": "error", "error": f"metadata not found for request_id={request_id}"}

    metadata = _safe_json_load(metadata_path.read_text(encoding="utf-8"), {})
    body_path = Path(metadata.get("body_path", ""))
    if not body_path.exists():
        return {"status": "error", "error": f"body file missing for request_id={request_id}"}

    body_text = body_path.read_bytes().decode("utf-8", errors="replace")
    links = list(dict.fromkeys(_HREF_RE.findall(body_text)))[:max_links]
    form_actions = list(dict.fromkeys(_ACTION_RE.findall(body_text)))[:max_links]
    input_names = list(dict.fromkeys(_INPUT_RE.findall(body_text)))[:max_links]

    return {
        "status": "ok",
        "request_id": request_id,
        "links": links,
        "form_actions": form_actions,
        "input_names": input_names,
        "counts": {
            "links": len(links),
            "form_actions": len(form_actions),
            "input_names": len(input_names),
        },
    }


@function_tool
async def wappalyzer_lookup(
    ctx: RunContextWrapper[SolverContext],
    url: str,
    session_name: str = "default",
    timeout_s: int = 10,
) -> dict[str, Any]:
    solver_ctx = ctx.context
    if solver_ctx is None:
        return {"status": "error", "error": "missing solver context"}

    if not url:
        return {"status": "error", "error": "url is required"}

    session_state = solver_ctx.session_state
    open_session(session_state, session_name=session_name)

    request_id = uuid.uuid4().hex[:16]
    artifact_dir = Path(solver_ctx.artifacts_dir) / "http" / session_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifact_dir / f"wappalyzer_{request_id}.json"

    endpoint = f"http://localhost:3000/extract?url={url}"
    command = ["curl", "-sS", "--max-time", str(timeout_s), endpoint]

    record_request(
        session_state,
        session_name=session_name,
        request_data={
            "request_id": request_id,
            "method": "GET",
            "url": endpoint,
            "tool": "wappalyzer_lookup",
            "command": command,
        },
    )

    proc = subprocess.run(command, capture_output=True, text=True)
    raw = proc.stdout or ""
    output_path.write_text(raw, encoding="utf-8")

    parsed = _safe_json_load(raw, {})
    tech_stack: list[str] = []
    if isinstance(parsed, dict):
        techs = parsed.get("technologies") or parsed.get("tech") or parsed.get("stack")
        if isinstance(techs, list):
            for item in techs:
                if isinstance(item, dict) and item.get("name"):
                    tech_stack.append(str(item["name"]))
                elif isinstance(item, str):
                    tech_stack.append(item)
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                tech_stack.append(item)
            elif isinstance(item, dict) and item.get("name"):
                tech_stack.append(str(item["name"]))

    record_response(
        session_state,
        session_name=session_name,
        response_data={
            "request_id": request_id,
            "tool": "wappalyzer_lookup",
            "return_code": proc.returncode,
            "stderr": proc.stderr,
            "raw_path": str(output_path.resolve()),
            "tech_stack": tech_stack,
        },
    )

    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "request_id": request_id,
        "raw_path": str(output_path.resolve()),
        "tech_stack": tech_stack,
        "stderr": proc.stderr,
    }
