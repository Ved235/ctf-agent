from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote_plus, urlencode, urldefrag, urljoin, urlparse, urlunparse

from cai.sdk.agents import RunContextWrapper, function_tool

from web_agent.session_manager import (
    get_cookie_jar_path,
    open_session,
    record_request,
    record_response,
)
from solver_types import SolverContext


class _DiscoveryHTMLParser(HTMLParser):
    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.links: list[str] = []
        self.scripts: list[str] = []
        self.forms: list[dict[str, Any]] = []
        self._current_form: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}

        def _abs(u: str) -> str | None:
            if not u:
                return None
            u = u.strip()
            if u.startswith("javascript:") or u.startswith("mailto:") or u.startswith("tel:"):
                return None
            abs_u = urljoin(self.page_url, u)
            abs_u, _frag = urldefrag(abs_u)
            return abs_u

        if tag in {"a", "link"}:
            u = _abs(attrs_dict.get("href", ""))
            if u:
                self.links.append(u)
        elif tag in {"script", "img"}:
            u = _abs(attrs_dict.get("src", ""))
            if u:
                self.scripts.append(u)
        elif tag == "form":
            action = _abs(attrs_dict.get("action", "")) or self.page_url
            method = (attrs_dict.get("method") or "GET").upper()
            self._current_form = {"method": method, "action": action, "input_names": []}
        elif tag == "input" and self._current_form is not None:
            name = attrs_dict.get("name", "").strip()
            if name:
                self._current_form["input_names"].append(name)

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._current_form is not None:
            self.forms.append(self._current_form)
            self._current_form = None


def _cap_unique(items: list[str], cap: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
        if len(out) >= cap:
            break
    return out


def _discover_from_html(body_text: str, page_url: str) -> dict[str, Any]:
    parser = _DiscoveryHTMLParser(page_url=page_url)
    try:
        parser.feed(body_text)
    except Exception:
        pass

    links = _cap_unique(parser.links, _get_int_env("CSAWAI_DISCOVER_LINKS_CAP", 200))
    scripts = _cap_unique(parser.scripts, _get_int_env("CSAWAI_DISCOVER_SCRIPTS_CAP", 100))
    forms_cap = _get_int_env("CSAWAI_DISCOVER_FORMS_CAP", 50)
    forms: list[dict[str, Any]] = []
    for f in parser.forms:
        if not isinstance(f, dict):
            continue
        forms.append(
            {
                "method": str(f.get("method", "GET")).upper(),
                "action": str(f.get("action", page_url)),
                "input_names": list(dict.fromkeys([str(x) for x in (f.get("input_names") or [])]))[:200],
            }
        )
        if len(forms) >= forms_cap:
            break

    return {"links": links, "scripts": scripts, "forms": forms}


def _derive_hints(headers: dict[str, str], body_text: str) -> list[str]:
    hints: list[str] = []

    def add(msg: str) -> None:
        if msg and msg not in hints and len(hints) < _get_int_env("CSAWAI_DISCOVER_HINTS_CAP", 50):
            hints.append(msg)

    lower_to_orig: dict[str, str] = {}
    for hk in headers.keys():
        lower_to_orig.setdefault(hk.lower(), hk)
    for key in ("server", "x-powered-by", "set-cookie", "location", "www-authenticate", "content-security-policy"):
        orig = lower_to_orig.get(key)
        if orig:
            add(f"header:{orig}={str(headers.get(orig, ''))[:200]}")

    lowered = body_text.lower()
    for needle, label in (
        ("csrf", "body:csrf"),
        ("login", "body:login"),
        ("password", "body:password"),
        ("admin", "body:admin"),
        ("traceback", "body:traceback"),
        ("exception", "body:exception"),
        ("stack trace", "body:stack-trace"),
        ("fatal error", "body:fatal-error"),
        ("sql", "body:sql"),
        ("jwt", "body:jwt"),
        ("session", "body:session"),
        ("upload", "body:upload"),
    ):
        if needle in lowered:
            add(label)

    return hints


def _parse_body_parser(
    *,
    content_type: str | None,
    headers: dict[str, str],
    body_text: str,
    page_url: str,
) -> dict[str, Any]:
    """
    Return a structured, token-efficient summary of the response body.
    This is intended for LLM decision-making during crawling, not for full fidelity inspection.
    """
    parsed: dict[str, Any] = {
        "kind": "unknown",
        "links": [],
        "scripts": [],
        "forms": [],
        "hints": [],
    }
    ct = (content_type or "").lower()

    # JSON
    if "application/json" in ct or ct.endswith("+json"):
        parsed["kind"] = "json"
        try:
            obj = json.loads(body_text)
            if isinstance(obj, dict):
                keys = list(obj.keys())
                parsed["top_level"] = "object"
                parsed["keys"] = keys[:200]
            elif isinstance(obj, list):
                parsed["top_level"] = "array"
                parsed["length"] = len(obj)
                # Keep only a tiny sample
                parsed["sample"] = obj[:3]
            else:
                parsed["top_level"] = type(obj).__name__
        except Exception as e:
            parsed["error"] = f"json_parse_failed: {e}"
        parsed["hints"] = _derive_hints(headers, body_text)
        return parsed

    # HTML
    if "text/html" in ct or "<html" in body_text.lower():
        parsed["kind"] = "html"
        title = ""
        # Very small extraction: title tag + visible-text excerpt
        m = re.search(r"<title[^>]*>(.*?)</title>", body_text, re.IGNORECASE | re.DOTALL)
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()
        parsed["title"] = title

        # Strip script/style blocks for a visible-text excerpt
        cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", body_text)
        cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
        cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        excerpt_len = _get_int_env("CSAWAI_PARSED_BODY_TEXT_EXCERPT_CHARS", 600)
        parsed["text_excerpt"] = cleaned[:excerpt_len]
        parsed["text_len"] = len(cleaned)
        try:
            parsed.update(_discover_from_html(body_text, page_url=page_url))
        except Exception:
            pass
        parsed["hints"] = _derive_hints(headers, body_text)
        return parsed

    # Form-urlencoded
    if "application/x-www-form-urlencoded" in ct:
        parsed["kind"] = "form_urlencoded"
        try:
            parsed["pairs"] = list(parse_qsl(body_text, keep_blank_values=True))[:200]
        except Exception as e:
            parsed["error"] = f"form_parse_failed: {e}"
        parsed["hints"] = _derive_hints(headers, body_text)
        return parsed

    # Script files (JS, PHP, etc.)
    script_exts = (".js", ".php", ".py", ".rb", ".pl", ".sh", ".ts", ".jsx", ".tsx")
    parsed_url = urlparse(page_url)
    path = parsed_url.path.lower()
    if any(path.endswith(ext) for ext in script_exts):
        parsed["kind"] = "script"
        excerpt_len = _get_int_env("CSAWAI_PARSED_BODY_TEXT_EXCERPT_CHARS", 600)
        parsed["excerpt"] = body_text[:excerpt_len]
        parsed["text_len"] = len(body_text)
        parsed["hints"] = _derive_hints(headers, body_text)
        return parsed

    # Plain text
    if ct.startswith("text/") or "charset=" in ct:
        parsed["kind"] = "text"
        excerpt_len = _get_int_env("CSAWAI_PARSED_BODY_TEXT_EXCERPT_CHARS", 600)
        parsed["excerpt"] = body_text[:excerpt_len]
        parsed["text_len"] = len(body_text)
        parsed["hints"] = _derive_hints(headers, body_text)
        return parsed

    # Fallback: unknown/binary-ish
    parsed["kind"] = "binary_or_unknown"
    parsed["hints"] = _derive_hints(headers, body_text)
    return parsed

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


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _truncate(s: str, n: int) -> str:
    if n <= 0:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + "...<truncated>"


@function_tool(strict_mode=False)
async def http_request_with_session(
    ctx: RunContextWrapper[SolverContext],
    session_name: str = "default",
    method: str = "GET",
    url: str = "",
    headers_json: dict[str, str] = {},
    params_json: dict[str, str] = {},
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
    open_session(session_state, session_name=session_name)
    cookie_jar = get_cookie_jar_path(session_state, session_name=session_name)

    # NOTE: defaults are empty dicts to keep OpenAI function schemas strict (no anyOf/null).
    # We never mutate these input dicts; we copy them.
    headers = dict(headers_json or {})
    params = dict(params_json or {})
    # headers_json/params_json are typed as dicts for tool-schema validity; keep defensive checks anyway.
    if not isinstance(headers, dict):
        return {"status": "error", "error": "headers_json must be an object"}
    if not isinstance(params, dict):
        return {"status": "error", "error": "params_json must be an object"}

    request_id = uuid.uuid4().hex[:16]
    artifacts_dir = Path(solver_ctx.artifacts_dir) / "http" / session_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Single artifact per HTTP request: JSON containing raw headers + full body.
    artifact_path = artifacts_dir / f"{request_id}.json"

    final_url = _append_query(url, params) if params else url

    # Use temp files for curl output then collapse into a single JSON artifact.
    hdr_tmp = tempfile.NamedTemporaryFile(delete=False)
    body_tmp = tempfile.NamedTemporaryFile(delete=False)
    hdr_tmp_path = hdr_tmp.name
    body_tmp_path = body_tmp.name
    hdr_tmp.close()
    body_tmp.close()

    command = [
        "curl",
        "-sS",
        "--compressed",
        "-X",
        method.upper(),
        "-D",
        hdr_tmp_path,
        "-o",
        body_tmp_path,
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
        },
    )

    proc = subprocess.run(command, capture_output=True, text=True)
    stdout_lines = (proc.stdout or "").splitlines()
    status_code = int(stdout_lines[0]) if stdout_lines and stdout_lines[0].isdigit() else None
    url_effective = stdout_lines[1] if len(stdout_lines) > 1 else final_url
    content_type = stdout_lines[2] if len(stdout_lines) > 2 else None

    try:
        header_text = Path(hdr_tmp_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        header_text = ""
    try:
        body_bytes = Path(body_tmp_path).read_bytes()
    except Exception:
        body_bytes = b""
    try:
        Path(hdr_tmp_path).unlink(missing_ok=True)
        Path(body_tmp_path).unlink(missing_ok=True)
    except Exception:
        pass

    status_line, parsed_headers, _header_blocks = _parse_headers(header_text)

    decoded_body = body_bytes.decode("utf-8", errors="replace")
    store_text_chars = _get_int_env("CSAWAI_HTTP_STORE_BODY_TEXT_CHARS", 200_000)
    stored_body_text = decoded_body if store_text_chars < 0 else _truncate(decoded_body, store_text_chars)
    stored_body_text_truncated = store_text_chars >= 0 and len(decoded_body) > store_text_chars
    body_sha256 = hashlib.sha256(body_bytes).hexdigest()

    artifact_obj = {
        "request": {
            "id": request_id,
            "session_name": session_name,
            "method": method.upper(),
            "url": url,
            "final_url": final_url,
            "headers": headers,
            "params": params,
            "data": data,
            "follow_redirects": follow_redirects,
            "timeout_s": timeout_s,
        },
        "response": {
            "status_code": status_code,
            "status_line": status_line,
            "url_effective": url_effective,
            "content_type": content_type,
            "headers_raw": header_text,
            "headers": parsed_headers,
            "body_length": len(body_bytes),
            "body_sha256": body_sha256,
            "body_text": stored_body_text,
            "body_text_truncated": stored_body_text_truncated,
            "body_b64": base64.b64encode(body_bytes).decode("ascii"),
            "return_code": proc.returncode,
            "stderr": proc.stderr or "",
        },
    }
    artifact_path.write_text(json.dumps(artifact_obj, indent=2), encoding="utf-8")

    record_response(
        session_state,
        session_name=session_name,
        response_data={
            "request_id": request_id,
            "method": method.upper(),
            "url_effective": url_effective,
            "status_code": status_code,
            "content_type": content_type,
            "artifact_path": str(artifact_path.resolve()),
            "body_length": len(body_bytes),
            "return_code": proc.returncode,
            "stderr": proc.stderr or "",
        },
    )

    error = None
    if proc.returncode != 0:
        error = f"curl exited with {proc.returncode}"

    body_parser = _parse_body_parser(
        content_type=content_type,
        headers=parsed_headers,
        body_text=stored_body_text,
        page_url=url_effective,
    )

    return {
        "status": "ok" if error is None else "error",
        "error": error,
        "request_id": request_id,
        "status_code": status_code,
        "status_line": status_line,
        "headers": parsed_headers,
        "body_length": len(body_bytes),
        "content_type": content_type,
        "url_effective": url_effective,
        "body_parser": body_parser,
        "stderr": _truncate(proc.stderr or "", _get_int_env("CSAWAI_HTTP_RETURN_STDERR_CHARS", 200)),
    }


@function_tool
async def load_http_response(
    ctx: RunContextWrapper[SolverContext],
    session_name: str = "default",
    request_id: str = "",
    body_chars: int = 20_000,
) -> dict[str, Any]:
    if not request_id:
        return {"status": "error", "error": "request_id is required"}

    solver_ctx = ctx.context
    if solver_ctx is None:
        return {"status": "error", "error": "missing solver context"}

    artifact_dir = Path(solver_ctx.artifacts_dir) / "http" / session_name
    artifact_path = artifact_dir / f"{request_id}.json"
    if not artifact_path.exists():
        return {"status": "error", "error": f"artifact not found for request_id={request_id}"}

    artifact_obj = _safe_json_load(artifact_path.read_text(encoding="utf-8"), {})
    resp = artifact_obj.get("response") if isinstance(artifact_obj, dict) else {}
    if not isinstance(resp, dict):
        resp = {}

    headers_raw = resp.get("headers_raw") if isinstance(resp.get("headers_raw"), str) else ""
    headers = resp.get("headers") if isinstance(resp.get("headers"), dict) else {}
    status_code = resp.get("status_code")
    status_line = resp.get("status_line")
    content_type = resp.get("content_type")
    url_effective = resp.get("url_effective")

    body_b64 = resp.get("body_b64", "")
    try:
        body_bytes = base64.b64decode(body_b64) if isinstance(body_b64, str) and body_b64 else b""
    except Exception:
        body_bytes = b""
    decoded_body = body_bytes.decode("utf-8", errors="replace")
    out_body = decoded_body if body_chars < 0 else _truncate(decoded_body, body_chars)
    body_truncated = body_chars >= 0 and len(decoded_body) > body_chars

    return {
        "status": "ok",
        "request_id": request_id,
        "artifact_path": str(artifact_path.resolve()),
        "status_code": status_code,
        "status_line": status_line,
        "content_type": content_type,
        "url_effective": url_effective,
        "headers_raw": headers_raw,
        "headers": headers,
        "body": out_body,
        "body_truncated": body_truncated,
        "body_length": len(body_bytes),
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

    endpoint = f"http://localhost:3000/extract?url={quote_plus(url)}"
    command = ["curl", "-sS", "--max-time", str(timeout_s), endpoint]

    record_request(
        session_state,
        session_name=session_name,
        request_data={
            "request_id": request_id,
            "method": "GET",
            "url": endpoint,
            "tool": "wappalyzer_lookup",
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
