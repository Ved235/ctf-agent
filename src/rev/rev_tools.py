from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from cai.sdk.agents import RunContextWrapper, function_tool

from rev.rev_mcp_runtime import call_server_tool, get_runtime
from rev.utils import int_env, truncate
from solver_types import SolverContext


def _error(tool_name: str, summary: str, error: str, key_data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "error",
        "tool_name": tool_name,
        "summary": summary,
        "key_data": key_data or {},
        "error": error,
    }


def _ok(tool_name: str, key_data: dict[str, Any] | None = None, summary: str = "ok") -> dict[str, Any]:
    return {
        "status": "ok",
        "tool_name": tool_name,
        "summary": summary,
        "key_data": key_data or {},
        "error": None,
    }


def _artifact_dir(ctx: SolverContext, kind: str) -> Path:
    d = Path(ctx.artifacts_dir) / "rev" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json_artifact(ctx: SolverContext, kind: str, data: dict[str, Any]) -> str:
    path = _artifact_dir(ctx, kind) / f"{uuid.uuid4().hex[:16]}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(path.resolve())


def _schema_for(ctx: SolverContext, server: str, tool_name: str) -> dict[str, Any]:
    runtime = get_runtime(ctx)
    specs = runtime.ida_tool_specs if server == "ida" else runtime.dbg_tool_specs
    for spec in specs:
        if spec.get("name") == tool_name and isinstance(spec.get("inputSchema"), dict):
            return spec["inputSchema"]
    return {}


def _missing_required(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    required = schema.get("required")
    if not isinstance(required, list):
        return []
    return [str(k) for k in required if k not in args]


def _json_any(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None
    return json.loads(text)


def _first_text(value: Any) -> str:
    queue: list[Any] = [value]
    while queue:
        item = queue.pop(0)
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            # Avoid returning plain address tokens as "code".
            if text.startswith("0x") and " " not in text and "\n" not in text and len(text) <= 24:
                continue
            return text
        if isinstance(item, list):
            queue.extend(item)
            continue
        if isinstance(item, dict):
            for key in ("lines", "decompilation", "pseudocode", "code", "text", "output", "result", "asm", "message"):
                v = item.get(key)
                if isinstance(v, str) and v.strip():
                    text = v.strip()
                    if text.startswith("0x") and " " not in text and "\n" not in text and len(text) <= 24:
                        continue
                    return text
                if isinstance(v, (dict, list)):
                    queue.append(v)
    return ""


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _dedupe(items: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _preview_from_result(res: dict[str, Any]) -> list[Any]:
    raw = res.get("content")
    items = _dedupe(_flatten(raw if isinstance(raw, list) else []))
    return items[: max(1, int_env("REV_TOOL_PREVIEW_ITEMS", 8))]


def _extract_error(res: dict[str, Any], preview: list[Any]) -> str:
    for item in preview:
        if not isinstance(item, dict):
            continue
        for key in ("error", "stderr"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return truncate(value.strip(), 240)
    joined = str(res.get("joined_text") or "").strip()
    if res.get("is_error") and joined:
        return truncate(joined.splitlines()[0].strip(), 240)
    return ""


def _compact(value: Any, text_chars: int, list_limit: int = 8) -> Any:
    if isinstance(value, str):
        return truncate(value, text_chars)
    if isinstance(value, list):
        return [_compact(v, text_chars, list_limit) for v in value[:list_limit]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in ("query", "count", "error", "addr", "name", "size", "cursor", "status", "stdout", "stderr", "result"):
            if key in value:
                out[key] = value[key]
        if isinstance(value.get("matches"), list):
            out["matches"] = value["matches"][:list_limit]
        if isinstance(value.get("data"), list):
            out["data"] = value["data"][:list_limit]
        if out:
            return out
        text = _first_text(value)
        return truncate(text, text_chars) if text else value
    return value


def _collect_function_rows(value: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            rows.extend(_collect_function_rows(item))
        return rows
    if isinstance(value, dict):
        name = str(value.get("name") or value.get("func_name") or value.get("function") or "").strip()
        addr = str(
            value.get("addr")
            or value.get("address")
            or value.get("ea")
            or value.get("start_ea")
            or value.get("offset")
            or ""
        ).strip()
        if name or addr:
            size = str(value.get("size") or "").strip()
            if addr:
                rows.append({"name": name, "addr": addr, "size": size})
        for item in value.values():
            rows.extend(_collect_function_rows(item))
    return rows


def _normalize_regions(raw: Any, default_size: int) -> list[dict[str, Any]]:
    queue: list[Any] = raw if isinstance(raw, list) else [raw]
    regions: list[dict[str, Any]] = []
    while queue:
        item = queue.pop(0)
        if item is None:
            continue
        if isinstance(item, dict):
            if item.get("regions") is not None:
                nested = item["regions"]
                if isinstance(nested, list):
                    queue.extend(nested)
                else:
                    queue.append(nested)
                continue
            addr_val = item.get("addr") or item.get("address") or item.get("ea")
            if addr_val is None:
                continue
            size_val = item.get("size") or item.get("len") or item.get("length")
            try:
                size = int(size_val) if size_val is not None else default_size
            except Exception:
                size = default_size
            addr = str(addr_val).strip()
            if addr:
                regions.append({"addr": addr, "size": max(1, size)})
            continue
        addr = str(item).strip()
        if addr:
            regions.append({"addr": addr, "size": max(1, default_size)})
    return _dedupe(regions)


async def _invoke(
    sctx: SolverContext,
    *,
    server: str,
    tool_name: str,
    args: dict[str, Any],
    artifact_kind: str,
) -> tuple[dict[str, Any], list[Any]]:
    schema = _schema_for(sctx, server, tool_name)
    missing = _missing_required(schema, args)
    if missing:
        return _error(tool_name, "invalid_input", f"missing required parameters: {missing}", {"missing_required": missing}), []

    try:
        res = await call_server_tool(sctx, server, tool_name, args)
    except Exception as e:
        return _error(tool_name, "tool_call_failed", truncate(str(e), 240)), []

    _write_json_artifact(
        sctx,
        artifact_kind,
        {"server": server, "tool_name": tool_name, "args": args, "result": res},
    )

    preview = _preview_from_result(res)
    err = _extract_error(res, preview)
    if err:
        return _error(tool_name, "tool_error", err), preview
    return _ok(tool_name), preview


@function_tool(strict_mode=False)
async def ida_list_funcs(
    ctx: RunContextWrapper[SolverContext],
    queries_json: str = "[]",
    limit: int = 200,
) -> dict[str, Any]:
    sctx = ctx.context
    if sctx is None:
        return _error("list_funcs", "missing_context", "missing solver context")

    try:
        queries = _json_any(queries_json)
    except Exception as e:
        return _error("list_funcs", "invalid_input", f"invalid queries_json: {e}")

    envelope, preview = await _invoke(
        sctx,
        server="ida",
        tool_name="list_funcs",
        args={"queries": [] if queries is None else queries},
        artifact_kind="ida_list_funcs",
    )

    rows = _collect_function_rows(preview)
    seen: set[tuple[str, str]] = set()
    unique_rows: list[dict[str, str]] = []
    for row in rows:
        key = (row.get("name", ""), row.get("addr", ""))
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    envelope["key_data"] = {"functions": unique_rows[: max(1, min(int(limit), 500))]}
    return envelope


@function_tool(strict_mode=False)
async def ida_find(
    ctx: RunContextWrapper[SolverContext],
    type: str,
    targets_json: str,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    sctx = ctx.context
    if sctx is None:
        return _error("find", "missing_context", "missing solver context")

    try:
        targets = _json_any(targets_json)
    except Exception as e:
        return _error("find", "invalid_input", f"invalid targets_json: {e}")

    alias = {
        "string": "string",
        "str": "string",
        "name": "string",
        "names": "string",
        "symbol": "string",
        "symbols": "string",
        "import": "string",
        "imports": "string",
        "immediate": "immediate",
        "imm": "immediate",
        "data_ref": "data_ref",
        "dataref": "data_ref",
        "datarefs": "data_ref",
        "code_ref": "code_ref",
        "coderef": "code_ref",
        "code_refs": "code_ref",
        "xref": "code_ref",
        "xrefs": "code_ref",
        "xrefs_to": "code_ref",
    }
    normalized_type = alias.get(type.strip().lower(), type.strip().lower())
    allowed_types = {"string", "immediate", "data_ref", "code_ref"}
    if normalized_type not in allowed_types:
        return _error("find", "invalid_input", f"unsupported search type: {type}", {"allowed_types": sorted(allowed_types)})

    if isinstance(targets, dict):
        for key in ("targets", "queries", "query", "target"):
            if key in targets:
                targets = targets[key]
                break
    if isinstance(targets, list):
        targets = [v for v in targets if v is not None and str(v).strip()]

    if targets is None or targets == "" or (isinstance(targets, list) and not targets):
        return _error("find", "invalid_input", "targets cannot be empty")

    envelope, preview = await _invoke(
        sctx,
        server="ida",
        tool_name="find",
        args={"type": normalized_type, "targets": targets, "limit": int(limit), "offset": int(offset)},
        artifact_kind="ida_find",
    )

    max_chars = max(120, int_env("REV_TOOL_SUMMARY_CHARS", 320))
    envelope["key_data"] = {"results": [_compact(item, max_chars) for item in preview[:8]]}
    return envelope


@function_tool(strict_mode=False)
async def ida_decompile(
    ctx: RunContextWrapper[SolverContext],
    addr: str,
) -> dict[str, Any]:
    sctx = ctx.context
    if sctx is None:
        return _error("decompile", "missing_context", "missing solver context")

    envelope, preview = await _invoke(
        sctx,
        server="ida",
        tool_name="decompile",
        args={"addr": addr},
        artifact_kind="ida_decompile",
    )
    decompilation = _first_text(preview)
    envelope["key_data"] = {"addr": addr, "decompilation": decompilation}
    if not decompilation and envelope.get("status") == "ok":
        envelope["status"] = "error"
        envelope["summary"] = "tool_error"
        envelope["error"] = "decompile returned no code text"
    return envelope


@function_tool(strict_mode=False)
async def ida_disassemble(
    ctx: RunContextWrapper[SolverContext],
    addr: str,
) -> dict[str, Any]:
    sctx = ctx.context
    if sctx is None:
        return _error("disasm", "missing_context", "missing solver context")

    envelope, preview = await _invoke(
        sctx,
        server="ida",
        tool_name="disasm",
        args={"addr": addr},
        artifact_kind="ida_disasm",
    )
    disassembly = _first_text(preview)
    envelope["key_data"] = {"addr": addr, "disassembly": disassembly}
    if not disassembly and envelope.get("status") == "ok":
        envelope["status"] = "error"
        envelope["summary"] = "tool_error"
        envelope["error"] = "disasm returned no assembly text"
    return envelope


@function_tool(strict_mode=False)
async def ida_rename(
    ctx: RunContextWrapper[SolverContext],
    batch_json: str,
) -> dict[str, Any]:
    sctx = ctx.context
    if sctx is None:
        return _error("rename", "missing_context", "missing solver context")

    try:
        batch = _json_any(batch_json)
    except Exception as e:
        return _error("rename", "invalid_input", f"invalid batch_json: {e}")

    envelope, preview = await _invoke(
        sctx,
        server="ida",
        tool_name="rename",
        args={"batch": batch},
        artifact_kind="ida_rename",
    )
    max_chars = max(120, int_env("REV_TOOL_SUMMARY_CHARS", 320))
    envelope["key_data"] = {"results": [_compact(item, max_chars) for item in preview[:6]]}
    return envelope


@function_tool(strict_mode=False)
async def ida_stack_frame(
    ctx: RunContextWrapper[SolverContext],
    addrs_json: str,
) -> dict[str, Any]:
    sctx = ctx.context
    if sctx is None:
        return _error("stack_frame", "missing_context", "missing solver context")

    try:
        addrs = _json_any(addrs_json)
    except Exception as e:
        return _error("stack_frame", "invalid_input", f"invalid addrs_json: {e}")

    envelope, preview = await _invoke(
        sctx,
        server="ida",
        tool_name="stack_frame",
        args={"addrs": addrs},
        artifact_kind="ida_stack_frame",
    )
    max_chars = max(120, int_env("REV_TOOL_SUMMARY_CHARS", 320))
    envelope["key_data"] = {"frames": [_compact(item, max_chars) for item in preview[:6]]}
    return envelope


@function_tool(strict_mode=False)
async def ida_get_bytes(
    ctx: RunContextWrapper[SolverContext],
    regions_json: str,
) -> dict[str, Any]:
    sctx = ctx.context
    if sctx is None:
        return _error("get_bytes", "missing_context", "missing solver context")

    try:
        raw = _json_any(regions_json)
    except Exception as e:
        return _error("get_bytes", "invalid_input", f"invalid regions_json: {e}")

    regions = _normalize_regions(raw, default_size=max(1, int_env("REV_GET_BYTES_DEFAULT_SIZE", 32)))
    if not regions:
        return _error("get_bytes", "invalid_input", "no valid regions; expected [{'addr':'0x...', 'size':N}]")

    envelope, preview = await _invoke(
        sctx,
        server="ida",
        tool_name="get_bytes",
        args={"regions": regions},
        artifact_kind="ida_get_bytes",
    )
    max_chars = max(120, int_env("REV_TOOL_SUMMARY_CHARS", 320))
    envelope["key_data"] = {
        "regions": regions[:8],
        "results": [_compact(item, max_chars) for item in preview[:6]],
    }
    return envelope


@function_tool(strict_mode=False)
async def ida_py_eval(
    ctx: RunContextWrapper[SolverContext],
    code: str,
) -> dict[str, Any]:
    sctx = ctx.context
    if sctx is None:
        return _error("py_eval", "missing_context", "missing solver context")

    envelope, preview = await _invoke(
        sctx,
        server="ida",
        tool_name="py_eval",
        args={"code": code},
        artifact_kind="ida_py_eval",
    )

    max_chars = max(200, int_env("REV_PY_EVAL_RETURN_CHARS", 3000))
    parsed: list[dict[str, Any]] = []
    for item in preview:
        if isinstance(item, dict):
            parsed.append(item)
        elif isinstance(item, str):
            try:
                obj = json.loads(item.strip())
            except Exception:
                continue
            if isinstance(obj, dict):
                parsed.append(obj)

    stderr: list[str] = []
    stdout: list[str] = []
    result: list[str] = []
    for item in parsed:
        if isinstance(item.get("stderr"), str) and item["stderr"].strip():
            stderr.append(item["stderr"].strip())
        if isinstance(item.get("stdout"), str) and item["stdout"].strip():
            stdout.append(item["stdout"].strip())
        if isinstance(item.get("result"), str) and item["result"].strip():
            result.append(item["result"].strip())

    key_data: dict[str, Any] = {}
    if result:
        key_data["result"] = truncate(result[0], max_chars)
    if stdout:
        key_data["stdout"] = truncate(stdout[0], max_chars)
    if stderr:
        key_data["stderr"] = truncate(stderr[0], max_chars)
    if not key_data:
        key_data["results"] = [_compact(item, max_chars) for item in preview[:4]]

    envelope["key_data"] = key_data
    if stderr:
        envelope["status"] = "error"
        envelope["summary"] = "tool_error"
        envelope["error"] = truncate(stderr[0], 240)
    return envelope


@function_tool(strict_mode=False)
async def dbg_set_file(
    ctx: RunContextWrapper[SolverContext],
    binary_path: str = "",
) -> dict[str, Any]:
    sctx = ctx.context
    if sctx is None:
        return _error("set_file", "missing_context", "missing solver context")

    runtime = get_runtime(sctx)
    source_path = binary_path.strip() or runtime.binary_path
    target_path = "/workspace/" + os.path.basename(source_path)

    envelope, preview = await _invoke(
        sctx,
        server="dbg",
        tool_name="set_file",
        args={"binary_path": target_path},
        artifact_kind="dbg_set_file",
    )
    max_chars = max(120, int_env("REV_TOOL_SUMMARY_CHARS", 320))
    envelope["key_data"] = {
        "binary_path": target_path,
        "results": [_compact(item, max_chars) for item in preview[:4]],
    }
    return envelope


@function_tool(strict_mode=False)
async def dbg_execute(
    ctx: RunContextWrapper[SolverContext],
    command: str,
) -> dict[str, Any]:
    sctx = ctx.context
    if sctx is None:
        return _error("execute", "missing_context", "missing solver context")

    envelope, preview = await _invoke(
        sctx,
        server="dbg",
        tool_name="execute",
        args={"command": command},
        artifact_kind="dbg_execute",
    )
    max_chars = max(120, int_env("REV_TOOL_SUMMARY_CHARS", 320))
    envelope["key_data"] = {
        "command": command,
        "results": [_compact(item, max_chars) for item in preview[:6]],
    }
    return envelope
