from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from cai.sdk.agents.mcp import MCPServer, MCPServerSse, MCPServerStdio
from mcp.types import CallToolResult, Tool as MCPTool

from rev.utils import env, int_env, float_env, json_list_env
from solver_types import SolverContext

IDA_ALLOWED_TOOLS = {
    "list_funcs",
    "find",
    "decompile",
    "disasm",
    "rename",
    "stack_frame",
    "get_bytes",
    "py_eval",
}

DBG_ALLOWED_TOOLS = {
    "set_file",
    "execute",
}


@dataclass
class RevRuntime:
    ida_server: MCPServer
    dbg_server: MCPServer
    ida_tools: list[str]
    dbg_tools: list[str]
    ida_tool_specs: list[dict[str, Any]]
    dbg_tool_specs: list[dict[str, Any]]
    tool_map_path: str
    binary_path: str
    binary_basename: str


class MCPServerHttp(MCPServer):
    def __init__(
        self,
        url: str,
        timeout_s: float = 15.0,
        cache_tools_list: bool = True,
        name: str | None = None,
    ):
        self.url = url
        self.timeout_s = timeout_s
        self._name = name or f"http-mcp: {url}"
        self.cache_tools_list = cache_tools_list
        self._cache_dirty = True
        self._tools_list: list[MCPTool] | None = None
        self.client: httpx.AsyncClient | None = None
        self.session_id: str | None = None
        self._req_id = 0
        self._cleanup_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._call_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    async def connect(self):
        if self.client is not None:
            return
        async with self._connect_lock:
            if self.client is not None:
                return
            self.client = httpx.AsyncClient(timeout=self.timeout_s)
            versions = _protocol_versions()
            last_err: Exception | None = None
            for version in versions:
                try:
                    init_res = await self._request(
                        method="initialize",
                        params={
                            "protocolVersion": version,
                            "capabilities": {"sampling": {}, "roots": {"listChanged": True}},
                            "clientInfo": {"name": "csawai-rev", "version": "0.1.0"},
                        },
                    )
                    if isinstance(init_res, dict) and "protocolVersion" in init_res:
                        await self._notify("notifications/initialized", {})
                        return
                except Exception as e:
                    last_err = e
            raise RuntimeError(f"initialize failed for {self.url}: {last_err}")

    async def cleanup(self):
        async with self._cleanup_lock:
            try:
                if self.client is not None:
                    await self.client.aclose()
            finally:
                self.client = None
                self.session_id = None
                self._cache_dirty = True
                self._tools_list = None
                self._req_id = 0

    async def list_tools(self) -> list[MCPTool]:
        if self.client is None:
            raise RuntimeError("Server not initialized. Call connect() first.")
        if self.cache_tools_list and not self._cache_dirty and self._tools_list is not None:
            return self._tools_list
        result = await self._request(method="tools/list", params={})
        tools_raw = result.get("tools", []) if isinstance(result, dict) else []
        self._tools_list = [MCPTool.model_validate(item) for item in tools_raw]
        self._cache_dirty = False
        return self._tools_list

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        if self.client is None:
            raise RuntimeError("Server not initialized. Call connect() first.")
        async with self._call_lock:
            result = await self._request(
                method="tools/call",
                params={"name": tool_name, "arguments": arguments or {}},
            )
            return CallToolResult.model_validate(result)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        await self._post(payload, expect_response=False)

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._req_id += 1
        payload = {"jsonrpc": "2.0", "id": self._req_id, "method": method, "params": params}
        body = await self._post(payload, expect_response=True)
        if not isinstance(body, dict):
            raise RuntimeError(f"invalid MCP response body for {method}")
        if body.get("error") is not None:
            raise RuntimeError(f"MCP error for {method}: {body.get('error')}")
        result = body.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"MCP result missing/invalid for {method}: {body}")
        return result

    async def _post(self, payload: dict[str, Any], expect_response: bool) -> dict[str, Any] | None:
        if self.client is None:
            raise RuntimeError("HTTP MCP client is not initialized")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        resp = await self.client.post(self.url, json=payload, headers=headers)
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"HTTP MCP POST failed ({resp.status_code}) at {self.url}: {resp.text[:400]}")
        sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        if not expect_response or not resp.content:
            return None
        return self._parse_response_body(resp)

    def _parse_response_body(self, resp: httpx.Response) -> dict[str, Any]:
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            parsed = resp.json()
            if isinstance(parsed, dict):
                return parsed
            raise RuntimeError(f"invalid JSON response type: {type(parsed).__name__}")
        if "text/event-stream" in ctype:
            for line in resp.text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:") :].strip()
                if not raw:
                    continue
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            raise RuntimeError("no JSON data frame found in event-stream response")
        parsed = json.loads(resp.text)
        if isinstance(parsed, dict):
            return parsed
        raise RuntimeError(f"unrecognized MCP response content-type: {ctype}")





def _protocol_versions() -> list[str]:
    raw = env("REV_MCP_PROTOCOL_VERSIONS", "")
    if raw:
        vals = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
        if vals:
            return vals
    return ["2025-03-26", "2024-11-05"]


def _is_dbg_like_tool_name(name: str) -> bool:
    n = name.strip().lower()
    if n.startswith("dbg_") or n.startswith("debug_"):
        return True
    dbg_markers = (
        "gdb",
        "pwndbg",
        "breakpoint",
        "backtrace",
        "register",
        "stacktrace",
        "single_step",
    )
    return any(marker in n for marker in dbg_markers)


def _filter_ida_tools(tools: list[MCPTool]) -> list[MCPTool]:
    return [tool for tool in tools if not _is_dbg_like_tool_name(tool.name)]


def _whitelist_tools(tools: list[MCPTool], allowed: set[str]) -> list[MCPTool]:
    return [tool for tool in tools if tool.name in allowed]


def _build_server(
    *,
    name: str,
    url: str,
    transport: str,
    stdio_command_env: str,
    stdio_args_env: str,
) -> MCPServer:
    t = transport.strip().lower()
    if t in {"sse", "http_sse"}:
        return MCPServerSse(params={"url": url}, cache_tools_list=True, name=name)
    if t in {"mcp", "http", "streamable_http"}:
        return MCPServerHttp(url=url, timeout_s=float_env("REV_MCP_CONNECT_TIMEOUT_S", 15), cache_tools_list=True, name=name)
    if t == "stdio":
        cmd = env(stdio_command_env, "")
        if not cmd:
            raise RuntimeError(f"{stdio_command_env} is required when transport=stdio for {name}")
        args = json_list_env(stdio_args_env, default=[])
        return MCPServerStdio(params={"command": cmd, "args": args}, cache_tools_list=True, name=name)
    raise RuntimeError(f"unsupported MCP transport '{transport}' for {name}")


async def bootstrap_loader(binary_basename: str) -> dict[str, Any]:
    loader_url = env("REV_LOADER_URL", "http://192.168.1.31:8080")
    timeout_s = int_env("REV_MCP_CONNECT_TIMEOUT_S", 15)
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(loader_url, json={"filename": binary_basename})
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"loader bootstrap failed: {resp.status_code} {resp.text[:400]}")
    out: dict[str, Any]
    try:
        out = resp.json()
    except Exception:
        out = {"text": resp.text[:1000]}
    return {"url": loader_url, "status_code": resp.status_code, "response": out}


async def init_rev_runtime(ctx: SolverContext, binary_path: str) -> RevRuntime:
    binary_basename = Path(binary_path).name

    ida_server = _build_server(
        name="ida_mcp",
        url=env("REV_IDA_MCP_URL", "http://192.168.1.31:8744/mcp"),
        transport=env("REV_IDA_MCP_TRANSPORT", "mcp"),
        stdio_command_env="REV_IDA_MCP_STDIO_COMMAND",
        stdio_args_env="REV_IDA_MCP_STDIO_ARGS",
    )
    dbg_server = _build_server(
        name="pwndbg_mcp",
        url=env("REV_PWNDBG_MCP_URL", "http://192.168.1.31:5500/debug"),
        transport=env("REV_PWNDBG_MCP_TRANSPORT", "sse"),
        stdio_command_env="REV_PWNDBG_MCP_STDIO_COMMAND",
        stdio_args_env="REV_PWNDBG_MCP_STDIO_ARGS",
    )

    retry_attempts = int_env("REV_MCP_RETRY_ATTEMPTS", 10)
    retry_backoff = float_env("REV_MCP_RETRY_BACKOFF_S", 1.5)

    async def connect_list(server: MCPServer) -> list[MCPTool]:
        last_err: Exception | None = None
        for i in range(retry_attempts):
            try:
                await server.connect()
                return await server.list_tools()
            except Exception as e:
                last_err = e
                if i == retry_attempts - 1:
                    break
                await asyncio.sleep(retry_backoff)
        raise RuntimeError(f"failed to connect/list tools for {server.name}: {last_err}")

    ida_raw = await connect_list(ida_server)
    dbg_raw = await connect_list(dbg_server)
    ida_filtered = _filter_ida_tools(ida_raw)
    ida_allowed = _whitelist_tools(ida_filtered, IDA_ALLOWED_TOOLS)
    dbg_allowed = _whitelist_tools(dbg_raw, DBG_ALLOWED_TOOLS)

    ida_raw_names = [t.name for t in ida_raw]
    ida_filtered_names = [t.name for t in ida_filtered]
    dbg_raw_names = [t.name for t in dbg_raw]
    ida_allowed_names = [t.name for t in ida_allowed]
    dbg_allowed_names = [t.name for t in dbg_allowed]
    ida_specs = [t.model_dump() for t in ida_allowed]
    dbg_specs = [t.model_dump() for t in dbg_allowed]
    ida_missing_allowed = sorted(list(IDA_ALLOWED_TOOLS - set(ida_allowed_names)))
    dbg_missing_allowed = sorted(list(DBG_ALLOWED_TOOLS - set(dbg_allowed_names)))

    print(f"[rev] IDA MCP tools (raw): {ida_raw_names}")
    print(f"[rev] IDA MCP tools (filtered): {ida_filtered_names}")
    print(f"[rev] IDA MCP tools (allowed): {ida_allowed_names}")
    print(f"[rev] DBG MCP tools (raw): {dbg_raw_names}")
    print(f"[rev] DBG MCP tools (allowed): {dbg_allowed_names}")

    tool_map_path = str((Path(ctx.docs_dir) / "mcp_tools_map.json").resolve())
    Path(tool_map_path).write_text(
        json.dumps(
            {
                "ida_tools_raw": ida_raw_names,
                "ida_tools_filtered": ida_filtered_names,
                "dbg_tools_raw": dbg_raw_names,
                "ida_tools_allowed": ida_allowed_names,
                "dbg_tools_allowed": dbg_allowed_names,
                "ida_allowed_missing_from_server": ida_missing_allowed,
                "dbg_allowed_missing_from_server": dbg_missing_allowed,
                "ida_tool_specs": ida_specs,
                "dbg_tool_specs": dbg_specs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runtime = RevRuntime(
        ida_server=ida_server,
        dbg_server=dbg_server,
        ida_tools=ida_allowed_names,
        dbg_tools=dbg_allowed_names,
        ida_tool_specs=ida_specs,
        dbg_tool_specs=dbg_specs,
        tool_map_path=tool_map_path,
        binary_path=str(Path(binary_path).resolve()),
        binary_basename=binary_basename,
    )
    ctx.runtime["rev_runtime"] = runtime
    if not isinstance(ctx.runtime.get("rev_artifact_index"), dict):
        ctx.runtime["rev_artifact_index"] = {}
    return runtime


async def cleanup_rev_runtime(ctx: SolverContext) -> None:
    runtime = ctx.runtime.get("rev_runtime")
    if runtime is None:
        return
    assert isinstance(runtime, RevRuntime)
    try:
        await runtime.ida_server.cleanup()
    finally:
        await runtime.dbg_server.cleanup()
    ctx.runtime.pop("rev_runtime", None)


def get_runtime(ctx: SolverContext) -> RevRuntime:
    runtime = ctx.runtime.get("rev_runtime")
    if runtime is None or not isinstance(runtime, RevRuntime):
        raise RuntimeError("REV runtime is not initialized")
    return runtime


def _normalize_call_result(res: CallToolResult) -> dict[str, Any]:
    texts: list[str] = []
    content: list[Any] = []
    for item in getattr(res, "content", []) or []:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            texts.append(text)
            try:
                content.append(json.loads(text))
            except Exception:
                content.append(text)
        else:
            try:
                content.append(item.model_dump())  # type: ignore[attr-defined]
            except Exception:
                content.append(str(item))
    return {
        "is_error": bool(getattr(res, "isError", False)),
        "texts": texts,
        "content": content,
        "joined_text": "\n".join(texts),
    }


async def call_server_tool(
    ctx: SolverContext,
    server: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = get_runtime(ctx)
    srv = runtime.ida_server if server == "ida" else runtime.dbg_server
    available = runtime.ida_tools if server == "ida" else runtime.dbg_tools
    if tool_name not in available:
        raise RuntimeError(f"tool '{tool_name}' not available on {server}; available={available}")
    timeout_s = float_env("REV_MCP_TOOL_TIMEOUT_S", 15.0)
    try:
        result = await asyncio.wait_for(srv.call_tool(tool_name, arguments or {}), timeout=timeout_s)
    except TimeoutError as e:
        raise RuntimeError(f"tool call timeout: {server}.{tool_name} after {timeout_s}s") from e
    norm = _normalize_call_result(result)
    norm["tool_name"] = tool_name
    norm["server"] = server
    return norm
