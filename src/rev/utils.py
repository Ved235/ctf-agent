from __future__ import annotations

import json
import os
from typing import Any


def int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def truncate(text: str, n: int) -> str:
    if n <= 0:
        return ""
    return text if len(text) <= n else text[:n] + "...<truncated>"


def json_list_env(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default or [])
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return [part for part in raw.split(" ") if part]
