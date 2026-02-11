#!/usr/bin/env python3
"""Copy patched openai_responses.py into installed cai-framework."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SOURCE = Path("/workspace/scripts/openai_responses.py")


def main() -> int:
    try:
        import cai.sdk.agents.models.openai_responses as openai_responses
    except Exception as exc:
        print(f"[patch] import error: {exc}", file=sys.stderr)
        return 1

    if not SOURCE.is_file():
        print(f"[patch] source not found: {SOURCE}", file=sys.stderr)
        return 1

    target_path = Path(openai_responses.__file__).resolve()
    backup_path = target_path.with_suffix(".py.bak")

    try:
        if not backup_path.exists():
            shutil.copy2(target_path, backup_path)
        shutil.copy2(SOURCE, target_path)
    except Exception as exc:
        print(f"[patch] copy error: {exc}", file=sys.stderr)
        return 1

    print(f"[patch] applied: {SOURCE} -> {target_path}")
    print(f"[patch] backup: {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
