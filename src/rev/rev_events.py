from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rev.rev_types import JSONValue, RevEventRecord
from solver_types import SolverContext


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_event_store(ctx: SolverContext) -> None:
    events_path = Path(ctx.events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    if not events_path.exists():
        events_path.touch()


def read_events(ctx: SolverContext) -> list[RevEventRecord]:
    ensure_event_store(ctx)
    out: list[RevEventRecord] = []
    with Path(ctx.events_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(RevEventRecord.model_validate(json.loads(line)))
    return out


def append_event(
    ctx: SolverContext,
    *,
    actor: str,
    event_type: str,
    payload: dict[str, JSONValue] | None = None,
) -> RevEventRecord:
    ensure_event_store(ctx)
    ctx.step_counter += 1

    rec = RevEventRecord(
        event_id=uuid.uuid4().hex,
        ts=_now_iso(),
        actor=actor,
        event_type=event_type,
        payload=payload or {},
        step_index=ctx.step_counter,
    )

    with Path(ctx.events_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec.model_dump(), ensure_ascii=True) + "\n")

    return rec
