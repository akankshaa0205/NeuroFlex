from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON data atomically to disk using temp-file + rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(f"{target.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(target)
    return target


def export_session_csv(path: str | Path, payload: dict[str, Any] | Iterable[dict[str, Any]]) -> Path:
    """Export session data to CSV. Flatten nested dicts to columns.

    This keeps the interface simple for early repository usage while matching the
    PRD requirement for structured export.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]]
    if isinstance(payload, dict):
        rows = [payload]
    else:
        rows = list(payload)

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return target
