from __future__ import annotations

import json
from pathlib import Path

REQUIRED_KEYS = {"id", "dataset", "context", "candidate"}


def validate_record(record: dict) -> None:
    missing = REQUIRED_KEYS.difference(record.keys())
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"Missing required fields: {missing_fields}")


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)

    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                validate_record(obj)
            except ValueError as exc:
                raise ValueError(f"Invalid record at line {i} in {p}: {exc}") from exc
            rows.append(obj)

    return rows


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            validate_record(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
