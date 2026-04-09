from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

from app.data.rect_patch_feedback import RECT_PATCH_FEEDBACK_COLUMNS
from config import ROOT_DIR, RECT_PATCH_DATA_SETTINGS


_SCHEMA_PATH = ROOT_DIR / "schemas" / "data" / "rect_patch_feedback.v1.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {column: row.get(column, "") for column in RECT_PATCH_FEEDBACK_COLUMNS}
    if "accepted" in normalized:
        normalized["accepted"] = bool(normalized["accepted"])
    return normalized


def validate_rect_patch_feedback_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_row(row)
    try:
        validate(instance=normalized, schema=_SCHEMA)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.path) or "<root>"
        raise ValueError(f"Invalid rectangular patch feedback row: {location}: {exc.message}") from exc
    return normalized


def ensure_rect_patch_feedback_header(csv_path: Path = RECT_PATCH_DATA_SETTINGS.raw_feedback_path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RECT_PATCH_FEEDBACK_COLUMNS))
        writer.writeheader()


def append_rect_patch_feedback_row(
    row: dict[str, Any],
    csv_path: Path = RECT_PATCH_DATA_SETTINGS.raw_feedback_path,
) -> dict[str, Any]:
    normalized = validate_rect_patch_feedback_row(row)
    ensure_rect_patch_feedback_header(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RECT_PATCH_FEEDBACK_COLUMNS))
        writer.writerow(normalized)
    return normalized