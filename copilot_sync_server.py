from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import ROOT_DIR, SESSIONS_DIR


WORKSPACE_ROOT = ROOT_DIR
DEFAULT_CURRENT_FILE = "app/antenna/materials.py"
SYNC_STORE_PATH = SESSIONS_DIR / "copilot_sync" / "messages.json"
_STORE_LOCK = Lock()

app = FastAPI(
    title="Copilot Sync Relay",
    version="0.1.0",
    description="Lightweight HTTP relay for server-side and client-side Copilot coordination.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SyncMessageIn(BaseModel):
    sender: Literal["server-copilot", "client-copilot", "human"] = "server-copilot"
    text: str = Field(min_length=1)
    topic: str = "general"
    related_files: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    proposed_actions: list[str] = Field(default_factory=list)
    conversation_id: str = "default"


class ClearMessagesRequest(BaseModel):
    conversation_id: str | None = None


class FileGuidanceRequest(BaseModel):
    file_path: str = Field(min_length=1)
    goal: str = "align the next implementation step between copilots"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_store_exists() -> None:
    with _STORE_LOCK:
        SYNC_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not SYNC_STORE_PATH.exists():
            SYNC_STORE_PATH.write_text(
                json.dumps({"schema_version": "copilot_sync.v1", "messages": []}, indent=2),
                encoding="utf-8",
            )


def _read_store() -> dict[str, Any]:
    _ensure_store_exists()
    with _STORE_LOCK:
        return json.loads(SYNC_STORE_PATH.read_text(encoding="utf-8"))


def _write_store(payload: dict[str, Any]) -> None:
    with _STORE_LOCK:
        SYNC_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SYNC_STORE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_message(message: SyncMessageIn) -> dict[str, Any]:
    store = _read_store()
    message_payload: dict[str, Any] = message.model_dump()
    entry: dict[str, Any] = {
        "id": len(store.get("messages", [])) + 1,
        "timestamp": _utc_now(),
        **message_payload,
    }
    store.setdefault("messages", []).append(entry)
    _write_store(store)
    return entry


def _resolve_workspace_file(file_path: str) -> tuple[Path, str]:
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = (WORKSPACE_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    root_resolved = WORKSPACE_ROOT.resolve()
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not candidate.is_relative_to(root_resolved):
        raise HTTPException(status_code=400, detail="Only workspace files are allowed")

    relative_path = candidate.relative_to(root_resolved).as_posix()
    return candidate, relative_path


def _extract_functions(file_text: str) -> list[str]:
    functions: list[str] = []
    for line in file_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            name = stripped.split("def ", 1)[1].split("(", 1)[0].strip()
            functions.append(name)
    return functions


def _build_file_guidance(file_path: str, goal: str) -> dict[str, Any]:
    file_abs_path, relative_path = _resolve_workspace_file(file_path)
    file_text = file_abs_path.read_text(encoding="utf-8")
    preview = "\n".join(file_text.splitlines()[:80])
    functions = _extract_functions(file_text)

    summary = (
        f"`{relative_path}` currently exposes {', '.join(functions) if functions else 'no top-level helpers'} "
        "and should stay simple, deterministic, and JSON-friendly for both copilots."
    )
    notes = [
        "Preserve non-throwing fallback behavior so coordination stays robust.",
        "Keep return payloads stable and explicit; avoid hidden side effects.",
        f"Goal for the paired copilot: {goal}",
    ]
    future_function: dict[str, Any] = {
        "name": "propose_next_helper",
        "signature": "def propose_next_helper(...) -> dict[str, Any]:",
        "why": "Mirror the existing style and keep the server/client upgrade path synchronized.",
        "implementation_outline": [
            "Use a small module-level lookup table.",
            "Return JSON-safe primitives only.",
            "Keep the default path tolerant and predictable.",
        ],
    }

    if relative_path == "app/antenna/materials.py" or "get_substrate_properties" in file_text:
        summary = (
            "`app/antenna/materials.py` is a focused substrate lookup helper. "
            "It currently handles substrate dielectric data and uses a safe FR-4 fallback."
        )
        notes = [
            "Mirror the current `get_substrate_properties()` pattern instead of introducing a new style.",
            "Use a module-level `_CONDUCTOR_LIBRARY` beside `_SUBSTRATE_LIBRARY` for future parity.",
            "Keep fallback behavior non-throwing and default to `Copper (annealed)` for conductor lookups.",
            f"Requested coordination goal: {goal}",
        ]
        future_function = {
            "name": "get_conductor_properties",
            "signature": "def get_conductor_properties(name: str | None) -> dict[str, Any]:",
            "why": (
                "The repo already names conductor materials in capability and schema flows, "
                "but this module only resolves substrate details today."
            ),
            "implementation_outline": [
                "Add a `_CONDUCTOR_LIBRARY` constant keyed by display name such as `Copper (annealed)`.",
                "Normalize the input with the same `strip()` and fallback style used in `get_substrate_properties()`.",
                "Return a JSON-safe dict with `name` plus conductor attributes like `conductivity`.",
                "Cast numeric fields to `float` before returning so both copilots see a stable contract.",
            ],
        }

    return {
        "path": relative_path,
        "functions": functions,
        "summary": summary,
        "implementation_notes": notes,
        "future_function": future_function,
        "preview": preview,
    }


def _run_git_command(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_snapshot() -> dict[str, Any]:
    branch = _run_git_command("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    last_commit = _run_git_command("log", "-1", "--pretty=%h %s") or "unavailable"
    raw_status = _run_git_command("status", "--short") or ""

    changed_files: list[dict[str, str]] = []
    for line in raw_status.splitlines():
        if len(line) < 4:
            continue
        changed_files.append({"status": line[:2].strip() or "??", "path": line[3:].strip()})

    return {
        "branch": branch,
        "last_commit": last_commit,
        "changed_files": changed_files,
        "change_count": len(changed_files),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    _ensure_store_exists()
    return {
        "status": "ok",
        "service": app.title,
        "version": app.version,
        "store_path": str(SYNC_STORE_PATH),
    }


@app.get("/api/v1/copilot-sync/bootstrap")
def bootstrap(file_path: str = DEFAULT_CURRENT_FILE) -> dict[str, Any]:
    store = _read_store()
    current_file = _build_file_guidance(
        file_path,
        goal="tell the client-side copilot how to implement the next helper in the same style",
    )
    recent_messages = store.get("messages", [])[-10:]

    return {
        "status": "ok",
        "service": app.title,
        "workspace_root": str(WORKSPACE_ROOT),
        "recent_changes": _git_snapshot(),
        "current_file": current_file,
        "recent_messages": recent_messages,
        "handoff_message": {
            "target": "client-copilot",
            "summary": (
                "Stay aligned with `app/antenna/materials.py`: keep helper functions lookup-based, "
                "non-throwing, and JSON-safe. The next recommended addition is `get_conductor_properties()`."
            ),
        },
    }


@app.get("/api/v1/copilot-sync/messages")
def list_messages(
    conversation_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    store = _read_store()
    messages = list(store.get("messages", []))
    if conversation_id:
        messages = [item for item in messages if item.get("conversation_id") == conversation_id]
    messages = messages[-limit:]
    return {
        "status": "ok",
        "count": len(messages),
        "messages": messages,
    }


@app.get("/api/v1/copilot-sync/changes")
def get_changes() -> dict[str, Any]:
    return {
        "status": "ok",
        "recent_changes": _git_snapshot(),
    }


@app.post("/api/v1/copilot-sync/message")
def post_message(payload: SyncMessageIn) -> dict[str, Any]:
    entry = _append_message(payload)
    return {
        "status": "ok",
        "message": entry,
        "poll_hint": "/api/v1/copilot-sync/messages",
    }


@app.post("/api/v1/copilot-sync/clear")
def clear_messages(payload: ClearMessagesRequest) -> dict[str, Any]:
    store = _read_store()
    messages = list(store.get("messages", []))

    if payload.conversation_id:
        kept = [item for item in messages if item.get("conversation_id") != payload.conversation_id]
        removed = len(messages) - len(kept)
        store["messages"] = kept
    else:
        removed = len(messages)
        store["messages"] = []

    _write_store(store)
    return {
        "status": "ok",
        "removed": removed,
    }


@app.post("/api/v1/copilot-sync/file-guidance")
def file_guidance(payload: FileGuidanceRequest) -> dict[str, Any]:
    return {
        "status": "ok",
        "guidance": _build_file_guidance(payload.file_path, goal=payload.goal),
    }


def run() -> None:
    uvicorn.run("copilot_sync_server:app", host="0.0.0.0", port=8011, reload=False)


if __name__ == "__main__":
    run()
