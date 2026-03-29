from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import SESSIONS_DIR


class SessionStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or SESSIONS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    def save(self, session_id: str, payload: dict[str, Any]) -> None:
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        path = self._path_for(session_id)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, session_id: str) -> dict[str, Any]:
        path = self._path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def create(self, session_id: str, trace_id: str, request_payload: dict[str, Any], ann_payload: dict[str, Any], command_package: dict[str, Any], max_iterations: int) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        state: dict[str, Any] = {
            "session_id": session_id,
            "trace_id": trace_id,
            "status": "accepted",
            "created_at": now,
            "updated_at": now,
            "current_iteration": 0,
            "max_iterations": int(max_iterations),
            "request": request_payload,
            "current_ann_prediction": ann_payload,
            "current_command_package": command_package,
            "history": [
                {
                    "type": "initial_plan",
                    "timestamp": now,
                    "iteration_index": 0,
                    "ann_prediction": ann_payload,
                    "command_package": command_package,
                }
            ],
        }
        self.save(session_id, state)
        return state
