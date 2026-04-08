from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.policy_runtime import default_policy_runtime_state
from config import ANN_SETTINGS, API_SETTINGS, SESSIONS_DIR


class SessionStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or SESSIONS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    @staticmethod
    def payload_checksum(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, session_id: str, payload: dict[str, Any]) -> None:
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        path = self._path_for(session_id)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, session_id: str) -> dict[str, Any]:
        path = self._path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def create(
        self,
        session_id: str,
        trace_id: str,
        request_payload: dict[str, Any],
        ann_payload: dict[str, Any],
        command_package: dict[str, Any] | None,
        max_iterations: int,
        surrogate_validation: dict[str, Any] | None = None,
        initial_status: str = "accepted",
        stop_reason: str | None = None,
        decision_reason: str = "initial_plan_created",
        objective_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        command_checksum = self.payload_checksum(command_package) if command_package is not None else None
        command_schema_version = command_package.get("schema_version") if command_package is not None else None
        command_catalog_version = command_package.get("command_catalog_version") if command_package is not None else None
        state: dict[str, Any] = {
            "session_id": session_id,
            "trace_id": trace_id,
            "status": initial_status,
            "stop_reason": stop_reason,
            "created_at": now,
            "updated_at": now,
            "current_iteration": 0,
            "max_iterations": int(max_iterations),
            "request": request_payload,
            "current_ann_prediction": ann_payload,
            "current_surrogate_validation": surrogate_validation,
            "current_command_package": command_package,
            "objective_targets": request_payload.get("optimization_targets", {}),
            "objective_state": objective_state or {},
            "policy_runtime": default_policy_runtime_state(),
            "artifact_manifest": {
                "manifest_version": "artifact_manifest.v1",
                "session_id": session_id,
                "trace_id": trace_id,
                "created_at": now,
                "updated_at": now,
                "api_version": API_SETTINGS.version,
                "request_schema_version": request_payload.get("schema_version"),
                "command_schema_version": command_schema_version,
                "command_catalog_version": command_catalog_version,
                "ws_event_schema_version": "session_event.v1",
                "ann_model_version": ann_payload.get("ann_model_version", ANN_SETTINGS.model_version),
                "ann_checkpoint_path": str(ANN_SETTINGS.checkpoint_path),
                "ann_metadata_path": str(ANN_SETTINGS.metadata_path),
                "context_bundle_version": "context_bundle.v1",
                "acceptance_policy_snapshot": request_payload.get("optimization_policy", {}).get("acceptance", {}),
                "objective_targets_snapshot": request_payload.get("optimization_targets", {}),
                "client_capability_profile": request_payload.get("client_capabilities", {}),
                "latest_iteration_index": 0,
                "latest_command_package_checksum_sha256": command_checksum,
                "latest_surrogate_validation": surrogate_validation,
                "planner_mode": "fixed",
                "llm_policy_snapshot": default_policy_runtime_state(),
                "latest_planning_decision": None,
                "history": [
                    {
                        "timestamp": now,
                        "iteration_index": 0,
                        "decision_reason": decision_reason,
                        "stop_reason": stop_reason,
                        "command_package_checksum_sha256": command_checksum,
                        "planning_provenance": None,
                    }
                ],
            },
            "history": [
                {
                    "type": "initial_plan",
                    "timestamp": now,
                    "iteration_index": 0,
                    "decision_reason": decision_reason,
                    "stop_reason": stop_reason,
                    "surrogate_validation": surrogate_validation,
                    "planning_provenance": None,
                    "ann_prediction": ann_payload,
                    "command_package": command_package,
                }
            ],
        }
        self.save(session_id, state)
        return state
