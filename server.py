from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app.core.json_contracts import ContractValidationError, validate_contract
from app.core.schemas import OptimizeRequest, OptimizeResponse
from app.core.session_store import SessionStore
from central_brain import CentralBrain
from config import API_SETTINGS


brain = CentralBrain()
session_store = SessionStore()
app = FastAPI(title=API_SETTINGS.title, version=API_SETTINGS.version)


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": API_SETTINGS.title,
        "version": API_SETTINGS.version,
        "ann_model_ready": brain.ann_predictor.is_ready(),
    }


@app.post("/api/v1/optimize", response_model=OptimizeResponse)
def optimize(payload: dict[str, Any]) -> OptimizeResponse:
    try:
        validate_contract("optimize_request", payload)
        request_model = OptimizeRequest.model_validate(payload)
        response = brain.optimize(request_model)

        if response.command_package is not None:
            validate_contract("command_package", response.command_package)
        validate_contract("optimize_response", response.model_dump(mode="json"))
        return response
    except ContractValidationError as exc:
        raise HTTPException(status_code=422, detail={"error_code": "SCHEMA_VALIDATION_FAILED", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/client-feedback")
def client_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        validate_contract("client_feedback", payload)
        result = brain.process_feedback(payload)
        if "next_command_package" in result:
            validate_contract("command_package", result["next_command_package"])
        return result
    except ContractValidationError as exc:
        raise HTTPException(status_code=422, detail={"error_code": "SCHEMA_VALIDATION_FAILED", "message": str(exc)}) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error_code": "SESSION_NOT_FOUND", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error_code": "FEEDBACK_PROCESSING_FAILED", "message": str(exc)}) from exc


@app.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """Retrieve current session state and iteration history."""
    try:
        session = session_store.load(session_id)
        surrogate_validation = session.get("current_surrogate_validation")
        surrogate_summary = None
        if isinstance(surrogate_validation, dict):
            residual = surrogate_validation.get("residual", {})
            surrogate_summary = {
                "accepted": bool(surrogate_validation.get("accepted", False)),
                "confidence": float(surrogate_validation.get("confidence", 0.0)),
                "threshold": float(surrogate_validation.get("threshold", 0.0)),
                "decision_reason": surrogate_validation.get("decision_reason"),
                "center_frequency_abs_error_ghz": float(residual.get("center_frequency_abs_error_ghz", 0.0)),
                "bandwidth_abs_error_mhz": float(residual.get("bandwidth_abs_error_mhz", 0.0)),
            }
        return {
            "session_id": session_id,
            "status": session.get("status", "unknown"),
            "stop_reason": session.get("stop_reason"),
            "current_iteration": session.get("current_iteration", 0),
            "max_iterations": session.get("max_iterations", 0),
            "surrogate_validation": surrogate_validation,
            "surrogate_summary": surrogate_summary,
            "history_count": len(session.get("history", [])),
            "latest_entry": session.get("history", [{}])[-1] if session.get("history") else None,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error_code": "SESSION_NOT_FOUND", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error_code": "SESSION_QUERY_FAILED", "message": str(exc)}) from exc


@app.websocket("/api/v1/sessions/{session_id}/stream")
async def stream_session_events(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time session progress updates."""
    await websocket.accept()
    last_history_count = 0
    
    try:
        while True:
            try:
                session = session_store.load(session_id)
                current_history_count = len(session.get("history", []))
                
                if current_history_count > last_history_count:
                    new_entries = session.get("history", [])[last_history_count:]
                    for entry in new_entries:
                        event = {
                            "schema_version": "session_event.v1",
                            "event_type": "iteration.completed",
                            "session_id": session_id,
                            "trace_id": session.get("trace_id", ""),
                            "iteration_index": int(entry.get("iteration_index", session.get("current_iteration", 0))),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "payload": {
                                "stage": session.get("status", "running"),
                                "message": "Session history updated",
                                "accepted": bool(entry.get("evaluation", {}).get("accepted", False)),
                                "history_entry": entry,
                            },
                        }
                        validate_contract("session_event", event)
                        await websocket.send_json(event)
                    last_history_count = current_history_count
                
                if session.get("status") in ["completed", "max_iterations_reached", "stopped"]:
                    terminal_event_type = "session.completed"
                    if session.get("status") in ["max_iterations_reached", "stopped"]:
                        terminal_event_type = "session.failed"
                    terminal_event = {
                        "schema_version": "session_event.v1",
                        "event_type": terminal_event_type,
                        "session_id": session_id,
                        "trace_id": session.get("trace_id", ""),
                        "iteration_index": int(session.get("current_iteration", 0)),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "payload": {
                            "stage": session.get("status", "unknown"),
                            "message": "Session reached terminal state",
                            "accepted": bool(session.get("status") == "completed"),
                        },
                    }
                    validate_contract("session_event", terminal_event)
                    await websocket.send_json(terminal_event)
                    break
                
                await asyncio.sleep(1)
                
            except FileNotFoundError:
                missing_event = {
                    "schema_version": "session_event.v1",
                    "event_type": "session.failed",
                    "session_id": session_id,
                    "trace_id": "",
                    "iteration_index": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "payload": {
                        "stage": "error",
                        "message": f"Session {session_id} not found",
                        "accepted": False,
                    },
                }
                validate_contract("session_event", missing_event)
                await websocket.send_json(missing_event)
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            error_event = {
                "schema_version": "session_event.v1",
                "event_type": "session.failed",
                "session_id": session_id,
                "trace_id": "",
                "iteration_index": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "stage": "error",
                    "message": str(exc),
                    "accepted": False,
                },
            }
            validate_contract("session_event", error_event)
            await websocket.send_json(error_event)
        except Exception:
            pass
