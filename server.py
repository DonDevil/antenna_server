from __future__ import annotations

import asyncio
import json
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
        return {
            "session_id": session_id,
            "status": session.get("status", "unknown"),
            "current_iteration": session.get("current_iteration", 0),
            "max_iterations": session.get("max_iterations", 0),
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
                        await websocket.send_json({
                            "event_type": "iteration_update",
                            "session_id": session_id,
                            "current_iteration": session.get("current_iteration", 0),
                            "status": session.get("status", "running"),
                            "entry": entry,
                        })
                    last_history_count = current_history_count
                
                if session.get("status") in ["completed", "max_iterations_reached", "stopped"]:
                    await websocket.send_json({
                        "event_type": "session_complete",
                        "session_id": session_id,
                        "status": session.get("status"),
                        "final_iteration": session.get("current_iteration", 0),
                    })
                    break
                
                await asyncio.sleep(1)
                
            except FileNotFoundError:
                await websocket.send_json({
                    "event_type": "error",
                    "error_code": "SESSION_NOT_FOUND",
                    "message": f"Session {session_id} not found",
                })
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({
                "event_type": "error",
                "error_code": "STREAM_ERROR",
                "message": str(exc),
            })
        except Exception:
            pass
