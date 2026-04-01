from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.core.capabilities_catalog import load_capabilities_catalog
from app.core.exceptions import FamilyProfileConstraintError, UnsupportedAntennaFamilyError
from app.core.family_registry import list_supported_families
from app.core.json_contracts import ContractValidationError, validate_contract
from app.core.schemas import OptimizeRequest, OptimizeResponse
from app.core.session_store import SessionStore
from app.llm.intent_parser import summarize_user_intent
from app.llm.ollama_client import check_ollama_health, generate_text
from central_brain import CentralBrain
from config import API_SETTINGS


brain = CentralBrain()
session_store = SessionStore()
app = FastAPI(title=API_SETTINGS.title, version=API_SETTINGS.version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": API_SETTINGS.title,
        "version": API_SETTINGS.version,
        "ann_model_ready": brain.ann_predictor.is_ready(),
    }


@app.get("/api/v1/capabilities")
def capabilities() -> dict[str, Any]:
    return load_capabilities_catalog()


@app.post("/api/v1/intent/parse")
def parse_intent(payload: dict[str, Any]) -> dict[str, Any]:
    user_request = payload.get("user_request")
    if not isinstance(user_request, str) or not user_request.strip():
        raise HTTPException(
            status_code=422,
            detail={"error_code": "INVALID_INTENT_REQUEST", "message": "user_request must be a non-empty string"},
        )
    summary = summarize_user_intent(user_request)
    return {
        "status": "ok",
        "intent_summary": summary,
    }


@app.post("/api/v1/chat")
def chat(payload: dict[str, Any]) -> dict[str, Any]:
    user_message = payload.get("message")
    current_requirements = payload.get("requirements", {})
    if not isinstance(user_message, str) or not user_message.strip():
        raise HTTPException(
            status_code=422,
            detail={"error_code": "INVALID_CHAT_REQUEST", "message": "message must be a non-empty string"},
        )
    if not isinstance(current_requirements, dict):
        current_requirements = {}

    intent = summarize_user_intent(user_message)
    capabilities_catalog = load_capabilities_catalog()
    parsed_freq = intent.get("parsed_frequency_ghz")
    parsed_bw = intent.get("parsed_bandwidth_mhz")
    parsed_family = intent.get("parsed_antenna_family")

    merged = {
        "frequency_ghz": parsed_freq if parsed_freq is not None else current_requirements.get("frequency_ghz"),
        "bandwidth_mhz": parsed_bw if parsed_bw is not None else current_requirements.get("bandwidth_mhz"),
        "antenna_family": parsed_family if parsed_family is not None else current_requirements.get("antenna_family"),
    }

    supported_families = list_supported_families()
    frequency_range = capabilities_catalog.get("frequency_range_ghz", {})
    bandwidth_range = capabilities_catalog.get("bandwidth_range_mhz", {})
    conductor_materials = capabilities_catalog.get("available_conductor_materials", [])
    substrate_materials = capabilities_catalog.get("available_substrate_materials", [])

    freq_min = frequency_range.get("min")
    freq_max = frequency_range.get("max")
    bw_min = bandwidth_range.get("min")
    bw_max = bandwidth_range.get("max")

    text = user_message.lower()
    if "available" in text and "famil" in text:
        assistant_message = (
            "Available antenna families are: "
            f"{', '.join(supported_families)}. "
            "For rectangular patch, use microstrip_patch. "
            "Tell me target frequency (GHz), bandwidth (MHz), and chosen family to start the pipeline."
        )
    elif "capabil" in text or ("range" in text and ("frequency" in text or "bandwidth" in text)):
        assistant_message = (
            "Current capability ranges are: "
            f"frequency {freq_min} to {freq_max} GHz, "
            f"bandwidth {bw_min} to {bw_max} MHz. "
            "These are configurable from the capabilities catalog."
        )
    elif "substrate" in text or "conductor" in text or "material" in text:
        assistant_message = (
            f"Available conductor materials: {', '.join(map(str, conductor_materials))}. "
            f"Available substrate materials: {', '.join(map(str, substrate_materials))}."
        )
    else:
        missing = [
            key
            for key, value in merged.items()
            if value in (None, "")
        ]
        if check_ollama_health(timeout_sec=3):
            llm_text = generate_text(
                system_prompt=(
                    "You are an antenna design assistant. "
                    "Answer user's question in 2-5 lines and guide them to provide frequency_ghz, "
                    "bandwidth_mhz, and antenna_family. Keep it concise and practical."
                ),
                prompt=(
                    f"User message: {user_message}\n"
                    f"Current extracted requirements: {merged}\n"
                    f"Supported families: {supported_families}\n"
                    f"Capability ranges: frequency_ghz={frequency_range}, bandwidth_mhz={bandwidth_range}\n"
                    f"Available conductor materials: {conductor_materials}\n"
                    f"Available substrate materials: {substrate_materials}\n"
                    "If user asks family choices, list all exactly."
                ),
                timeout_sec=30,
            )
            assistant_message = llm_text or "Tell me frequency (GHz), bandwidth (MHz), and antenna family."
        else:
            if missing:
                assistant_message = (
                    "Got it. I still need: "
                    f"{', '.join(missing)}. "
                    f"Supported families: {', '.join(supported_families)}. "
                    f"Frequency range: {freq_min}-{freq_max} GHz. "
                    f"Bandwidth range: {bw_min}-{bw_max} MHz."
                )
            else:
                assistant_message = (
                    "Great, requirements captured. "
                    f"frequency={merged['frequency_ghz']} GHz, bandwidth={merged['bandwidth_mhz']} MHz, "
                    f"family={merged['antenna_family']}. You can start the pipeline now."
                )

    return {
        "status": "ok",
        "assistant_message": assistant_message,
        "intent_summary": intent,
        "requirements": merged,
        "supported_families": supported_families,
        "capabilities": {
            "frequency_range_ghz": frequency_range,
            "bandwidth_range_mhz": bandwidth_range,
            "available_conductor_materials": conductor_materials,
            "available_substrate_materials": substrate_materials,
        },
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
    except UnsupportedAntennaFamilyError as exc:
        raise HTTPException(status_code=422, detail={"error_code": "FAMILY_NOT_SUPPORTED", "message": str(exc)}) from exc
    except FamilyProfileConstraintError as exc:
        raise HTTPException(status_code=422, detail={"error_code": "FAMILY_PROFILE_CONSTRAINT_FAILED", "message": str(exc)}) from exc
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
            "intent_summary": session.get("intent_summary"),
            "surrogate_validation": surrogate_validation,
            "surrogate_summary": surrogate_summary,
            "policy_runtime": session.get("policy_runtime", {}),
            "latest_planning_decision": session.get("artifact_manifest", {}).get("latest_planning_decision"),
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
