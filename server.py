from __future__ import annotations

import asyncio
import threading
import uuid
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
from app.llm.ollama_client import check_ollama_health, generate_text, warmup_model
from app.planning.v2_command_contract import V2CommandValidationError
from central_brain import CentralBrain
from config import ANN_SETTINGS, API_SETTINGS, OLLAMA_SETTINGS


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

_runtime_health_lock = threading.Lock()
_runtime_health: dict[str, Any] = {
    "warmup_in_progress": False,
    "ann_status": "none",
    "ann_message": "warmup_not_started",
    "llm_status": "none",
    "llm_message": "warmup_not_started",
}
_chat_memory_lock = threading.Lock()
_chat_requirements_memory: dict[str, dict[str, Any]] = {}


def _set_runtime_health(**updates: Any) -> None:
    with _runtime_health_lock:
        _runtime_health.update(updates)


def _warm_dependencies() -> None:
    try:
        ann_ready = brain.ann_predictor.warm_up()
        if ann_ready:
            _set_runtime_health(ann_status="available", ann_message="ann_model_loaded")
            print(f"[startup] ANN ready for use: {ANN_SETTINGS.model_version}")
        else:
            ann_error = brain.ann_predictor.last_error() or "ann_model_unavailable"
            _set_runtime_health(ann_status="none", ann_message=ann_error)
            print(f"[startup] ANN unavailable: {ann_error}")

        fast_name = OLLAMA_SETTINGS.fast_model_name
        big_name = OLLAMA_SETTINGS.big_model_name

        print(f"[startup] Warming fast intent model '{fast_name}'...")
        fast_ready = warmup_model(
            timeout_sec=max(60, int(OLLAMA_SETTINGS.timeout_sec)),
            model_name=fast_name,
        )
        print(f"[startup] Warming chat model '{big_name}'...")
        big_ready = warmup_model(
            timeout_sec=max(120, int(OLLAMA_SETTINGS.timeout_sec)),
            model_name=big_name,
        )

        llm_ready = fast_ready or big_ready
        if llm_ready:
            message_parts: list[str] = []
            if fast_ready:
                message_parts.append(f"fast_ready:{fast_name}")
            if big_ready:
                message_parts.append(f"big_ready:{big_name}")
            llm_message = ", ".join(message_parts) if message_parts else "llm_ready"
            _set_runtime_health(llm_status="available", llm_message=llm_message)
            if fast_ready:
                print(f"[startup] Fast intent model ready: {fast_name}")
            if big_ready:
                print(f"[startup] Chat model ready: {big_name}")
        else:
            llm_message = "ollama_unreachable_or_model_not_loaded"
            if check_ollama_health(timeout_sec=3):
                llm_message = "llm_warmup_failed"
            _set_runtime_health(llm_status="none", llm_message=llm_message)
            print(f"[startup] LLM unavailable: {llm_message}")
    finally:
        _set_runtime_health(warmup_in_progress=False)


def start_background_warmup(force: bool = False) -> None:
    with _runtime_health_lock:
        if bool(_runtime_health.get("warmup_in_progress")):
            return
        if not force and _runtime_health.get("ann_status") == "available" and _runtime_health.get("llm_status") == "available":
            return
        _runtime_health["warmup_in_progress"] = True
        if _runtime_health.get("ann_status") != "available":
            _runtime_health["ann_status"] = "loading"
            _runtime_health["ann_message"] = "warming_ann"
        if _runtime_health.get("llm_status") != "available":
            _runtime_health["llm_status"] = "loading"
        _runtime_health["llm_message"] = (
            f"warming_fast={OLLAMA_SETTINGS.fast_model_name}, "
            f"warming_big={OLLAMA_SETTINGS.big_model_name}"
        )
    worker = threading.Thread(target=_warm_dependencies, name="dependency-warmup", daemon=True)
    worker.start()


@app.on_event("startup")
async def startup_warmup() -> None:
    start_background_warmup()


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    start_background_warmup()

    ann_artifacts_ready = brain.ann_predictor.is_ready()
    ann_loaded = brain.ann_predictor.is_loaded()
    if ann_artifacts_ready and not ann_loaded:
        ann_loaded = brain.ann_predictor.warm_up()
        if ann_loaded:
            _set_runtime_health(ann_status="available", ann_message="ann_model_loaded")
        else:
            ann_error = brain.ann_predictor.last_error() or "ann_model_unavailable"
            _set_runtime_health(ann_status="none", ann_message=ann_error)
    elif ann_loaded:
        _set_runtime_health(ann_status="available", ann_message="ann_model_loaded")

    ollama_reachable = check_ollama_health(timeout_sec=1)
    with _runtime_health_lock:
        llm_status = str(_runtime_health.get("llm_status", "none"))
    if ollama_reachable and llm_status == "none":
        start_background_warmup(force=True)
    elif not ollama_reachable and llm_status == "available":
        _set_runtime_health(llm_status="none", llm_message="ollama_unreachable")

    with _runtime_health_lock:
        ann_status = str(_runtime_health.get("ann_status", "none"))
        ann_message = _runtime_health.get("ann_message")
        llm_status = str(_runtime_health.get("llm_status", "none"))
        llm_message = _runtime_health.get("llm_message")

    if not ann_artifacts_ready and ann_status != "loading":
        ann_status = "none"

    return {
        "status": "ok",
        "service": API_SETTINGS.title,
        "version": API_SETTINGS.version,
        "ann_model_ready": ann_artifacts_ready,
        "ann_model_loaded": ann_loaded,
        "ann_status": ann_status,
        "ann_model_version": ANN_SETTINGS.model_version,
        "llm_status": llm_status,
        "llm_model": OLLAMA_SETTINGS.model_name,
        "fast_llm_model": OLLAMA_SETTINGS.fast_model_name,
        "big_llm_model": OLLAMA_SETTINGS.big_model_name,
        "ollama_base_url": OLLAMA_SETTINGS.base_url,
        "ollama_reachable": ollama_reachable,
        "ann_message": ann_message,
        "llm_message": llm_message,
        "live_retraining": brain.live_retraining.status(),
    }


@app.get("/api/v1/capabilities")
def capabilities() -> dict[str, Any]:
    return load_capabilities_catalog()


def _first_non_empty_string(value: Any) -> str | None:
    if isinstance(value, list) and value:
        value = value[0]
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _requirements_from_saved_session(session_payload: dict[str, Any]) -> dict[str, Any]:
    request_payload = session_payload.get("request", {}) if isinstance(session_payload.get("request"), dict) else {}
    target_spec = request_payload.get("target_spec", {}) if isinstance(request_payload.get("target_spec"), dict) else {}
    design_constraints = (
        request_payload.get("design_constraints", {})
        if isinstance(request_payload.get("design_constraints"), dict)
        else {}
    )
    allowed_substrates = _coerce_string_list(design_constraints.get("allowed_substrates"))
    allowed_materials = _coerce_string_list(design_constraints.get("allowed_materials"))
    return {
        "frequency_ghz": target_spec.get("frequency_ghz"),
        "bandwidth_mhz": target_spec.get("bandwidth_mhz"),
        "antenna_family": target_spec.get("antenna_family"),
        "patch_shape": target_spec.get("patch_shape"),
        "substrate_material": _first_non_empty_string(allowed_substrates),
        "conductor_material": _first_non_empty_string(allowed_materials),
        "allowed_substrates": allowed_substrates,
        "allowed_materials": allowed_materials,
        "design_constraints": design_constraints,
        "target_spec": target_spec,
    }


def _hydrate_chat_requirements(session_id: str | None, current_requirements: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if session_id:
        try:
            merged.update(_requirements_from_saved_session(session_store.load(session_id)))
        except FileNotFoundError:
            pass
        with _chat_memory_lock:
            remembered = _chat_requirements_memory.get(session_id, {})
            if isinstance(remembered, dict):
                merged.update(dict(remembered))
    merged.update(current_requirements)
    return merged


def _store_chat_requirements(session_id: str, requirements: dict[str, Any]) -> None:
    with _chat_memory_lock:
        _chat_requirements_memory[session_id] = dict(requirements)


def _merge_chat_requirements(intent: dict[str, Any], current_requirements: dict[str, Any]) -> dict[str, Any]:
    target_spec = current_requirements.get("target_spec", {}) if isinstance(current_requirements.get("target_spec"), dict) else {}
    design_constraints = (
        current_requirements.get("design_constraints", {})
        if isinstance(current_requirements.get("design_constraints"), dict)
        else {}
    )

    parsed_freq = intent.get("parsed_frequency_ghz")
    parsed_bw = intent.get("parsed_bandwidth_mhz")
    parsed_family = intent.get("parsed_antenna_family")
    parsed_patch_shape = intent.get("parsed_patch_shape")
    parsed_substrate_material = intent.get("parsed_substrate_material")
    parsed_conductor_material = intent.get("parsed_conductor_material")

    substrate_material = (
        parsed_substrate_material
        or current_requirements.get("substrate_material")
        or _first_non_empty_string(current_requirements.get("allowed_substrates"))
        or _first_non_empty_string(design_constraints.get("allowed_substrates"))
    )
    conductor_material = (
        parsed_conductor_material
        or current_requirements.get("conductor_material")
        or _first_non_empty_string(current_requirements.get("allowed_materials"))
        or _first_non_empty_string(design_constraints.get("allowed_materials"))
    )
    allowed_substrates = [substrate_material] if substrate_material else _coerce_string_list(current_requirements.get("allowed_substrates")) or _coerce_string_list(design_constraints.get("allowed_substrates"))
    allowed_materials = [conductor_material] if conductor_material else _coerce_string_list(current_requirements.get("allowed_materials")) or _coerce_string_list(design_constraints.get("allowed_materials"))

    merged = {
        "frequency_ghz": parsed_freq if parsed_freq is not None else current_requirements.get("frequency_ghz", target_spec.get("frequency_ghz")),
        "bandwidth_mhz": parsed_bw if parsed_bw is not None else current_requirements.get("bandwidth_mhz", target_spec.get("bandwidth_mhz")),
        "antenna_family": parsed_family if parsed_family is not None else current_requirements.get("antenna_family", target_spec.get("antenna_family")),
        "patch_shape": parsed_patch_shape if parsed_patch_shape is not None else current_requirements.get("patch_shape", target_spec.get("patch_shape")),
        "substrate_material": substrate_material,
        "conductor_material": conductor_material,
        "allowed_substrates": allowed_substrates,
        "allowed_materials": allowed_materials,
        "design_constraints": {
            **(design_constraints if isinstance(design_constraints, dict) else {}),
            "allowed_substrates": allowed_substrates,
            "allowed_materials": allowed_materials,
        },
        "target_spec": {
            **(target_spec if isinstance(target_spec, dict) else {}),
            "frequency_ghz": parsed_freq if parsed_freq is not None else current_requirements.get("frequency_ghz", target_spec.get("frequency_ghz")),
            "bandwidth_mhz": parsed_bw if parsed_bw is not None else current_requirements.get("bandwidth_mhz", target_spec.get("bandwidth_mhz")),
            "antenna_family": parsed_family if parsed_family is not None else current_requirements.get("antenna_family", target_spec.get("antenna_family")),
            "patch_shape": parsed_patch_shape if parsed_patch_shape is not None else current_requirements.get("patch_shape", target_spec.get("patch_shape")),
        },
    }
    return merged


def _missing_required_chat_fields(merged: dict[str, Any]) -> list[str]:
    return [
        field_name
        for field_name in ("frequency_ghz", "bandwidth_mhz", "antenna_family")
        if merged.get(field_name) in (None, "")
    ]


def _pretty_requirement_name(field_name: str) -> str:
    return field_name.replace("_ghz", " (GHz)").replace("_mhz", " (MHz)").replace("_", " ")


def _build_natural_chat_summary(
    *,
    merged: dict[str, Any],
    missing_required: list[str],
    pipeline_requested: bool,
) -> str:
    lines = ["Here’s what I have for your antenna so far:"]

    if merged.get("antenna_family"):
        lines.append(f"- Antenna family: {merged['antenna_family']}")
    if merged.get("patch_shape"):
        lines.append(f"- Patch shape: {merged['patch_shape']}")
    if merged.get("frequency_ghz") is not None:
        lines.append(f"- Resonant frequency target: {merged['frequency_ghz']} GHz")
    if merged.get("bandwidth_mhz") is not None:
        lines.append(f"- Bandwidth target: {merged['bandwidth_mhz']} MHz")
    if merged.get("substrate_material"):
        lines.append(f"- Substrate: {merged['substrate_material']}")
    if merged.get("conductor_material"):
        lines.append(f"- Conductor: {merged['conductor_material']}")

    if len(lines) == 1:
        lines.append("- I have not locked in the design parameters yet.")

    if missing_required:
        missing_text = ", ".join(_pretty_requirement_name(item) for item in missing_required)
        lines.append(f"I still need {missing_text} before the pipeline can start.")
    elif pipeline_requested:
        lines.append("Everything essential is captured, so you can start the pipeline now if you want.")
    else:
        lines.append("If that matches your intent, you can start the pipeline whenever you're ready.")

    return "\n".join(lines)


def _build_fallback_chat_reply(
    *,
    user_message: str,
    merged: dict[str, Any],
    missing_required: list[str],
    supported_families: list[str],
    conductor_materials: list[Any],
    substrate_materials: list[Any],
    frequency_range: dict[str, Any],
    bandwidth_range: dict[str, Any],
) -> str:
    text = user_message.lower()
    parts: list[str] = []

    if any(token in text for token in ("what is", "why", "how", "difference", "explain", "doubt")):
        if "microstrip" in text or "patch" in text:
            parts.append(
                "A microstrip patch is the easiest printed antenna family to start with because it is simple to fabricate, tune, and simulate."
            )
        if "substrate" in text or "fr4" in text or "rogers" in text or "material" in text:
            parts.append(
                "The substrate affects resonance, size, bandwidth, and loss: FR-4 is practical and affordable, while Rogers materials usually give lower RF loss and more stable behavior."
            )
        if "gain" in text or "vswr" in text:
            parts.append(
                "Gain and VSWR are tuning outcomes, so we normally refine them through the optimization loop after the starting geometry is generated."
            )

    if "famil" in text:
        parts.append(f"The main supported families are {', '.join(map(str, supported_families))}.")
    if "capabil" in text or "range" in text:
        parts.append(
            f"The working range is about {frequency_range.get('min')} to {frequency_range.get('max')} GHz, with bandwidth targets from {bandwidth_range.get('min')} to {bandwidth_range.get('max')} MHz."
        )
    if "substrate" in text or "conductor" in text or "material" in text:
        parts.append(
            f"Supported conductor materials include {', '.join(map(str, conductor_materials))}, and substrate choices include {', '.join(map(str, substrate_materials))}."
        )

    captured_parts: list[str] = []
    if merged.get("frequency_ghz") is not None:
        captured_parts.append(f"{merged['frequency_ghz']} GHz")
    if merged.get("bandwidth_mhz") is not None:
        captured_parts.append(f"{merged['bandwidth_mhz']} MHz bandwidth")
    if merged.get("antenna_family"):
        captured_parts.append(f"the {merged['antenna_family']} family")

    if captured_parts:
        parts.append("So far I have " + ", ".join(captured_parts) + ".")

    if missing_required:
        missing_text = ", ".join(_pretty_requirement_name(item) for item in missing_required)
        parts.append(f"To move forward, I still need {missing_text}.")
    elif not parts:
        parts.append("I have enough to move forward and can help explain or refine any design choice before you start the pipeline.")

    return " ".join(parts).strip()


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

    raw_session_id = payload.get("session_id")
    session_id = str(raw_session_id).strip() if raw_session_id is not None and str(raw_session_id).strip() else f"chat-{uuid.uuid4()}"
    current_requirements = _hydrate_chat_requirements(session_id, current_requirements)

    intent = summarize_user_intent(user_message)
    capabilities_catalog = load_capabilities_catalog()
    supported_families = capabilities_catalog.get("supported_families", list_supported_families())
    frequency_range = capabilities_catalog.get("frequency_range_ghz", {})
    bandwidth_range = capabilities_catalog.get("bandwidth_range_mhz", {})
    conductor_materials = capabilities_catalog.get("available_conductor_materials", [])
    substrate_materials = capabilities_catalog.get("available_substrate_materials", [])

    merged = _merge_chat_requirements(intent, current_requirements)
    _store_chat_requirements(session_id, merged)
    missing_required = _missing_required_chat_fields(merged)
    text = user_message.lower()
    pipeline_requested = any(
        marker in text
        for marker in (
            "start pipeline",
            "start the pipeline",
            "run pipeline",
            "begin pipeline",
            "start optimization",
            "go ahead",
            "proceed",
        )
    )

    natural_summary = _build_natural_chat_summary(
        merged=merged,
        missing_required=missing_required,
        pipeline_requested=pipeline_requested,
    )

    assistant_message: str | None = None
    if check_ollama_health(timeout_sec=3):
        llm_text = generate_text(
            system_prompt=(
                "You are a friendly antenna design copilot inside an interactive design studio. "
                "Talk naturally like a helpful engineering assistant, not a rigid form bot. "
                "Your goals are to answer doubts about antennas and CST implementation, gather missing design inputs, "
                "and help the user feel ready to start the pipeline. "
                "Ask at most two concise follow-up questions when important details are missing. "
                "Do not invent unknown specifications. Keep the response conversational and practical."
            ),
            prompt=(
                f"User message: {user_message}\n"
                f"Captured details so far: {merged}\n"
                f"Missing required fields: {missing_required}\n"
                f"Supported families: {supported_families}\n"
                f"Available conductor materials: {conductor_materials}\n"
                f"Available substrate materials: {substrate_materials}\n"
                f"Capability ranges: frequency_ghz={frequency_range}, bandwidth_mhz={bandwidth_range}\n"
                f"Pipeline requested in this turn: {pipeline_requested}\n"
                "If the user is asking a technical question, answer it first in plain language. "
                "If the required details are already present, confirm that naturally and mention they can start the pipeline when ready. "
                "Do not output JSON."
            ),
            timeout_sec=45,
            model_name=OLLAMA_SETTINGS.big_model_name,
        )
        if llm_text:
            assistant_message = f"{llm_text.strip()}\n\n{natural_summary}"

    if assistant_message is None:
        fallback_message = _build_fallback_chat_reply(
            user_message=user_message,
            merged=merged,
            missing_required=missing_required,
            supported_families=list(map(str, supported_families)),
            conductor_materials=list(conductor_materials),
            substrate_materials=list(substrate_materials),
            frequency_range=frequency_range,
            bandwidth_range=bandwidth_range,
        )
        assistant_message = f"{fallback_message}\n\n{natural_summary}" if fallback_message else natural_summary

    return {
        "status": "ok",
        "session_id": session_id,
        "assistant_message": assistant_message,
        "intent_summary": intent,
        "requirements": merged,
        "captured_details": merged,
        "missing_requirements": missing_required,
        "ready_to_start_pipeline": len(missing_required) == 0,
        "pipeline_intent_detected": pipeline_requested,
        "natural_summary": natural_summary,
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
    except V2CommandValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "V2_COMMAND_VALIDATION_FAILED",
                "message": str(exc),
                "details": exc.as_detail(),
            },
        ) from exc
    except UnsupportedAntennaFamilyError as exc:
        raise HTTPException(status_code=422, detail={"error_code": "FAMILY_NOT_SUPPORTED", "message": str(exc)}) from exc
    except FamilyProfileConstraintError as exc:
        raise HTTPException(status_code=422, detail={"error_code": "FAMILY_PROFILE_CONSTRAINT_FAILED", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _process_client_result(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        validate_contract("client_feedback", payload)
        result = brain.process_feedback(payload)
        if "next_command_package" in result:
            validate_contract("command_package", result["next_command_package"])
        return result
    except ContractValidationError as exc:
        raise HTTPException(status_code=422, detail={"error_code": "SCHEMA_VALIDATION_FAILED", "message": str(exc)}) from exc
    except V2CommandValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "V2_COMMAND_VALIDATION_FAILED",
                "message": str(exc),
                "details": exc.as_detail(),
            },
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error_code": "SESSION_NOT_FOUND", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error_code": "FEEDBACK_PROCESSING_FAILED", "message": str(exc)}) from exc


@app.post("/api/v1/client-feedback")
def client_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    return _process_client_result(payload)


@app.post("/api/v1/result")
def ingest_result(payload: dict[str, Any]) -> dict[str, Any]:
    return _process_client_result(payload)


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
        current_command_package = session.get("current_command_package")
        design_id = None
        if isinstance(current_command_package, dict):
            design_id = current_command_package.get("design_id")

        return {
            "session_id": session_id,
            "trace_id": session.get("trace_id"),
            "design_id": design_id,
            "status": session.get("status", "unknown"),
            "current_stage": session.get("status", "unknown"),
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
            "command_package": current_command_package,
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
