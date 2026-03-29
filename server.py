from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from app.core.json_contracts import ContractValidationError, validate_contract
from app.core.schemas import OptimizeRequest, OptimizeResponse
from central_brain import CentralBrain
from config import API_SETTINGS


brain = CentralBrain()
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
        return {
            "status": "received",
            "message": "feedback accepted",
            "payload_keys": list(payload.keys()),
        }
    except ContractValidationError as exc:
        raise HTTPException(status_code=422, detail={"error_code": "SCHEMA_VALIDATION_FAILED", "message": str(exc)}) from exc
