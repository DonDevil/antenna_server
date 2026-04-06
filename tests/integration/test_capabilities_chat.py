from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import server
from app.core.session_store import SessionStore


def _build_test_client(tmp_path: Path) -> TestClient:
    test_store = SessionStore(base_dir=tmp_path / "sessions")
    server.session_store = test_store
    server.brain.session_store = test_store
    return TestClient(server.app)


def test_capabilities_endpoint_returns_ranges_and_materials(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    data = response.json()

    assert "frequency_range_ghz" in data
    assert "bandwidth_range_mhz" in data
    assert "available_conductor_materials" in data
    assert "available_substrate_materials" in data
    assert isinstance(data["available_conductor_materials"], list)
    assert isinstance(data["available_substrate_materials"], list)


def test_chat_answers_material_and_capability_questions(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "What are your frequency and bandwidth capabilities and available substrate materials?",
            "requirements": {},
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert "assistant_message" in data
    assert isinstance(data["assistant_message"], str)
    assert "capabilities" in data
    assert "available_substrate_materials" in data["capabilities"]
    assert isinstance(data["capabilities"]["available_substrate_materials"], list)


def test_health_endpoint_reports_dependency_statuses(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["ann_status"] in {"available", "loading", "none"}
    assert data["llm_status"] in {"available", "loading", "none"}
    assert isinstance(data["ann_model_ready"], bool)
    assert isinstance(data["ollama_reachable"], bool)
    assert isinstance(data["llm_model"], str)
    assert data["llm_model"]
