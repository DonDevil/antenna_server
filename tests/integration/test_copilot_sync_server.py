from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import copilot_sync_server


def _build_test_client(tmp_path: Path) -> TestClient:
    copilot_sync_server.SYNC_STORE_PATH = tmp_path / "copilot_sync_messages.json"
    return TestClient(copilot_sync_server.app)


def test_copilot_sync_message_roundtrip(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    post_response = client.post(
        "/api/v1/copilot-sync/message",
        json={
            "sender": "server-copilot",
            "topic": "materials",
            "text": "Need parity guidance for the client helper.",
            "related_files": ["app/antenna/materials.py"],
            "questions": ["Can you mirror the fallback behavior?"],
            "proposed_actions": ["Add get_conductor_properties()"],
        },
    )
    assert post_response.status_code == 200

    list_response = client.get("/api/v1/copilot-sync/messages")
    assert list_response.status_code == 200
    data = list_response.json()

    assert data["count"] == 1
    assert data["messages"][0]["topic"] == "materials"
    assert data["messages"][0]["questions"] == ["Can you mirror the fallback behavior?"]


def test_bootstrap_contains_current_file_guidance(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    response = client.get("/api/v1/copilot-sync/bootstrap")
    assert response.status_code == 200
    data = response.json()

    assert data["current_file"]["path"] == "app/antenna/materials.py"
    assert data["current_file"]["future_function"]["name"] == "get_conductor_properties"
    assert "get_substrate_properties" in data["current_file"]["functions"]
