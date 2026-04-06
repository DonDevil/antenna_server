from __future__ import annotations

import json
from typing import Any

import httpx

from config import OLLAMA_SETTINGS

_HEALTH_PROBE_TIMEOUT_SEC = 3


def check_ollama_health(timeout_sec: int = _HEALTH_PROBE_TIMEOUT_SEC) -> bool:
    """Return True only if Ollama is reachable within the given timeout."""
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.get(f"{OLLAMA_SETTINGS.base_url}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def generate_json(
    *,
    prompt: str,
    system_prompt: str,
    timeout_sec: int | None = None,
) -> dict[str, Any] | None:
    timeout = timeout_sec or int(OLLAMA_SETTINGS.timeout_sec)
    payload = {
        "model": OLLAMA_SETTINGS.model_name,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "15m",
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{OLLAMA_SETTINGS.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return None

    text = data.get("response")
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def generate_text(
    *,
    prompt: str,
    system_prompt: str,
    timeout_sec: int | None = None,
) -> str | None:
    timeout = timeout_sec or int(OLLAMA_SETTINGS.timeout_sec)
    payload = {
        "model": OLLAMA_SETTINGS.model_name,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "keep_alive": "15m",
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{OLLAMA_SETTINGS.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return None

    text = data.get("response")
    if not isinstance(text, str):
        return None
    text = text.strip()
    return text or None


def warmup_model(timeout_sec: int | None = None) -> bool:
    if not check_ollama_health(timeout_sec=_HEALTH_PROBE_TIMEOUT_SEC):
        return False
    text = generate_text(
        prompt="Reply with READY only.",
        system_prompt="Warm the model and reply with READY only.",
        timeout_sec=timeout_sec,
    )
    return text is not None