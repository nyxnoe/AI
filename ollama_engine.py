"""
ollama_engine.py
----------------
Wrapper around the local Ollama HTTP API.
Supports both streaming and non-streaming modes.

Install Ollama:  brew install ollama
Pull a model:    ollama pull llama3
Start server:    ollama serve   (runs on http://localhost:11434)
"""

from __future__ import annotations
import json
import re
import logging
from typing import Generator
import requests

logger = logging.getLogger(__name__)

OLLAMA_URL        = "http://localhost:11434/api/generate"
OLLAMA_MODEL      = "llama3"       # default; caller can override
TIMEOUT_STREAM    = 120
TIMEOUT_BLOCKING  = 60


def run_ollama(prompt: str, model: str = OLLAMA_MODEL) -> dict | None:
    """Non-streaming call — returns parsed JSON result or None on failure."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=TIMEOUT_BLOCKING,
        )
        if response.status_code != 200:
            logger.warning(f"Ollama HTTP {response.status_code}: {response.text[:120]}")
            return None
        text = response.json().get("response", "")
        return _extract_json(text)
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not reachable — is 'ollama serve' running?")
        return None
    except requests.exceptions.Timeout:
        logger.warning(f"Ollama timed out after {TIMEOUT_BLOCKING}s")
        return None
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return None


def stream_ollama(prompt: str, model: str = OLLAMA_MODEL) -> Generator[str, None, dict | None]:
    """
    Streaming generator — yields text deltas, returns final parsed JSON via StopIteration.

    Usage in Flask:
        gen = stream_ollama(prompt)
        try:
            while True:
                delta = next(gen)
                yield f"data: {json.dumps({'delta': delta})}\\n\\n"
        except StopIteration as e:
            result = e.value   # dict | None
    """
    full_text = ""
    result    = None
    try:
        with requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=TIMEOUT_STREAM,
        ) as resp:
            if resp.status_code != 200:
                logger.warning(f"Ollama stream HTTP {resp.status_code}")
                return None

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    full_text += token
                    yield token
                if chunk.get("done"):
                    break

        result = _extract_json(full_text)

    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not reachable during stream")
    except requests.exceptions.Timeout:
        logger.warning(f"Ollama stream timed out after {TIMEOUT_STREAM}s")
    except Exception as e:
        logger.error(f"Ollama stream error: {e}")

    return result


def is_available() -> bool:
    """Quick health-check — True if Ollama server is reachable."""
    try:
        r = requests.get("http://localhost:11434", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from a text blob."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        logger.warning("Ollama response contained no JSON block")
        return None
    try:
        return json.loads(match.group())
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Ollama JSON parse error: {e}")
        return None