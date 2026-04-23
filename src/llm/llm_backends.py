"""Backends HTTP para llamadas a LLMs.

Define una interfaz comun (LLMBackend) y sus implementaciones concretas,
siguiendo el mismo patron ABC que ocr_readers.py.
"""

import base64
import io
import logging
import time
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Interfaz base para backends de LLM."""

    @abstractmethod
    def chat(self, messages: list[dict]) -> str | None:
        """Envía mensajes al LLM y retorna el texto de la respuesta, o None si falla."""
        pass


class OllamaBackend(LLMBackend):
    """Backend para servidor local Ollama (POST /api/chat)."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def chat(self, messages: list[dict]) -> str | None:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            t0 = time.perf_counter()
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            elapsed = time.perf_counter() - t0
            response.raise_for_status()
            logger.info("OllamaBackend respondió en %.1fs", elapsed)
            return response.json()["message"]["content"]
        except httpx.ConnectError:
            logger.warning("Ollama no disponible en %s", self._base_url)
            return None
        except Exception as exc:
            logger.warning("Error llamando a Ollama: %s", exc)
            return None

    def chat_vision(self, messages: list[dict], images: list) -> str | None:
        """Envía mensajes + imágenes (numpy arrays) al LLM multimodal y retorna texto."""
        from PIL import Image as PILImage
        import numpy as np

        encoded: list[str] = []
        for img in images:
            pil = PILImage.fromarray(img.astype("uint8")) if isinstance(img, np.ndarray) else img
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        msgs = list(messages)
        if msgs and msgs[-1].get("role") == "user":
            msgs[-1] = dict(msgs[-1], images=encoded)

        payload = {
            "model": self._model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            t0 = time.perf_counter()
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            elapsed = time.perf_counter() - t0
            response.raise_for_status()
            logger.info("OllamaBackend (vision) respondió en %.1fs", elapsed)
            return response.json()["message"]["content"]
        except httpx.ConnectError:
            logger.warning("Ollama no disponible en %s", self._base_url)
            return None
        except Exception as exc:
            logger.warning("Error llamando a Ollama (vision): %s", exc)
            return None
