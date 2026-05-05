"""Tipos de evento SSE y registro de colas por run."""

import asyncio
from typing import Literal

EventType = Literal[
    "run_started",
    "pdf_loaded",
    "page_processed",
    "cheque_detected",
    "cheque_extracted",
    "run_completed",
    "error",
]

# Sentinel que indica al generador SSE que el stream terminó.
END_OF_STREAM = ("__end__", None)

_queues: dict[int, asyncio.Queue] = {}


def get_queue(run_id: int) -> asyncio.Queue:
    """Devuelve (creando si hace falta) la cola asociada a un run."""
    if run_id not in _queues:
        _queues[run_id] = asyncio.Queue()
    return _queues[run_id]


def drop_queue(run_id: int) -> None:
    _queues.pop(run_id, None)
