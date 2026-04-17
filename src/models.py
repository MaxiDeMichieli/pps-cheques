"""Modelos de datos para cheques."""

from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class DatosCheque:
    """Datos extraidos de un cheque."""
    # Monto (OCR heuristico)
    monto: float | None = None
    monto_raw: str = ""
    monto_score: float = 0.0

    # Monto (LLM)
    monto_llm_confidence: float | None = None

    # Fecha de emision (LLM)
    fecha_emision: str | None = None        # ISO: "YYYY-MM-DD"
    fecha_emision_raw: str | None = None    # tal como lo leyo el LLM
    fecha_emision_llm_confidence: float | None = None

    # Metadata
    imagen_path: str = ""
    pdf_origen: str = ""
    pagina: int = 0
    indice_en_pagina: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def guardar_cheques_json(runs: list[dict], path: str):
    """Appends run entries to the JSON array, preserving existing entries.

    Each run is a dict with keys: fecha_proceso, nombre_archivo, cheques.
    """
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except json.JSONDecodeError:
                existing = []

    existing.extend(runs)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def cargar_cheques_json(path: str) -> list[dict]:
    """Carga runs desde archivo JSON. Cada run tiene fecha_proceso, nombre_archivo y cheques."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
