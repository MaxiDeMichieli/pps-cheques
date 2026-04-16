"""Modelos de datos para cheques."""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class DatosCheque:
    """Datos extraidos de un cheque."""
    # Monto (OCR heuristico)
    monto: float | None = None
    monto_raw: str = ""
    monto_score: float = 0.0

    # Monto (LLM)
    monto_llm_confidence: float | None = None

    # Monto manuscrito (del campo "LA CANTIDAD DE PESOS")
    monto_manuscrito: float | None = None
    monto_manuscrito_raw: str | None = None
    monto_manuscrito_score: float = 2.0
    monto_manuscrito_confidence_ocr: float = 0.0
    monto_inconsistencia_pct: float | None = None
    monto_manuscrito_zona_base64: str | None = None
    validacion_alineada: bool = False

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


def guardar_cheques_json(cheques: list[DatosCheque], path: str):
    """Guarda lista de cheques en archivo JSON."""
    data = {
        "total_cheques": len(cheques),
        "cheques": [c.to_dict() for c in cheques]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cargar_cheques_json(path: str) -> list[dict]:
    """Carga cheques desde archivo JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cheques", [])
