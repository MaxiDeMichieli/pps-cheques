"""Pydantic schemas para request/response de la API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    pdf_filename: str
    status: str
    total_cheques: int | None
    error: str | None


class ChequeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    pagina: int
    indice_en_pagina: int
    imagen_path: str

    monto: float | None
    monto_raw: str
    monto_score: float
    monto_llm_confidence: float | None

    fecha_emision: str | None
    fecha_emision_raw: str | None
    fecha_emision_llm_confidence: float | None

    fecha_pago: str | None
    fecha_pago_raw: str | None
    fecha_pago_llm_confidence: float | None

    sucursal: str | None
    sucursal_raw: str | None
    sucursal_score: float

    numero_sucursal: str | None
    numero_cheque: str | None
    numero_cuenta: str | None

    cuit_librador: str | None
    nombre_librador: str | None

    edited_fields: dict
    extracted: bool


class ChequePatch(BaseModel):
    """Campos editables vía PATCH. Todos opcionales."""

    monto: float | None = None
    fecha_emision: str | None = None
    fecha_pago: str | None = None
    sucursal: str | None = None
    numero_sucursal: str | None = None
    numero_cheque: str | None = None
    numero_cuenta: str | None = None
    cuit_librador: str | None = None
    nombre_librador: str | None = None


class RunCreated(BaseModel):
    run_id: int
    pdf_filename: str
