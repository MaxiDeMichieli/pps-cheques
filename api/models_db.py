"""ORM models para Run y Cheque."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    pdf_filename: Mapped[str] = mapped_column(String, nullable=False)
    pdf_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|running|completed|failed
    total_cheques: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)

    cheques: Mapped[list["Cheque"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="Cheque.id"
    )


class Cheque(Base):
    __tablename__ = "cheques"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True, nullable=False)
    pagina: Mapped[int] = mapped_column(Integer, nullable=False)
    indice_en_pagina: Mapped[int] = mapped_column(Integer, nullable=False)
    imagen_path: Mapped[str] = mapped_column(String, nullable=False)

    monto: Mapped[float | None] = mapped_column(Float, nullable=True)
    monto_raw: Mapped[str] = mapped_column(String, default="")
    monto_score: Mapped[float] = mapped_column(Float, default=0.0)
    monto_llm_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    fecha_emision: Mapped[str | None] = mapped_column(String, nullable=True)
    fecha_emision_raw: Mapped[str | None] = mapped_column(String, nullable=True)
    fecha_emision_llm_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    fecha_pago: Mapped[str | None] = mapped_column(String, nullable=True)
    fecha_pago_raw: Mapped[str | None] = mapped_column(String, nullable=True)
    fecha_pago_llm_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    sucursal: Mapped[str | None] = mapped_column(String, nullable=True)
    sucursal_raw: Mapped[str | None] = mapped_column(String, nullable=True)
    sucursal_score: Mapped[float] = mapped_column(Float, default=0.0)

    numero_sucursal: Mapped[str | None] = mapped_column(String, nullable=True)
    numero_cheque: Mapped[str | None] = mapped_column(String, nullable=True)
    numero_cuenta: Mapped[str | None] = mapped_column(String, nullable=True)

    cuit_librador: Mapped[str | None] = mapped_column(String, nullable=True)
    nombre_librador: Mapped[str | None] = mapped_column(String, nullable=True)

    edited_fields: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    extracted: Mapped[bool] = mapped_column(Boolean, default=False)

    run: Mapped[Run] = relationship(back_populates="cheques")
