"""Bridge entre el pipeline sync de src/ y la cola async de eventos SSE.

Cada run tiene su propia ``asyncio.Queue``. Un thread worker corre el pipeline
y emite eventos a esa cola vía ``loop.call_soon_threadsafe``. El endpoint SSE
consume los eventos.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from fastapi import BackgroundTasks

from src.extractors.cheque_extractor import ChequeExtractor
from src.llm.llm_backends import OllamaBackend
from src.llm.llm_validator import LLMValidator
from src.models import DatosCheque
from src.ocr.ocr_readers import DocTRReader
from src.pipeline import procesar_pdf

from .db import SessionLocal
from .events import get_queue
from .models_db import Cheque, Run

logger = logging.getLogger(__name__)


# El extractor (modelos OCR/LLM) es caro de inicializar. Cacheamos uno global.
_extractor: ChequeExtractor | None = None


def _get_extractor() -> ChequeExtractor:
    global _extractor
    if _extractor is None:
        logger.info("Inicializando OCR (docTR)...")
        ocr_reader = DocTRReader()
        try:
            backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
            llm = LLMValidator(backend=backend)
            logger.info("LLM Ollama detectado")
        except Exception as exc:
            logger.warning("LLM no disponible (%s); siguiendo sin LLM", exc)
            llm = None
        _extractor = ChequeExtractor(ocr_reader, llm_validator=llm)
    return _extractor


def schedule_run(background: BackgroundTasks, run_id: int, pdf_path: Path) -> None:
    """Programa la ejecución del pipeline en una task de fondo."""
    background.add_task(_run_pipeline_async, run_id, pdf_path)


async def _run_pipeline_async(run_id: int, pdf_path: Path) -> None:
    queue = get_queue(run_id)
    loop = asyncio.get_running_loop()

    def emit(event_type: str, data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

    _set_run_status(run_id, "running")
    emit("run_started", {"run_id": run_id})

    started = time.perf_counter()
    try:
        await asyncio.to_thread(_run_pipeline_sync, run_id, pdf_path, emit)
    except Exception as exc:
        logger.exception("Pipeline run %d falló", run_id)
        _set_run_error(run_id, str(exc))
        emit("error", {"message": str(exc)})
        emit("run_completed", {"total_cheques": 0, "duration_s": 0, "status": "failed"})
        return

    total = _finalize_run(run_id)
    emit(
        "run_completed",
        {
            "total_cheques": total,
            "duration_s": round(time.perf_counter() - started, 2),
            "status": "completed",
        },
    )


def _run_pipeline_sync(run_id: int, pdf_path: Path, emit) -> None:
    """Corre el pipeline sync. Persiste cada cheque en DB y emite eventos."""
    extractor = _get_extractor()

    cheque_ids: dict[tuple[int, int], int] = {}

    def on_pdf_loaded(total_paginas: int) -> None:
        emit("pdf_loaded", {"total_pages": total_paginas})

    def on_cheque_detected(pagina: int, idx: int, ruta_img: str) -> None:
        with SessionLocal() as db:
            cheque = Cheque(
                run_id=run_id,
                pagina=pagina,
                indice_en_pagina=idx,
                imagen_path=ruta_img,
            )
            db.add(cheque)
            db.commit()
            db.refresh(cheque)
            cheque_ids[(pagina, idx)] = cheque.id
            emit(
                "cheque_detected",
                {
                    "cheque_id": cheque.id,
                    "pagina": pagina,
                    "indice": idx,
                    "imagen_url": f"/api/cheques/{cheque.id}/imagen",
                },
            )

    def on_cheque_extracted(pagina: int, idx: int, datos: DatosCheque) -> None:
        cheque_id = cheque_ids[(pagina, idx)]
        with SessionLocal() as db:
            cheque = db.get(Cheque, cheque_id)
            if cheque is None:
                return
            _apply_datos(cheque, datos)
            db.commit()
            db.refresh(cheque)
            emit(
                "cheque_extracted",
                _cheque_payload(cheque),
            )

    procesar_pdf(
        str(pdf_path),
        extractor,
        output_dir="output",
        on_pdf_loaded=on_pdf_loaded,
        on_cheque_detected=on_cheque_detected,
        on_cheque_extracted=on_cheque_extracted,
    )


def _apply_datos(cheque: Cheque, datos: DatosCheque) -> None:
    cheque.monto = datos.monto
    cheque.monto_raw = datos.monto_raw
    cheque.monto_score = datos.monto_score
    cheque.monto_llm_confidence = datos.monto_llm_confidence
    cheque.fecha_emision = datos.fecha_emision
    cheque.fecha_emision_raw = datos.fecha_emision_raw
    cheque.fecha_emision_llm_confidence = datos.fecha_emision_llm_confidence
    cheque.fecha_pago = datos.fecha_pago
    cheque.fecha_pago_raw = datos.fecha_pago_raw
    cheque.fecha_pago_llm_confidence = datos.fecha_pago_llm_confidence
    cheque.sucursal = datos.sucursal
    cheque.sucursal_raw = datos.sucursal_raw
    cheque.sucursal_score = datos.sucursal_score
    cheque.numero_sucursal = datos.numero_sucursal
    cheque.numero_cheque = datos.numero_cheque
    cheque.numero_cuenta = datos.numero_cuenta
    cheque.cuit_librador = datos.cuit_librador
    cheque.nombre_librador = datos.nombre_librador
    cheque.extracted = True


def _cheque_payload(cheque: Cheque) -> dict:
    """Serializa Cheque para el evento SSE (mismas keys que ChequeOut)."""
    return {
        "id": cheque.id,
        "run_id": cheque.run_id,
        "pagina": cheque.pagina,
        "indice_en_pagina": cheque.indice_en_pagina,
        "imagen_path": cheque.imagen_path,
        "monto": cheque.monto,
        "monto_raw": cheque.monto_raw,
        "monto_score": cheque.monto_score,
        "monto_llm_confidence": cheque.monto_llm_confidence,
        "fecha_emision": cheque.fecha_emision,
        "fecha_emision_raw": cheque.fecha_emision_raw,
        "fecha_emision_llm_confidence": cheque.fecha_emision_llm_confidence,
        "fecha_pago": cheque.fecha_pago,
        "fecha_pago_raw": cheque.fecha_pago_raw,
        "fecha_pago_llm_confidence": cheque.fecha_pago_llm_confidence,
        "sucursal": cheque.sucursal,
        "sucursal_raw": cheque.sucursal_raw,
        "sucursal_score": cheque.sucursal_score,
        "numero_sucursal": cheque.numero_sucursal,
        "numero_cheque": cheque.numero_cheque,
        "numero_cuenta": cheque.numero_cuenta,
        "cuit_librador": cheque.cuit_librador,
        "nombre_librador": cheque.nombre_librador,
        "edited_fields": cheque.edited_fields or {},
        "extracted": cheque.extracted,
    }


def _set_run_status(run_id: int, status: str) -> None:
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if run is not None:
            run.status = status
            db.commit()


def _set_run_error(run_id: int, message: str) -> None:
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if run is not None:
            run.status = "failed"
            run.error = message
            db.commit()


def _finalize_run(run_id: int) -> int:
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if run is None:
            return 0
        total = db.query(Cheque).filter(Cheque.run_id == run_id).count()
        run.total_cheques = total
        run.status = "completed"
        db.commit()
        return total
