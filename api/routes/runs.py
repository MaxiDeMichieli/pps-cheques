"""Endpoints relacionados con Run (upload, listado, eventos, export)."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ..db import OUTPUT_DIR, get_session
from ..events import drop_queue, get_queue
from ..models_db import Cheque, Run
from ..schemas import ChequeOut, RunCreated, RunOut

router = APIRouter(prefix="/api/runs", tags=["runs"])

UPLOADS_DIR = OUTPUT_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


@router.post("", response_model=RunCreated)
async def crear_run(
    background: BackgroundTasks,
    pdf: UploadFile = File(...),
    db: Session = Depends(get_session),
):
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Archivo debe ser un PDF")

    safe_name = f"{uuid4().hex}_{Path(pdf.filename).name}"
    dest = UPLOADS_DIR / safe_name
    contents = await pdf.read()
    dest.write_bytes(contents)

    run = Run(pdf_filename=pdf.filename, pdf_path=str(dest), status="pending")
    db.add(run)
    db.commit()
    db.refresh(run)

    # Fase 2 enchufa el pipeline real aquí
    from ..pipeline_runner import schedule_run
    schedule_run(background, run.id, dest)

    return RunCreated(run_id=run.id, pdf_filename=pdf.filename)


@router.get("", response_model=list[RunOut])
def listar_runs(db: Session = Depends(get_session)):
    return db.query(Run).order_by(Run.created_at.desc()).all()


@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: int, db: Session = Depends(get_session)):
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run no encontrado")
    return run


@router.get("/{run_id}/cheques", response_model=list[ChequeOut])
def listar_cheques(run_id: int, db: Session = Depends(get_session)):
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run no encontrado")
    return (
        db.query(Cheque)
        .filter(Cheque.run_id == run_id)
        .order_by(Cheque.pagina, Cheque.indice_en_pagina)
        .all()
    )


@router.get("/{run_id}/export.json")
def exportar_run(run_id: int, db: Session = Depends(get_session)):
    """Exporta el run en el mismo formato que ``output/cheques.json`` legacy."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run no encontrado")
    cheques = (
        db.query(Cheque)
        .filter(Cheque.run_id == run_id)
        .order_by(Cheque.pagina, Cheque.indice_en_pagina)
        .all()
    )
    return {
        "fecha_proceso": run.created_at.isoformat(timespec="seconds"),
        "nombre_archivo": run.pdf_filename,
        "cheques": [
            {
                "monto": c.monto,
                "monto_raw": c.monto_raw,
                "monto_score": c.monto_score,
                "monto_llm_confidence": c.monto_llm_confidence,
                "fecha_emision": c.fecha_emision,
                "fecha_emision_raw": c.fecha_emision_raw,
                "fecha_emision_llm_confidence": c.fecha_emision_llm_confidence,
                "fecha_pago": c.fecha_pago,
                "fecha_pago_raw": c.fecha_pago_raw,
                "fecha_pago_llm_confidence": c.fecha_pago_llm_confidence,
                "sucursal": c.sucursal,
                "sucursal_raw": c.sucursal_raw,
                "sucursal_score": c.sucursal_score,
                "numero_sucursal": c.numero_sucursal,
                "numero_cheque": c.numero_cheque,
                "numero_cuenta": c.numero_cuenta,
                "cuit_librador": c.cuit_librador,
                "nombre_librador": c.nombre_librador,
                "imagen_path": c.imagen_path,
                "pdf_origen": run.pdf_filename,
                "pagina": c.pagina,
                "indice_en_pagina": c.indice_en_pagina,
            }
            for c in cheques
        ],
    }


@router.get("/{run_id}/events")
async def stream_eventos(run_id: int, db: Session = Depends(get_session)):
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run no encontrado")

    queue = get_queue(run_id)

    async def gen():
        # No descartamos la cola al desconectar el cliente: el worker del
        # pipeline puede seguir empujando eventos y queremos que un nuevo
        # cliente pueda reconectarse sin perderlos. La cola vive mientras
        # vive el proceso del backend.
        while True:
            try:
                event_type, data = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "{}"}
                continue
            yield {"event": event_type, "data": json.dumps(data, default=str)}
            if event_type == "run_completed":
                drop_queue(run_id)
                break

    return EventSourceResponse(gen())
