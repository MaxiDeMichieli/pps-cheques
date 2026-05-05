"""Endpoints relacionados con Cheque (PATCH, imagen)."""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..db import get_session
from ..models_db import Cheque
from ..schemas import ChequeOut, ChequePatch

router = APIRouter(prefix="/api/cheques", tags=["cheques"])


@router.patch("/{cheque_id}", response_model=ChequeOut)
def actualizar_cheque(
    cheque_id: int,
    patch: ChequePatch,
    db: Session = Depends(get_session),
):
    cheque = db.get(Cheque, cheque_id)
    if cheque is None:
        raise HTTPException(status_code=404, detail="Cheque no encontrado")

    cambios = patch.model_dump(exclude_unset=True)
    edited = dict(cheque.edited_fields or {})
    now = datetime.utcnow().isoformat(timespec="seconds")

    for campo, valor in cambios.items():
        setattr(cheque, campo, valor)
        edited[campo] = now

    cheque.edited_fields = edited
    db.commit()
    db.refresh(cheque)
    return cheque


@router.get("/{cheque_id}/imagen")
def get_imagen(cheque_id: int, db: Session = Depends(get_session)):
    cheque = db.get(Cheque, cheque_id)
    if cheque is None:
        raise HTTPException(status_code=404, detail="Cheque no encontrado")

    img_path = Path(cheque.imagen_path)
    if not img_path.is_absolute():
        img_path = Path.cwd() / img_path

    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada en disco")

    return FileResponse(img_path, media_type="image/png")
