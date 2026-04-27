"""Extractor de campos del librador de un cheque (CUIT y nombre).

Estrategia:
  Ancla CUIT: busca el token que empieza con 'CUIT', toma todos los tokens
  de la misma linea (ventana cy).
  - cuit_librador: primer token de la linea que coincide con el patron de
    CUIT argentino (XX-XXXXXXXX-X, XXX-XXXXXXXX-X, o 11 digitos).
  - nombre_librador: tokens a la derecha del token CUIT, unidos por espacio.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..ocr.ocr_readers import OCRReader, OCRResult
from .fecha_extractor import _VENTANA_CY

logger = logging.getLogger(__name__)

_CUIT_ANCHOR_RE = re.compile(r'^cuit', re.IGNORECASE)

# Matches formatted (XX-XXXXXXXX-X, XXX-XXXXXXXX-X) or bare 11-digit CUIT.
# Uses search so it handles OCR glue like "30-69078321-1-" or "CUIT30-...".
_CUIT_VALUE_RE = re.compile(r'\d{2,3}-\d{7,8}-\d|\d{11}')

_DEBUG_ZONA = "campos_librador_zona.png"


@dataclass
class CamposLibradorResult:
    cuit: str | None
    nombre: str | None
    tokens: list[OCRResult] = field(default_factory=list)


class CamposLibradorExtractor:
    """Extrae CUIT y nombre del librador de un cheque via OCR."""

    def __init__(self, ocr_reader: OCRReader):
        self._ocr = ocr_reader

    def extraer(
        self,
        cheque_img: np.ndarray,
        debug_dir: Path | None = None,
    ) -> CamposLibradorResult:
        h, w = cheque_img.shape[:2]
        y0 = int(h * 0.40)
        y1 = int(h * 0.85)
        x0 = int(w * 0.10)
        x1 = int(w * 0.90)
        zona = cheque_img[y0:y1, x0:x1]
        tokens = self._ocr.read(zona)
        logger.info("CamposLibrador tokens: %s", [(t.text, round(t.cy, 3)) for t in tokens])

        anchor = next(
            (t for t in tokens if _CUIT_ANCHOR_RE.match(t.text.strip())),
            None,
        )
        if anchor is None:
            logger.info("CamposLibrador CUIT anchor: no encontrado")
            return CamposLibradorResult(cuit=None, nombre=None, tokens=tokens)

        logger.info("CamposLibrador CUIT anchor: token=%r cy=%.3f", anchor.text, anchor.cy)
        line_tokens = sorted(
            [t for t in tokens if abs(t.cy - anchor.cy) < _VENTANA_CY],
            key=lambda t: t.cx,
        )
        logger.info("CamposLibrador linea tokens: %s", [t.text for t in line_tokens])

        if debug_dir is not None:
            Image.fromarray(zona).save(debug_dir / _DEBUG_ZONA)

        cuit: str | None = None
        cuit_token: OCRResult | None = None
        for t in line_tokens:
            m = _CUIT_VALUE_RE.search(t.text)
            if m:
                cuit = m.group(0)
                cuit_token = t
                logger.info("CamposLibrador CUIT extraido: %r de token %r", cuit, t.text)
                break

        nombre: str | None = None
        if cuit_token is not None:
            nombre_tokens = [t for t in line_tokens if t.cx > cuit_token.cx]
            if nombre_tokens:
                nombre = " ".join(t.text for t in nombre_tokens)
                logger.info("CamposLibrador nombre extraido: %r", nombre)

        if cuit is None:
            logger.info("CamposLibrador CUIT no encontrado en linea: %s", [t.text for t in line_tokens])

        return CamposLibradorResult(cuit=cuit, nombre=nombre, tokens=line_tokens)
