"""Extractor de zona de fecha de emision de cheques.

Lee los tokens OCR de la franja donde aparece la linea de ciudad + fecha,
usando el token "EL" como ancla inferior (siempre esta en la linea siguiente
al "CIUDAD, DD DE MES DE AAAA" de emision).
"""

import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..ocr.ocr_readers import OCRReader, OCRResult

logger = logging.getLogger(__name__)

_MESES_NOMBRES = {
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
}


def _es_token_fecha(text: str) -> bool:
    """Indica si un token pertenece tipicamente a la linea de ciudad+fecha."""
    t = text.strip().lower()
    if t in _MESES_NOMBRES:
        return True
    # Año: 4 digitos entre 2000-2099
    if re.match(r'^20\d{2}$', t):
        return True
    return False


_DEBUG_FECHA_ZONA = "fecha_zona.png"

# Acepta 'EL', 'El', 'el', 'L', 'l' — errores comunes de OCR para el token "EL"
_EL_RE = re.compile(r'^[Ee]?[Ll]$')


class FechaEmisionExtractor:
    """Lee los tokens OCR de la zona de fecha de emision."""

    def __init__(self, ocr_reader: OCRReader):
        self._ocr = ocr_reader

    def leer_tokens(self, cheque_img: np.ndarray, debug_dir: Path | None = None) -> list[OCRResult]:
        """Devuelve tokens OCR de la linea de fecha de emision.

        Paso 1: OCR sobre una franja amplia para localizar "EL" y estimar la altura de linea.
        Paso 2: Recortar exactamente la linea anterior al "EL" y re-ejecutar OCR.
        Fallback: filtrar del scan amplio por tokens de mes/año conocido.
        """
        h, w = cheque_img.shape[:2]
        scan_h = int(h * 0.55)
        scan_x0 = int(w * 0.10)
        zona = cheque_img[0:scan_h, scan_x0:w]
        tokens_scan = self._ocr.read(zona)

        el_token = next(
            (t for t in tokens_scan if _EL_RE.match(t.text.strip()) and t.cy > 0.35),
            None,
        )
        logger.info(
            "EL token: %s",
            f"text={el_token.text!r} cy={el_token.cy:.3f} h={el_token.height:.3f}" if el_token else "no encontrado",
        )
        logger.info("Tokens scan: %s", [(t.text, round(t.cy, 3)) for t in tokens_scan])

        if el_token:
            tokens = self._crop_fecha(cheque_img, tokens_scan, el_token, w, scan_h, scan_x0, debug_dir)
            if tokens:
                return tokens

        return self._fallback_fecha(tokens_scan, zona, debug_dir)

    def _crop_fecha(
        self,
        cheque_img: np.ndarray,
        tokens_scan: list[OCRResult],
        el_token: OCRResult,
        w: int,
        scan_h: int,
        scan_x0: int,
        debug_dir: Path | None,
    ) -> list[OCRResult]:
        """Recorta la linea de fecha usando el token EL como ancla."""
        el_abs_y = int(el_token.cy * scan_h)
        # Floor de 0.07 por si el OCR devuelve un caracter parcial ('L' en vez de 'EL')
        # con bounding box incorrectamente pequeño. Espaciado real ~1.5x altura de token.
        token_h_norm = max(el_token.height, 0.07)
        token_h_px = int(token_h_norm * scan_h)
        centro_fecha = el_abs_y - int(token_h_px * 1.5)
        y0 = max(0, centro_fecha - token_h_px)
        y1 = min(scan_h, centro_fecha + token_h_px)
        scan_x1 = self._detectar_limite_derecho(tokens_scan, centro_fecha, scan_h, scan_x0, w, token_h_norm)

        fecha_crop = cheque_img[y0:y1, scan_x0:scan_x1]
        logger.info(
            "Fecha crop [y=%d:%d, x=%d:%d, token_h=%dpx]",
            y0, y1, scan_x0, scan_x1, token_h_px,
        )

        if fecha_crop.size == 0 or y1 <= y0:
            return []

        if debug_dir is not None:
            Image.fromarray(fecha_crop).save(debug_dir / _DEBUG_FECHA_ZONA)
        tokens = self._ocr.read(fecha_crop)
        logger.info("Fecha crop -> %d tokens", len(tokens))
        return tokens

    @staticmethod
    def _detectar_limite_derecho(
        tokens_scan: list[OCRResult],
        centro_fecha: int,
        scan_h: int,
        scan_x0: int,
        w: int,
        token_h_norm: float,
    ) -> int:
        """Devuelve el x absoluto donde comienza el identificador del cheque en la fila de fecha."""
        cy_fecha_norm = centro_fecha / scan_h
        fila = [t for t in tokens_scan if abs(t.cy - cy_fecha_norm) < token_h_norm]
        id_tokens = [t for t in fila if re.match(r'^\d{6,}$', t.text.strip()) and t.cx > 0.4]
        if id_tokens:
            leftmost = min(id_tokens, key=lambda t: t.cx)
            x1 = scan_x0 + int(leftmost.cx * (w - scan_x0))
            logger.info("Limite derecho: token %r en cx=%.2f -> x=%d", leftmost.text, leftmost.cx, x1)
            return x1
        return w

    def _fallback_fecha(
        self,
        tokens_scan: list[OCRResult],
        zona: np.ndarray,
        debug_dir: Path | None,
    ) -> list[OCRResult]:
        """Fallback: filtra tokens del scan amplio por mes/año conocido."""
        fecha_anclas = [t for t in tokens_scan if _es_token_fecha(t.text)]
        if fecha_anclas:
            cy_fila = sum(t.cy for t in fecha_anclas) / len(fecha_anclas)
            resultado = [t for t in tokens_scan if abs(t.cy - cy_fila) < 0.08]
            logger.info("Fallback por ancla de fecha -> %d tokens", len(resultado))
            if debug_dir is not None:
                Image.fromarray(zona).save(debug_dir / _DEBUG_FECHA_ZONA)
            return resultado

        logger.info("Sin anclas, devolviendo tokens del scan amplio (%d)", len(tokens_scan))
        if debug_dir is not None:
            Image.fromarray(zona).save(debug_dir / _DEBUG_FECHA_ZONA)
        return tokens_scan
