"""Extractor de zona de fecha de emision de cheques.

Lee los tokens OCR de la franja donde aparece la linea de ciudad + fecha,
usando el token "EL" como ancla inferior (siempre esta en la linea siguiente
al "CIUDAD, DD DE MES DE AAAA" de emision).
"""

import logging
import re

import numpy as np

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


class FechaEmisionExtractor:
    """Lee los tokens OCR de la zona de fecha de emision."""

    # Tolerancia vertical para agrupar tokens en la misma fila (en cy normalizado)
    _FILA_TOLERANCIA = 0.06

    def __init__(self, ocr_reader: OCRReader):
        self._ocr = ocr_reader

    def leer_tokens(self, cheque_img: np.ndarray) -> list[OCRResult]:
        """Devuelve tokens OCR de la zona de fecha.

        Escanea la franja superior-central (donde aparece la linea
        "CIUDAD, DD DE MES DE AAAA") y usa el token "EL" como limite inferior.
        El limite derecho se recorta antes del recuadro de monto/numero de cheque.
        Solo devuelve los tokens de la fila exacta donde se detecta la fecha.

        Args:
            cheque_img: Imagen RGB del cheque recortado.

        Returns:
            Lista de OCRResult de la linea de fecha unicamente.
        """
        h, w = cheque_img.shape[:2]
        # Excluye el logo del banco (izq), el header impreso (arriba),
        # y el recuadro de monto/nro de cheque (der, cortamos en 0.70)
        zona = cheque_img[0:int(h * 0.45), int(w * 0.15):int(w * 0.70)]
        tokens = self._ocr.read(zona)

        # Buscar "EL" como ancla: la fecha de emision esta en la fila anterior
        el_token = next(
            (t for t in tokens if t.text.strip().upper() == "EL"),
            None,
        )
        cy_max = el_token.cy - 0.05 if el_token else 1.0

        tokens_sobre_el = [t for t in tokens if t.cy < cy_max]

        # Buscar tokens que pertenezcan a la linea de fecha (mes o año)
        fecha_anclas = [t for t in tokens_sobre_el if _es_token_fecha(t.text)]

        if fecha_anclas:
            cy_fila = sum(t.cy for t in fecha_anclas) / len(fecha_anclas)
            resultado = [
                t for t in tokens_sobre_el
                if abs(t.cy - cy_fila) < self._FILA_TOLERANCIA
            ]
            logger.info(
                "Fila de fecha detectada en cy=%.2f -> %d tokens",
                cy_fila, len(resultado),
            )
            return resultado

        # Fallback: banda estrecha justo encima del EL
        if el_token:
            resultado = [
                t for t in tokens_sobre_el
                if t.cy > cy_max - 0.20
            ]
            logger.info(
                "Ancla de fecha no encontrada, fallback banda sobre EL -> %d tokens",
                len(resultado),
            )
            return resultado

        logger.info(
            "Sin anclas, devolviendo zona completa (%d tokens)", len(tokens_sobre_el)
        )
        return tokens_sobre_el
