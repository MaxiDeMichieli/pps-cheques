"""Extractor de zona de fecha de emision de cheques.

Lee los tokens OCR de la franja donde aparece la linea de ciudad + fecha,
usando el token "EL" como ancla inferior (siempre esta en la linea siguiente
al "CIUDAD, DD DE MES DE AAAA" de emision).
"""

import logging

import numpy as np

from ..ocr.ocr_readers import OCRReader, OCRResult

logger = logging.getLogger(__name__)


class FechaEmisionExtractor:
    """Lee los tokens OCR de la zona de fecha de emision."""

    def __init__(self, ocr_reader: OCRReader):
        self._ocr = ocr_reader

    def leer_tokens(self, cheque_img: np.ndarray) -> list[OCRResult]:
        """Devuelve tokens OCR de la zona de fecha.

        Escanea la franja superior-central (donde aparece la linea
        "CIUDAD, DD DE MES DE AAAA") y usa el token "EL" como limite inferior,
        ya que la linea de fecha siempre esta encima del "EL DD DE MES" de pago.

        Args:
            cheque_img: Imagen RGB del cheque recortado.

        Returns:
            Lista de OCRResult de la zona de fecha, filtrada por posicion.
        """
        h, w = cheque_img.shape[:2]
        # Franja superior, dejando el logo del banco (izq) y el recuadro del monto (der)
        zona = cheque_img[0:int(h * 0.45), int(w * 0.15):int(w * 0.80)]
        tokens = self._ocr.read(zona)

        # Buscar "EL" como ancla: la fecha de emision esta en la fila anterior
        el_token = next(
            (t for t in tokens if t.text.strip().upper() == "EL"),
            None,
        )

        if el_token:
            # Banda entre el 30% inferior del header y la fila del EL:
            # - limite superior (0.30): descarta encabezado del cheque (nro. serie, tipo)
            # - limite inferior (el_cy - 0.05): descarta el EL y lo que sigue
            cy_min = el_token.cy * 0.30
            cy_max = el_token.cy - 0.05
            tokens = [t for t in tokens if cy_min < t.cy < cy_max]
            logger.info(
                "Ancla 'EL' en cy=%.2f -> %d tokens de fecha",
                el_token.cy, len(tokens),
            )
        else:
            logger.info(
                "Ancla 'EL' no encontrada, usando zona completa (%d tokens)",
                len(tokens),
            )

        return tokens
