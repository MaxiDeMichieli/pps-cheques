"""Orquestador de extraccion de campos de cheques.

Coordina MontoExtractor, FechaEmisionExtractor y LLMValidator para producir
un DatosCheque completo a partir de la imagen de un cheque.
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .fecha_emision_extractor import FechaEmisionExtractor
from ..models import DatosCheque
from .monto_extractor import MontoExtractor
from ..ocr.ocr_readers import OCRReader

if TYPE_CHECKING:
    from ..llm.llm_validator import LLMValidator

logger = logging.getLogger(__name__)


class ChequeExtractor:
    """Extrae todos los campos de un cheque combinando OCR y validacion LLM."""

    def __init__(
        self,
        ocr_reader: OCRReader,
        llm_validator: "LLMValidator | None" = None,
    ):
        self._monto_ext = MontoExtractor(ocr_reader)
        self._fecha_ext = FechaEmisionExtractor(ocr_reader, llm_validator=llm_validator)

    def extraer(
        self,
        cheque_img: np.ndarray,
        debug_dir: Path | None = None,
    ) -> DatosCheque:
        """Extrae monto y fecha_emision de un cheque.

        Args:
            cheque_img: Imagen RGB del cheque recortado.

        Returns:
            DatosCheque con todos los campos extraidos.
        """
        # ---- OCR ----
        t0 = time.perf_counter()
        monto_result = self._monto_ext.extraer(cheque_img, debug_dir=debug_dir)
        fecha_result = self._fecha_ext.extraer(cheque_img, debug_dir=debug_dir)
        ocr_elapsed = time.perf_counter() - t0
        logger.info(
            "OCR zonas (monto=%d tokens, fecha_iso=%r, fecha_tokens=%d): %.1fs",
            len(monto_result.zona_tokens), fecha_result.fecha_iso, len(fecha_result.tokens), ocr_elapsed,
        )

        return DatosCheque(
            monto=monto_result.monto,
            monto_raw=monto_result.monto_raw,
            monto_score=monto_result.monto_score,
            monto_llm_confidence=None,
            fecha_emision=fecha_result.fecha_iso,
            fecha_emision_raw=fecha_result.fecha_iso,
            fecha_emision_llm_confidence=None,
        )
