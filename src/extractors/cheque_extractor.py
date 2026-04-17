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
        self._fecha_ext = FechaEmisionExtractor(ocr_reader)
        self._llm = llm_validator

    def extraer(
        self,
        cheque_img: np.ndarray,
        batch_context: list[str] | None = None,
        debug_dir: Path | None = None,
    ) -> DatosCheque:
        """Extrae monto y fecha_emision de un cheque.

        Args:
            cheque_img: Imagen RGB del cheque recortado.
            batch_context: Montos raw de otros cheques del lote, para contexto LLM.

        Returns:
            DatosCheque con todos los campos extraidos.
        """
        # ---- OCR ----
        t0 = time.perf_counter()
        monto_result = self._monto_ext.extraer(cheque_img, debug_dir=debug_dir)
        fecha_tokens = self._fecha_ext.leer_tokens(cheque_img, debug_dir=debug_dir)
        ocr_elapsed = time.perf_counter() - t0
        logger.info(
            "OCR zonas (monto=%d, fecha=%d tokens): %.1fs",
            len(monto_result.zona_tokens), len(fecha_tokens), ocr_elapsed,
        )

        monto_final = monto_result.monto
        monto_raw_final = monto_result.monto_raw
        monto_llm_confidence = None
        fecha_emision = None
        fecha_emision_raw = None
        fecha_emision_llm_confidence = None

        # ---- LLM (si disponible) ----
        if self._llm is not None:
            all_tokens = monto_result.zona_tokens + fecha_tokens
            llm_results = self._llm.extract_fields(all_tokens, batch_context or [])

            llm_monto = llm_results.get("monto")
            llm_fecha = llm_results.get("fecha_emision")

            if llm_monto is not None:
                monto_llm_confidence = llm_monto.confidence
                if llm_monto.confidence >= 0.70 and llm_monto.normalized is not None:
                    monto_final = llm_monto.normalized
                    monto_raw_final = llm_monto.value or monto_result.monto_raw

            if llm_fecha is not None:
                fecha_emision_llm_confidence = llm_fecha.confidence
                logger.info(
                    "LLM fecha: value=%r normalized=%r conf=%.2f",
                    llm_fecha.value, llm_fecha.normalized, llm_fecha.confidence,
                )
                if llm_fecha.confidence >= 0.70:
                    fecha_emision = llm_fecha.normalized
                    fecha_emision_raw = llm_fecha.value

        return DatosCheque(
            monto=monto_final,
            monto_raw=monto_raw_final,
            monto_score=monto_result.monto_score,
            monto_llm_confidence=monto_llm_confidence,
            fecha_emision=fecha_emision,
            fecha_emision_raw=fecha_emision_raw,
            fecha_emision_llm_confidence=fecha_emision_llm_confidence,
        )
