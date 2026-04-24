"""Orquestador de extraccion de campos de cheques.

Coordina MontoExtractor, FechaEmisionExtractor y LLMValidator para producir
un DatosCheque completo a partir de la imagen de un cheque.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
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

        monto = monto_result.monto
        monto_raw = monto_result.monto_raw
        fecha_emision = fecha_result.fecha_iso
        monto_llm_confidence = None
        fecha_llm_confidence = None

        if self._llm is not None:
            today_max = date.today().isoformat()
            fecha_future = None
            monto_future = None
            with ThreadPoolExecutor(max_workers=2) as pool:
                if fecha_emision is None:
                    fecha_future = pool.submit(self._llm.infer_fecha, fecha_result.tokens, today_max)
                if monto is None:
                    monto_future = pool.submit(self._llm.extract_fields, monto_result.zona_tokens, [])
            if fecha_future is not None:
                llm_fecha = fecha_future.result()
                logger.info("LLM fecha: %r conf=%.2f", llm_fecha.normalized, llm_fecha.confidence)
                if llm_fecha.normalized is not None:
                    fecha_emision = llm_fecha.normalized
                    fecha_llm_confidence = llm_fecha.confidence
            if monto_future is not None:
                llm_monto = monto_future.result().get("monto")
                if llm_monto and llm_monto.normalized is not None and llm_monto.confidence >= 0.70:
                    logger.info("LLM monto: %r conf=%.2f", llm_monto.normalized, llm_monto.confidence)
                    monto = llm_monto.normalized
                    monto_raw = llm_monto.value or monto_raw
                    monto_llm_confidence = llm_monto.confidence

        return DatosCheque(
            monto=monto,
            monto_raw=monto_raw,
            monto_score=monto_result.monto_score,
            monto_llm_confidence=monto_llm_confidence,
            fecha_emision=fecha_emision,
            fecha_emision_raw=fecha_emision,
            fecha_emision_llm_confidence=fecha_llm_confidence,
        )
