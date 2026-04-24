"""Orquestador de extraccion de campos de cheques.

Coordina MontoExtractor, FechaEmisionExtractor, FechaPagoExtractor y LLMValidator
para producir un DatosCheque completo a partir de la imagen de un cheque.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .fecha_emision_extractor import FechaEmisionExtractor
from .fecha_pago_extractor import FechaPagoExtractor
from ..models import DatosCheque
from .monto_extractor import MontoExtractor
from ..ocr.ocr_readers import OCRReader

if TYPE_CHECKING:
    from ..llm.llm_validator import LLMValidator, LLMExtractionResult

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
        self._fecha_pago_ext = FechaPagoExtractor(ocr_reader)
        self._llm = llm_validator

    def extraer(
        self,
        cheque_img: np.ndarray,
        debug_dir: Path | None = None,
    ) -> DatosCheque:
        """Extrae todos los campos de un cheque.

        Args:
            cheque_img: Imagen RGB del cheque recortado.

        Returns:
            DatosCheque con todos los campos extraidos.
        """
        t0 = time.perf_counter()
        monto_result = self._monto_ext.extraer(cheque_img, debug_dir=debug_dir)
        fecha_result = self._fecha_ext.extraer(cheque_img, debug_dir=debug_dir)
        fecha_pago_result = self._fecha_pago_ext.extraer(cheque_img, debug_dir=debug_dir)
        logger.info(
            "OCR zonas (monto=%d tokens, fecha_iso=%r, fecha_pago_iso=%r, fecha_tokens=%d): %.1fs",
            len(monto_result.zona_tokens), fecha_result.fecha_iso,
            fecha_pago_result.fecha_iso, len(fecha_result.tokens),
            time.perf_counter() - t0,
        )

        monto = monto_result.monto
        monto_raw = monto_result.monto_raw
        fecha_emision = fecha_result.fecha_iso
        fecha_pago = fecha_pago_result.fecha_iso
        monto_llm_confidence = None
        fecha_llm_confidence = None
        fecha_pago_llm_confidence = None

        if self._llm is not None:
            today_max = date.today().isoformat()
            with ThreadPoolExecutor(max_workers=3) as pool:
                fecha_future = (
                    pool.submit(self._llm.infer_fecha, fecha_result.tokens, today_max)
                    if fecha_emision is None else None
                )
                fecha_pago_future = (
                    pool.submit(self._llm.infer_fecha, fecha_pago_result.tokens, None, 365)
                    if fecha_pago is None else None
                )
                monto_future = (
                    pool.submit(self._llm.extract_fields, monto_result.zona_tokens, [])
                    if monto is None else None
                )

            fecha_emision, fecha_llm_confidence = self._resolve_fecha(
                fecha_future, "fecha_emision", fecha_emision, fecha_llm_confidence
            )
            fecha_pago, fecha_pago_llm_confidence = self._resolve_fecha(
                fecha_pago_future, "fecha_pago", fecha_pago, fecha_pago_llm_confidence
            )
            monto, monto_raw, monto_llm_confidence = self._resolve_monto(
                monto_future, monto, monto_raw
            )

        return DatosCheque(
            monto=monto,
            monto_raw=monto_raw,
            monto_score=monto_result.monto_score,
            monto_llm_confidence=monto_llm_confidence,
            fecha_emision=fecha_emision,
            fecha_emision_raw=fecha_emision,
            fecha_emision_llm_confidence=fecha_llm_confidence,
            fecha_pago=fecha_pago,
            fecha_pago_raw=fecha_pago,
            fecha_pago_llm_confidence=fecha_pago_llm_confidence,
        )

    @staticmethod
    def _resolve_fecha(future, field_name, current_value, current_conf):
        if future is None:
            return current_value, current_conf
        result: "LLMExtractionResult" = future.result()
        logger.info("LLM %s: %r conf=%.2f", field_name, result.normalized, result.confidence)
        if result.normalized is not None:
            return result.normalized, result.confidence
        return current_value, current_conf

    @staticmethod
    def _resolve_monto(future, current_monto, current_raw):
        if future is None:
            return current_monto, current_raw, None
        llm_monto = future.result().get("monto")
        if llm_monto and llm_monto.normalized is not None and llm_monto.confidence >= 0.70:
            logger.info("LLM monto: %r conf=%.2f", llm_monto.normalized, llm_monto.confidence)
            return llm_monto.normalized, llm_monto.value or current_raw, llm_monto.confidence
        return current_monto, current_raw, None
