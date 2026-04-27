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

from .campos_librador_extractor import CamposLibradorExtractor
from .fecha_emision_extractor import FechaEmisionExtractor
from .fecha_pago_extractor import FechaPagoExtractor
from ..models import DatosCheque
from .fecha_extractor import Fecha
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
        self._librador_ext = CamposLibradorExtractor(ocr_reader)
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
        librador_result = self._librador_ext.extraer(cheque_img, debug_dir=debug_dir)
        logger.info(
            "OCR zonas (monto=%d tokens, fecha_iso=%r, fecha_pago_iso=%r, fecha_tokens=%d, cuit=%r, nombre=%r): %.1fs",
            len(monto_result.zona_tokens), fecha_result.fecha_iso,
            fecha_pago_result.fecha_iso, len(fecha_result.tokens),
            librador_result.cuit, librador_result.nombre,
            time.perf_counter() - t0,
        )

        monto = monto_result.monto
        monto_raw = monto_result.monto_raw
        fecha_emision = fecha_result.fecha_iso
        fecha_pago = fecha_pago_result.fecha_iso
        cuit_librador = librador_result.cuit
        nombre_librador = librador_result.nombre
        monto_llm_confidence = None
        fecha_llm_confidence = None
        fecha_pago_llm_confidence = None

        if self._llm is not None:
            today_max = date.today().isoformat()
            def _log_fecha(label: str, f: "Fecha | None") -> None:
                if f:
                    logger.info(
                        "%s: dia=%r(v=%r) mes=%r(v=%r) anno=%r(v=%r) iso=%r",
                        label,
                        f.dia_raw, f.dia, f.mes_raw, f.mes, f.anno_raw, f.anno,
                        f.to_iso(),
                    )
                else:
                    logger.info("%s: None", label)

            _log_fecha("fecha_emision parcial", fecha_result.partial)
            _log_fecha("fecha_pago parcial", fecha_pago_result.partial)

            with ThreadPoolExecutor(max_workers=3) as pool:
                fecha_future = (
                    pool.submit(
                        self._llm.infer_fecha,
                        fecha_result.tokens, today_max, None, fecha_result.partial,
                    )
                    if fecha_emision is None else None
                )
                fecha_pago_future = (
                    pool.submit(
                        self._llm.infer_fecha,
                        fecha_pago_result.tokens, None, 365, fecha_pago_result.partial,
                    )
                    if fecha_pago is None else None
                )
                monto_future = (
                    pool.submit(self._llm.extract_fields, monto_result.zona_tokens, [])
                    if monto is None else None
                )

            fecha_emision, fecha_llm_confidence = self._resolve_fecha(
                fecha_future, "fecha_emision", fecha_emision, fecha_llm_confidence,
                fecha_result.partial,
            )
            fecha_pago, fecha_pago_llm_confidence = self._resolve_fecha(
                fecha_pago_future, "fecha_pago", fecha_pago, fecha_pago_llm_confidence,
                fecha_pago_result.partial,
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
            cuit_librador=cuit_librador,
            nombre_librador=nombre_librador,
        )

    @staticmethod
    def _resolve_fecha(future, field_name, current_value, current_conf, fecha: "Fecha | None" = None):
        if future is None:
            return current_value, current_conf
        result: "LLMExtractionResult" = future.result()
        logger.info("LLM %s: %r conf=%.2f", field_name, result.normalized, result.confidence)
        if result.normalized is not None:
            normalized = ChequeExtractor._apply_fecha_overrides(result.normalized, fecha, field_name)
            return normalized, result.confidence
        return current_value, current_conf

    @staticmethod
    def _apply_fecha_overrides(iso: str, fecha: "Fecha | None", field_name: str) -> str:
        """Locks OCR-validated components into the LLM result to prevent temperature drift."""
        if fecha is None:
            return iso
        try:
            year, month, day = iso.split('-')
        except ValueError:
            return iso
        if fecha.anno:
            year = fecha.anno
        if fecha.mes:
            month = fecha.mes
        if fecha.dia:
            day = fecha.dia
        overridden = f"{year}-{month}-{day}"
        if overridden != iso:
            logger.info("LLM %s overridden by OCR: %r -> %r", field_name, iso, overridden)
        return overridden

    @staticmethod
    def _resolve_monto(future, current_monto, current_raw):
        if future is None:
            return current_monto, current_raw, None
        llm_monto = future.result().get("monto")
        if llm_monto and llm_monto.normalized is not None and llm_monto.confidence >= 0.70:
            logger.info("LLM monto: %r conf=%.2f", llm_monto.normalized, llm_monto.confidence)
            return llm_monto.normalized, llm_monto.value or current_raw, llm_monto.confidence
        return current_monto, current_raw, None
