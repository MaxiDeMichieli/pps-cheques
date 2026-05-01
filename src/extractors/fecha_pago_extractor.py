"""Extractor de fecha de pago de cheques.

Estrategia hibrida (en orden de prioridad):
  1. Ancla EL: el token 'EL' está en la misma línea que la fecha de pago.
     Se toma el EL más bajo (mayor cy) para evitar falsos positivos de la
     línea de emision.
  2. Ancla PAGUESE: si no hay EL, se busca 'Páguese/PAGUESE'. La fecha de
     pago está una línea ARRIBA (menor cy) de ese token, estimada a 1.5×
     la altura del token.
  3. Cluster DE inferior: se busca el cluster de tokens 'DE' más bajo
     (mayor cy) con al menos 2 tokens, descartando boilerplate. Es el
     complemento del cluster que FechaEmisionExtractor descarta.
  4. Fallback por keywords: agrupa tokens de mes/año, toma la banda más baja.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..ocr.ocr_readers import OCRReader, OCRResult
from .fecha_extractor import (
    Fecha,
    FechaResult,
    _agrupar_de_clusters,
    _es_token_fecha,
    _fecha_completa_a_iso,
    _filtrar_tokens_fecha_estructura,
    _BOILERPLATE_RE,
    _DE_RE,
    _EL_INICIO_RE,
    _VENTANA_CY,
)

logger = logging.getLogger(__name__)

_PAGUESE_RE = re.compile(r'^p[aá]guese', re.IGNORECASE)
_DEBUG_FECHA_PAGO_ZONA = "fecha_pago_zona.png"


class FechaPagoExtractor:
    """Extrae la fecha de pago de un cheque via OCR sobre el scan amplio."""

    def __init__(self, ocr_reader: OCRReader):
        self._ocr = ocr_reader

    def extraer(
        self,
        cheque_img: np.ndarray,
        debug_dir: Path | None = None,
    ) -> FechaResult:
        h, w = cheque_img.shape[:2]
        scan_h = int(h * 0.55)
        scan_x0 = int(w * 0.10)
        scan_x1 = int(w * 0.70)
        zona = cheque_img[0:scan_h, scan_x0:scan_x1]
        tokens_scan = self._ocr.read(zona)
        logger.info("FechaPago tokens scan: %s", [(t.text, round(t.cy, 3)) for t in tokens_scan])

        # 1. EL anchor — same line as fecha_pago
        result = self._extraer_por_el(tokens_scan, cheque_img, scan_h, scan_x0, scan_x1, debug_dir)
        if result is not None:
            logger.info("FechaPago anchor=EL iso=%r", result.fecha_iso)
            return result

        # 2. PAGUESE anchor — fecha_pago is one line above it
        result = self._extraer_por_paguese(tokens_scan, cheque_img, scan_h, scan_x0, scan_x1, debug_dir)
        if result is not None:
            logger.info("FechaPago anchor=PAGUESE iso=%r", result.fecha_iso)
            return result

        # 3. Lower DE cluster
        result = self._extraer_por_de_cluster_inferior(tokens_scan, cheque_img, scan_h, scan_x0, scan_x1, debug_dir)
        if result is not None:
            logger.info("FechaPago anchor=DE-cluster iso=%r", result.fecha_iso)
            return result

        # 4. Fallback: lowest date-keyword band
        fallback_tokens = self._fallback_banda_inferior(tokens_scan, zona, debug_dir)
        logger.info("FechaPago anchor=fallback-keyword iso=None tokens=%d", len(fallback_tokens))
        return FechaResult(fecha_iso=None, tokens=fallback_tokens)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_debug_crop(
        cheque_img: np.ndarray,
        tokens_scan: list[OCRResult],
        cy_pago: float,
        scan_h: int,
        scan_x0: int,
        scan_x1: int,
    ) -> np.ndarray:
        ancla = next(
            (t for t in tokens_scan if abs(t.cy - cy_pago) < _VENTANA_CY and t.height > 0),
            None,
        )
        token_h_norm = max(ancla.height if ancla else 0.0, 0.07)
        token_h_px = int(token_h_norm * scan_h)
        centro_px = int(cy_pago * scan_h)
        margen_px = int(token_h_px * 1.5)
        y0 = max(0, centro_px - margen_px)
        y1 = min(scan_h, centro_px + margen_px)
        return cheque_img[y0:y1, scan_x0:scan_x1]

    @staticmethod
    def _result_desde_scan_window(scan_window: list[OCRResult]) -> FechaResult:
        filtered, source_tokens, partial = _filtrar_tokens_fecha_estructura(scan_window, skip_el_prefix=True)
        if len(filtered) == 1:
            iso = _fecha_completa_a_iso(filtered[0].text)
            if iso is not None:
                logger.info("FechaPago completa por OCR: %s", iso)
                return FechaResult(fecha_iso=iso, tokens=scan_window)
            logger.info(
                "FechaPago OCR incompleto, tokens estructurales: %s",
                [t.text for t in source_tokens],
            )
            return FechaResult(fecha_iso=None, tokens=source_tokens, partial=partial)
        logger.info("FechaPago OCR incompleto, tokens: %s", [t.text for t in scan_window])
        return FechaResult(fecha_iso=None, tokens=scan_window)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _extraer_por_el(
        self,
        tokens_scan: list[OCRResult],
        cheque_img: np.ndarray,
        scan_h: int,
        scan_x0: int,
        scan_x1: int,
        debug_dir: Path | None,
    ) -> FechaResult | None:
        candidates = [t for t in tokens_scan if _EL_INICIO_RE.match(t.text.strip()) and t.cy > 0.35]
        if not candidates:
            logger.info("FechaPago EL-anchor: no encontrado")
            return None
        el_token = max(candidates, key=lambda t: t.cy)
        logger.info("FechaPago EL-anchor: token=%r cy=%.3f", el_token.text, el_token.cy)
        scan_window = [t for t in tokens_scan if abs(t.cy - el_token.cy) < _VENTANA_CY]
        logger.info(
            "FechaPago scan window EL (cy=%.3f +-%.3f): %s",
            el_token.cy, _VENTANA_CY, [(t.text, round(t.cy, 3)) for t in scan_window],
        )
        if debug_dir is not None:
            crop = self._get_debug_crop(cheque_img, tokens_scan, el_token.cy, scan_h, scan_x0, scan_x1)
            Image.fromarray(crop).save(debug_dir / _DEBUG_FECHA_PAGO_ZONA)
        if scan_window:
            return self._result_desde_scan_window(scan_window)
        return None

    def _extraer_por_paguese(
        self,
        tokens_scan: list[OCRResult],
        cheque_img: np.ndarray,
        scan_h: int,
        scan_x0: int,
        scan_x1: int,
        debug_dir: Path | None,
    ) -> FechaResult | None:
        paguese_token = next(
            (t for t in tokens_scan if _PAGUESE_RE.match(t.text.strip()) and t.cy > 0.35),
            None,
        )
        if paguese_token is None:
            logger.info("FechaPago PAGUESE-anchor: no encontrado")
            return None
        token_h = max(paguese_token.height, 0.07)
        cy_pago = paguese_token.cy - token_h * 1.5
        logger.info(
            "FechaPago PAGUESE-anchor: token=%r cy=%.3f -> cy_pago estimado=%.3f",
            paguese_token.text, paguese_token.cy, cy_pago,
        )
        scan_window = [t for t in tokens_scan if abs(t.cy - cy_pago) < _VENTANA_CY]
        logger.info(
            "FechaPago scan window PAGUESE (cy=%.3f +-%.3f): %s",
            cy_pago, _VENTANA_CY, [(t.text, round(t.cy, 3)) for t in scan_window],
        )
        if debug_dir is not None:
            crop = self._get_debug_crop(cheque_img, tokens_scan, cy_pago, scan_h, scan_x0, scan_x1)
            Image.fromarray(crop).save(debug_dir / _DEBUG_FECHA_PAGO_ZONA)
        if scan_window:
            return self._result_desde_scan_window(scan_window)
        return None

    def _extraer_por_de_cluster_inferior(
        self,
        tokens_scan: list[OCRResult],
        cheque_img: np.ndarray,
        scan_h: int,
        scan_x0: int,
        scan_x1: int,
        debug_dir: Path | None,
    ) -> FechaResult | None:
        de_tokens = sorted(
            [t for t in tokens_scan if _DE_RE.match(t.text.strip())],
            key=lambda t: t.cy,
        )
        if len(de_tokens) < 2:
            logger.info("FechaPago DE-cluster: menos de 2 tokens 'DE'")
            return None

        validos = []
        for cluster in _agrupar_de_clusters(de_tokens):
            if len(cluster) < 2:
                continue
            cy_centro = sum(t.cy for t in cluster) / len(cluster)
            vecinos = [t for t in tokens_scan if abs(t.cy - cy_centro) < _VENTANA_CY]
            if any(_BOILERPLATE_RE.match(t.text.strip()) for t in vecinos):
                logger.info("FechaPago DE-cluster: cy=%.3f descartado (boilerplate)", cy_centro)
                continue
            validos.append((cy_centro, cluster))

        if not validos:
            logger.info("FechaPago DE-cluster: ningún cluster válido")
            return None

        # Take the lowest cluster (highest cy = fecha_pago side)
        cy_centro, cluster = max(validos, key=lambda x: x[0])
        logger.info(
            "FechaPago DE-cluster inferior: cy=%.3f %s",
            cy_centro, [(t.text, round(t.cy, 3)) for t in cluster],
        )
        scan_window = [t for t in tokens_scan if abs(t.cy - cy_centro) < _VENTANA_CY]
        if debug_dir is not None:
            crop = self._get_debug_crop(cheque_img, tokens_scan, cy_centro, scan_h, scan_x0, scan_x1)
            Image.fromarray(crop).save(debug_dir / _DEBUG_FECHA_PAGO_ZONA)
        if scan_window:
            return self._result_desde_scan_window(scan_window)
        return None

    def _fallback_banda_inferior(
        self,
        tokens_scan: list[OCRResult],
        zona: np.ndarray,
        debug_dir: Path | None,
    ) -> list[OCRResult]:
        """Busca tokens de mes/año, agrupa en bandas y toma la más baja."""
        fecha_anclas = [t for t in tokens_scan if _es_token_fecha(t.text)]
        if not fecha_anclas:
            logger.info("FechaPago fallback: sin anclas de fecha")
            if debug_dir is not None:
                Image.fromarray(zona).save(debug_dir / _DEBUG_FECHA_PAGO_ZONA)
            return []

        # Group anclas into cy bands
        fecha_anclas_sorted = sorted(fecha_anclas, key=lambda t: t.cy)
        bands: list[list[OCRResult]] = [[fecha_anclas_sorted[0]]]
        for tok in fecha_anclas_sorted[1:]:
            if tok.cy - bands[-1][-1].cy < 0.06:
                bands[-1].append(tok)
            else:
                bands.append([tok])

        # Take the lowest band (fecha_pago is below fecha_emision)
        lower_band = bands[-1]
        cy_fila = sum(t.cy for t in lower_band) / len(lower_band)
        resultado = [t for t in tokens_scan if abs(t.cy - cy_fila) < 0.08]
        logger.info("FechaPago fallback banda inferior -> %d tokens", len(resultado))
        if debug_dir is not None:
            Image.fromarray(zona).save(debug_dir / _DEBUG_FECHA_PAGO_ZONA)
        return resultado
