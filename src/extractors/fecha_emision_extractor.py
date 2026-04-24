"""Extractor de zona de fecha de emision de cheques.

Estrategia hibrida:
  1. OCR amplio sobre la mitad superior del cheque.
  2. Detectar el cy de la linea de emision usando ciudad-coma, cluster DE, o ancla EL.
  3. Filtrar los tokens del scan amplio a esa banda y parsear DIA DE MES DE ANNO.
     Si el parseo produce una fecha completa y valida, devolver el ISO directamente.
  Fallback: filtrar del scan amplio por tokens de mes/anno conocido.
"""

import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..ocr.ocr_readers import OCRReader, OCRResult
from .fecha_extractor import (
    Fecha,
    FechaResult,
    _agrupar_de_clusters,
    _BOILERPLATE_RE,
    _DE_RE,
    _EL_INICIO_RE,
    _es_token_fecha,
    _fecha_completa_a_iso,
    _filtrar_tokens_fecha_estructura,
    _PLAZO_360_RE,
    _VENTANA_CY,
)

logger = logging.getLogger(__name__)

_DEBUG_FECHA_ZONA = "fecha_zona.png"
# Matches a token that is a city name ending with a comma, e.g. 'FEDERAL,' 'CUATIA,'
_CIUDAD_COMA_RE = re.compile(r'^[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ]+,$')


class FechaEmisionExtractor:
    """Extrae la fecha de emision de un cheque via OCR sobre el scan amplio."""

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
        scan_x1 = int(w * 0.80)
        zona = cheque_img[0:scan_h, scan_x0:scan_x1]
        tokens_scan = self._ocr.read(zona)
        logger.info("Tokens scan: %s", [(t.text, round(t.cy, 3)) for t in tokens_scan])

        # 1. Ciudad-coma anchor
        result = self._extraer_por_ciudad_coma(tokens_scan, cheque_img, scan_h, scan_x0, scan_x1, debug_dir)
        if result is not None:
            return result

        # 2. DE-cluster / EL-anchor
        result = self._extraer_por_de_el(tokens_scan, cheque_img, scan_h, scan_x0, scan_x1, debug_dir)
        if result is not None:
            return result

        # 3. Fallback
        fallback_tokens = self._fallback_zona(tokens_scan, zona, debug_dir)
        return FechaResult(fecha_iso=None, tokens=fallback_tokens)

    @staticmethod
    def _get_debug_crop(
        cheque_img: np.ndarray,
        tokens_scan: list[OCRResult],
        cy_emision: float,
        scan_h: int,
        scan_x0: int,
        scan_x1: int,
    ) -> np.ndarray:
        ancla = next(
            (t for t in tokens_scan if abs(t.cy - cy_emision) < _VENTANA_CY and t.height > 0),
            None,
        )
        token_h_norm = max(ancla.height if ancla else 0.0, 0.07)
        token_h_px = int(token_h_norm * scan_h)
        centro_px = int(cy_emision * scan_h)
        margen_px = int(token_h_px * 1.5)
        offset_px = int(token_h_px * 0.5)
        y0 = max(0, centro_px - margen_px - offset_px)
        y1 = min(scan_h, centro_px + margen_px - offset_px)
        return cheque_img[y0:y1, scan_x0:scan_x1]

    @staticmethod
    def _result_desde_scan_window(scan_window: list[OCRResult]) -> FechaResult:
        filtered, source_tokens, partial = _filtrar_tokens_fecha_estructura(scan_window)
        if len(filtered) == 1:
            iso = _fecha_completa_a_iso(filtered[0].text)
            if iso is not None:
                logger.info("Fecha completa por OCR: %s", iso)
                return FechaResult(fecha_iso=iso, tokens=scan_window)
            logger.info(
                "OCR incompleto, retornando tokens estructurales: %s",
                [t.text for t in source_tokens],
            )
            return FechaResult(fecha_iso=None, tokens=source_tokens, partial=partial)
        logger.info("OCR incompleto, retornando tokens: %s", [t.text for t in scan_window])
        return FechaResult(fecha_iso=None, tokens=scan_window)

    def _extraer_por_ciudad_coma(
        self,
        tokens_scan: list[OCRResult],
        cheque_img: np.ndarray,
        scan_h: int,
        scan_x0: int,
        scan_x1: int,
        debug_dir: Path | None,
    ) -> FechaResult | None:
        token = next(
            (t for t in tokens_scan if _CIUDAD_COMA_RE.match(t.text.strip()) and t.cy > 0.30),
            None,
        )
        if token is None:
            logger.info("Ciudad-coma: no encontrado")
            return None
        cy_emision = token.cy
        logger.info("Ciudad-coma ancla: token=%r cy=%.3f", token.text, cy_emision)
        scan_window = [
            t for t in tokens_scan
            if abs(t.cy - cy_emision) < _VENTANA_CY and t.cx > token.cx
        ]
        logger.info(
            "Scan window (cy=%.3f +-%.3f, cx>%.3f): %s",
            cy_emision, _VENTANA_CY, token.cx, [(t.text, round(t.cy, 3)) for t in scan_window],
        )
        if debug_dir is not None:
            fecha_crop = self._get_debug_crop(cheque_img, tokens_scan, cy_emision, scan_h, scan_x0, scan_x1)
            Image.fromarray(fecha_crop).save(debug_dir / _DEBUG_FECHA_ZONA)
        if scan_window:
            return self._result_desde_scan_window(scan_window)
        return None

    def _extraer_por_de_el(
        self,
        tokens_scan: list[OCRResult],
        cheque_img: np.ndarray,
        scan_h: int,
        scan_x0: int,
        scan_x1: int,
        debug_dir: Path | None,
    ) -> FechaResult | None:
        el_token = self._encontrar_el(tokens_scan)
        el_cy = el_token.cy if el_token else None
        cy_emision = self._cy_por_de_cluster(tokens_scan, el_cy)
        if cy_emision is None:
            cy_emision = self._cy_por_el_ancla(el_token)
        if cy_emision is None:
            return None
        scan_window = [t for t in tokens_scan if abs(t.cy - cy_emision) < _VENTANA_CY]
        logger.info(
            "Scan window (cy=%.3f +-%.3f): %s",
            cy_emision, _VENTANA_CY, [(t.text, round(t.cy, 3)) for t in scan_window],
        )
        if debug_dir is not None:
            fecha_crop = self._get_debug_crop(cheque_img, tokens_scan, cy_emision, scan_h, scan_x0, scan_x1)
            Image.fromarray(fecha_crop).save(debug_dir / _DEBUG_FECHA_ZONA)
        if scan_window:
            return self._result_desde_scan_window(scan_window)
        return None

    @staticmethod
    def _encontrar_el(tokens_scan: list[OCRResult]) -> OCRResult | None:
        token = next(
            (t for t in tokens_scan if _EL_INICIO_RE.match(t.text.strip()) and t.cy > 0.35),
            None,
        )
        logger.info(
            "EL-linea: %s",
            f"cy={token.cy:.3f} (token={token.text!r})" if token else "no detectada",
        )
        return token


    def _cy_por_de_cluster(self, tokens_scan: list[OCRResult], el_cy: float | None) -> float | None:
        de_tokens = sorted(
            [t for t in tokens_scan if _DE_RE.match(t.text.strip())],
            key=lambda t: t.cy,
        )
        if len(de_tokens) < 2:
            logger.info("DE-cluster: menos de 2 tokens 'DE' encontrados")
            return None

        validos = [c for c in _agrupar_de_clusters(de_tokens) if len(c) >= 2]
        for cluster in sorted(validos, key=lambda c: c[0].cy):
            cy_centro = sum(t.cy for t in cluster) / len(cluster)
            vecinos = [t for t in tokens_scan if abs(t.cy - cy_centro) < _VENTANA_CY]
            if any(_BOILERPLATE_RE.match(t.text.strip()) for t in vecinos):
                logger.info("DE-cluster: cy=%.3f descartado (boilerplate)", cy_centro)
                continue
            if el_cy is not None and cy_centro >= el_cy - 0.02:
                logger.info("DE-cluster: cy=%.3f descartado (linea de pago)", cy_centro)
                continue
            logger.info(
                "DE-cluster: usando cy=%.3f %s",
                cy_centro, [(t.text, round(t.cy, 3)) for t in cluster],
            )
            return cy_centro

        logger.info("DE-cluster: todos los clusters descartados")
        return None

    @staticmethod
    def _cy_por_el_ancla(el_token: OCRResult | None) -> float | None:
        if el_token is None:
            return None
        token_h = max(el_token.height, 0.07)
        cy_estimado = el_token.cy - token_h * 1.5
        logger.info("EL-ancla: estimando cy_emision=%.3f (el_cy=%.3f - 1.5*h)", cy_estimado, el_token.cy)
        return cy_estimado

    def _fallback_zona(
        self,
        tokens_scan: list[OCRResult],
        zona: np.ndarray,
        debug_dir: Path | None,
    ) -> list[OCRResult]:
        """Fallback general: acota la zona usando el boilerplate '360 dias' como
        limite inferior y el tope del 40 % de la altura del cheque como techo."""
        # Top 40 % del cheque en coordenadas relativas al scan (scan_h = 55 % del cheque)
        cy_techo = 0.40 / 0.55  # ~0.727

        boilerplate_tokens = [t for t in tokens_scan if _PLAZO_360_RE.match(t.text.strip())]
        if boilerplate_tokens:
            cy_boilerplate = min(t.cy for t in boilerplate_tokens)
            cy_techo = min(cy_techo, cy_boilerplate)
            logger.info("Fallback: limite inferior por 'plazo de 360' cy=%.3f", cy_boilerplate)
        else:
            logger.info("Fallback: sin boilerplate, usando techo cy=%.3f", cy_techo)

        tokens_zona = [t for t in tokens_scan if t.cy < cy_techo]
        logger.info("Fallback: %d tokens en zona (cy < %.3f)", len(tokens_zona), cy_techo)

        fecha_anclas = [t for t in tokens_zona if _es_token_fecha(t.text)]
        if fecha_anclas:
            cy_fila = sum(t.cy for t in fecha_anclas) / len(fecha_anclas)
            resultado = [t for t in tokens_zona if abs(t.cy - cy_fila) < 0.08]
            logger.info("Fallback por ancla de fecha -> %d tokens", len(resultado))
            if debug_dir is not None:
                Image.fromarray(zona).save(debug_dir / _DEBUG_FECHA_ZONA)
            return resultado

        logger.info("Sin anclas, devolviendo tokens de zona superior (%d)", len(tokens_zona))
        if debug_dir is not None:
            Image.fromarray(zona).save(debug_dir / _DEBUG_FECHA_ZONA)
        return tokens_zona
