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
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ..ocr.ocr_readers import OCRReader, OCRResult

if TYPE_CHECKING:
    from ..llm.llm_validator import LLMValidator

logger = logging.getLogger(__name__)

_MESES_NOMBRES = {
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
}


def _es_token_fecha(text: str) -> bool:
    t = text.strip().lower()
    if t in _MESES_NOMBRES:
        return True
    if re.match(r'^20\d{2}$', t):
        return True
    return False


# Acepta 'DE', 'de', 'DE.', '_DE_', etc.
_DE_RE = re.compile(r'^[^a-zA-Z0-9]*[Dd][Ee][^a-zA-Z0-9]*$')

# Acepta solo mayusculas 'DE', 'DE.', etc.
_DE_MAYUS_RE = re.compile(r'^[^a-zA-Z0-9]*DE[^a-zA-Z0-9]*$')

# Detecta 'EL' incluso fusionado con el dia siguiente (ELJ8, EL19DE, ELZo, EL.)
_EL_INICIO_RE = re.compile(r'^[Ee][Ll]')

# Tokens que identifican el boilerplate "La fecha de pago no puede exceder un plazo de 360 dias"
_BOILERPLATE_RE = re.compile(r'^(360|dias?)$', re.IGNORECASE)

_DEBUG_FECHA_ZONA = "fecha_zona.png"
_VENTANA_CY = 0.07
# Matches a token that is a city name ending with a comma, e.g. 'FEDERAL,' 'CUATIA,'
_CIUDAD_COMA_RE = re.compile(r'^[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ]+,$')

_MES_A_NUM = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
}


@dataclass
class FechaEmisionResult:
    fecha_iso: str | None
    tokens: list[OCRResult]


def _fecha_completa_a_iso(text: str) -> str | None:
    """Converts 'DD DE Mes DE YYYY' to ISO 'YYYY-MM-DD', or None if incomplete/invalid."""
    m = re.match(r'^(\d{1,2}) DE ([A-Za-z]+) DE (\d{4})$', text.strip())
    if not m:
        return None
    dia, mes, anio = m.group(1), m.group(2).lower(), m.group(3)
    num_mes = _MES_A_NUM.get(mes)
    if num_mes is None or not (1 <= int(dia) <= 31 and 2020 <= int(anio) <= 2030):
        return None
    return f"{anio}-{num_mes}-{int(dia):02d}"


def _limpiar_dia(text: str) -> str:
    digits = re.findall(r'\d+', text)
    for d in digits:
        num = int(d)
        if 1 <= num <= 31:
            return str(num).zfill(2)
    return text


def _limpiar_mes(text: str) -> str:
    text_lower = text.lower().strip()
    for mes in ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]:
        if mes in text_lower:
            return mes.capitalize()
    return text


def _limpiar_ano(text: str) -> str:
    years = re.findall(r'\d{4}', text)
    for year in years:
        num = int(year)
        if 2020 <= num <= 2030:
            return year
    return text


def _filtrar_tokens_fecha_estructura(tokens: list[OCRResult]) -> list[OCRResult]:
    """Extrae DIA/MES/ANNO de los tokens y los devuelve como un unico token combinado."""
    de_indices = [i for i, t in enumerate(tokens) if _DE_MAYUS_RE.match(t.text.strip())]

    if len(de_indices) < 2:
        logger.info("Estructura fecha: menos de 2 'DE' encontrados, retornando tokens originales")
        return tokens

    idx_de1 = de_indices[0]
    idx_de2 = de_indices[1]

    dia_token = None
    if idx_de1 > 0 and not _BOILERPLATE_RE.match(tokens[idx_de1 - 1].text.strip()):
        dia_token = tokens[idx_de1 - 1]

    mes_tokens = [
        tokens[i] for i in range(idx_de1 + 1, idx_de2)
        if not _BOILERPLATE_RE.match(tokens[i].text.strip())
    ]

    anio_token = None
    if idx_de2 + 1 < len(tokens) and not _BOILERPLATE_RE.match(tokens[idx_de2 + 1].text.strip()):
        anio_token = tokens[idx_de2 + 1]

    dia_raw = dia_token.text.strip() if dia_token else ""
    mes_raw = " ".join(t.text.strip() for t in mes_tokens) if mes_tokens else ""
    anio_raw = anio_token.text.strip() if anio_token else ""

    dia_text = _limpiar_dia(dia_raw)
    mes_text = _limpiar_mes(mes_raw)
    anio_text = _limpiar_ano(anio_raw)

    fecha_completa = f"{dia_text} DE {mes_text} DE {anio_text}"
    logger.info(
        "Estructura fecha: DIA=%r (limpio: %r), MES=%r (limpio: %r), ANNO=%r (limpio: %r) -> %r",
        dia_raw, dia_text, mes_raw, mes_text, anio_raw, anio_text, fecha_completa,
    )

    return [OCRResult(text=fecha_completa, confidence=1.0, cx=0.5, cy=0.5, height=0.1)]


class FechaEmisionExtractor:
    """Extrae la fecha de emision de un cheque via OCR sobre el scan amplio."""

    def __init__(
        self,
        ocr_reader: OCRReader,
        llm_validator: "LLMValidator | None" = None,
    ):
        self._ocr = ocr_reader
        self._llm = llm_validator

    def extraer(
        self,
        cheque_img: np.ndarray,
        debug_dir: Path | None = None,
    ) -> FechaEmisionResult:
        h, w = cheque_img.shape[:2]
        scan_h = int(h * 0.55)
        scan_x0 = int(w * 0.10)
        zona = cheque_img[0:scan_h, scan_x0:w]
        tokens_scan = self._ocr.read(zona)
        logger.info("Tokens scan: %s", [(t.text, round(t.cy, 3)) for t in tokens_scan])

        # 1. Ciudad-coma anchor
        ciudad_coma = self._encontrar_ciudad_coma(tokens_scan)
        if ciudad_coma is not None:
            cy_emision = ciudad_coma.cy
            logger.info("Ciudad-coma ancla: token=%r cy=%.3f", ciudad_coma.text, cy_emision)
            scan_window = [
                t for t in tokens_scan
                if abs(t.cy - cy_emision) < _VENTANA_CY and t.cx > ciudad_coma.cx
            ]
            logger.info("Scan window (cy=%.3f +-%.3f, cx>%.3f): %s", cy_emision, _VENTANA_CY, ciudad_coma.cx, [(t.text, round(t.cy, 3)) for t in scan_window])
            if debug_dir is not None:
                fecha_crop = self._get_debug_crop(cheque_img, tokens_scan, cy_emision, scan_h, scan_x0)
                Image.fromarray(fecha_crop).save(debug_dir / _DEBUG_FECHA_ZONA)
            if scan_window:
                result = self._result_desde_scan_window(scan_window)
                return result if result.fecha_iso else self._llm_fallback(result)

        # 2. DE-cluster / EL-anchor
        cy_emision = self._detectar_cy_emision(tokens_scan)
        if cy_emision is not None:
            scan_window = [t for t in tokens_scan if abs(t.cy - cy_emision) < _VENTANA_CY]
            logger.info("Scan window (cy=%.3f +-%.3f): %s", cy_emision, _VENTANA_CY, [(t.text, round(t.cy, 3)) for t in scan_window])
            if debug_dir is not None:
                fecha_crop = self._get_debug_crop(cheque_img, tokens_scan, cy_emision, scan_h, scan_x0)
                Image.fromarray(fecha_crop).save(debug_dir / _DEBUG_FECHA_ZONA)
            if scan_window:
                result = self._result_desde_scan_window(scan_window)
                return result if result.fecha_iso else self._llm_fallback(result)

        # 3. Fallback
        fallback_tokens = self._fallback_fecha(tokens_scan, zona, debug_dir)
        return self._llm_fallback(FechaEmisionResult(fecha_iso=None, tokens=fallback_tokens))

    @staticmethod
    def _get_debug_crop(
        cheque_img: np.ndarray,
        tokens_scan: list[OCRResult],
        cy_emision: float,
        scan_h: int,
        scan_x0: int,
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
        return cheque_img[y0:y1, scan_x0:]

    @staticmethod
    def _result_desde_scan_window(scan_window: list[OCRResult]) -> FechaEmisionResult:
        filtered = _filtrar_tokens_fecha_estructura(scan_window)
        if len(filtered) == 1:
            iso = _fecha_completa_a_iso(filtered[0].text)
            if iso is not None:
                logger.info("Fecha completa por OCR: %s", iso)
                return FechaEmisionResult(fecha_iso=iso, tokens=scan_window)
        logger.info("OCR incompleto, retornando tokens: %s", [t.text for t in scan_window])
        return FechaEmisionResult(fecha_iso=None, tokens=scan_window)

    def _llm_fallback(self, ocr_result: FechaEmisionResult) -> FechaEmisionResult:
        if self._llm is None:
            return ocr_result
        today_max = date.today().isoformat()
        llm_result = self._llm.infer_fecha(ocr_result.tokens, today_max)
        logger.info("LLM fecha: %r (conf=%.2f)", llm_result.normalized, llm_result.confidence)
        if llm_result.normalized is not None:
            return FechaEmisionResult(fecha_iso=llm_result.normalized, tokens=ocr_result.tokens)
        return ocr_result

    @staticmethod
    def _encontrar_ciudad_coma(tokens_scan: list[OCRResult]) -> OCRResult | None:
        token = next(
            (t for t in tokens_scan if _CIUDAD_COMA_RE.match(t.text.strip()) and t.cy > 0.30),
            None,
        )
        if token:
            logger.info("Ciudad-coma: encontrado %r en cy=%.3f", token.text, token.cy)
        else:
            logger.info("Ciudad-coma: no encontrado")
        return token

    def _detectar_cy_emision(self, tokens_scan: list[OCRResult]) -> float | None:
        el_token = self._encontrar_el(tokens_scan)
        el_cy = el_token.cy if el_token else None
        cy = self._cy_por_de_cluster(tokens_scan, el_cy)
        if cy is not None:
            return cy
        return self._cy_por_el_ancla(el_token)

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

    @staticmethod
    def _agrupar_de_clusters(de_tokens: list[OCRResult]) -> list[list[OCRResult]]:
        clusters: list[list[OCRResult]] = [[de_tokens[0]]]
        for tok in de_tokens[1:]:
            if tok.cy - clusters[-1][-1].cy < 0.06:
                clusters[-1].append(tok)
            else:
                clusters.append([tok])
        return clusters

    def _cy_por_de_cluster(self, tokens_scan: list[OCRResult], el_cy: float | None) -> float | None:
        de_tokens = sorted(
            [t for t in tokens_scan if _DE_RE.match(t.text.strip())],
            key=lambda t: t.cy,
        )
        if len(de_tokens) < 2:
            logger.info("DE-cluster: menos de 2 tokens 'DE' encontrados")
            return None

        validos = [c for c in self._agrupar_de_clusters(de_tokens) if len(c) >= 2]
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

    def _fallback_fecha(
        self,
        tokens_scan: list[OCRResult],
        zona: np.ndarray,
        debug_dir: Path | None,
    ) -> list[OCRResult]:
        """Fallback: filtra tokens del scan amplio por mes/anno conocido."""
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
