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
from pathlib import Path

import numpy as np
from PIL import Image

from ..ocr.ocr_readers import OCRReader, OCRResult

logger = logging.getLogger(__name__)

_MESES_NOMBRES = {
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
}


def _es_token_fecha(text: str) -> bool:
    t = text.strip().lower()
    if t in _MESES_NOMBRES:
        return True
    if _ANNO_RE.match(t):
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
# Ancla fiable para el limite inferior del fallback: 'plazo' de la leyenda 'plazo de 360 dias'
_PLAZO_360_RE = re.compile(r'^plazo$', re.IGNORECASE)

_ANNO_RE = re.compile(r'^20\d{2}$')

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


_DE_EMBEDDED_SPLIT_RE = re.compile(r'(DE)')


def _expandir_tokens_de(tokens: list[OCRResult]) -> list[OCRResult]:
    """Split tokens with embedded uppercase 'DE' into sub-tokens.

    Only splits when DE is adjacent to a digit or sits at the start of the token
    (e.g. '16DE' → ['16','DE'], 'DEZOZS' → ['DE','ZOZS'], 'DEZ025' → ['DE','Z025']).
    Pure-word tokens like 'FEDERAL' are left untouched because no digit is adjacent.
    """
    result: list[OCRResult] = []
    for t in tokens:
        text = t.text.strip()
        if _DE_MAYUS_RE.match(text):
            result.append(t)
            continue
        has_embedded = bool(re.search(r'\dDE|DE\d', text) or re.match(r'^DE.', text))
        if not has_embedded:
            result.append(t)
            continue
        partes = [p for p in _DE_EMBEDDED_SPLIT_RE.split(text) if p]
        for p in partes:
            result.append(OCRResult(text=p, confidence=t.confidence, cx=t.cx, cy=t.cy, height=t.height))
    return result


def _filtrar_tokens_fecha_estructura(
    tokens: list[OCRResult],
) -> tuple[list[OCRResult], list[OCRResult]]:
    """Extrae DIA/MES/ANNO de los tokens.

    Returns:
        (combined, source_tokens): combined is a single-element list with the assembled
        date string; source_tokens contains only the tokens that formed the structure
        (noise stripped). When fewer than 2 DE are found, returns (tokens, tokens).
    """
    tokens = _expandir_tokens_de(tokens)
    de_indices = [i for i, t in enumerate(tokens) if _DE_MAYUS_RE.match(t.text.strip())]

    if len(de_indices) < 2:
        logger.info("Estructura fecha: menos de 2 'DE' encontrados, retornando tokens originales")
        return tokens, tokens

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

    # Fallback: year may appear before the DE...DE structure due to OCR ordering or
    # because splitting an embedded DE (e.g. DEENERO) shifted the year out of position.
    if not _ANNO_RE.match(anio_text):
        assigned_ids = (
            {id(dia_token)} if dia_token else set()
        ) | {id(tokens[idx_de1]), id(tokens[idx_de2])} | {id(t) for t in mes_tokens}
        for t in tokens:
            if id(t) in assigned_ids:
                continue
            y = _limpiar_ano(t.text.strip())
            if _ANNO_RE.match(y):
                anio_token = t
                anio_raw = t.text.strip()
                anio_text = y
                break

    fecha_completa = f"{dia_text} DE {mes_text} DE {anio_text}"
    logger.info(
        "Estructura fecha: DIA=%r (limpio: %r), MES=%r (limpio: %r), ANNO=%r (limpio: %r) -> %r",
        dia_raw, dia_text, mes_raw, mes_text, anio_raw, anio_text, fecha_completa,
    )

    source_tokens: list[OCRResult] = []
    if dia_token:
        source_tokens.append(dia_token)
    source_tokens.append(tokens[idx_de1])
    source_tokens.extend(mes_tokens)
    source_tokens.append(tokens[idx_de2])
    if anio_token:
        source_tokens.append(anio_token)
    source_tokens = [t for t in source_tokens if t.text.strip()]

    return [OCRResult(text=fecha_completa, confidence=1.0, cx=0.5, cy=0.5, height=0.1)], source_tokens


class FechaEmisionExtractor:
    """Extrae la fecha de emision de un cheque via OCR sobre el scan amplio."""

    def __init__(self, ocr_reader: OCRReader):
        self._ocr = ocr_reader

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
        result = self._extraer_por_ciudad_coma(tokens_scan, cheque_img, scan_h, scan_x0, debug_dir)
        if result is not None:
            return result

        # 2. DE-cluster / EL-anchor
        result = self._extraer_por_de_el(tokens_scan, cheque_img, scan_h, scan_x0, debug_dir)
        if result is not None:
            return result

        # 3. Fallback
        fallback_tokens = self._fallback_zona(tokens_scan, zona, debug_dir)
        return FechaEmisionResult(fecha_iso=None, tokens=fallback_tokens)

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
        filtered, source_tokens = _filtrar_tokens_fecha_estructura(scan_window)
        if len(filtered) == 1:
            iso = _fecha_completa_a_iso(filtered[0].text)
            if iso is not None:
                logger.info("Fecha completa por OCR: %s", iso)
                return FechaEmisionResult(fecha_iso=iso, tokens=scan_window)
            logger.info(
                "OCR incompleto, retornando tokens estructurales: %s",
                [t.text for t in source_tokens],
            )
            return FechaEmisionResult(fecha_iso=None, tokens=source_tokens)
        logger.info("OCR incompleto, retornando tokens: %s", [t.text for t in scan_window])
        return FechaEmisionResult(fecha_iso=None, tokens=scan_window)

    def _extraer_por_ciudad_coma(
        self,
        tokens_scan: list[OCRResult],
        cheque_img: np.ndarray,
        scan_h: int,
        scan_x0: int,
        debug_dir: Path | None,
    ) -> FechaEmisionResult | None:
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
            fecha_crop = self._get_debug_crop(cheque_img, tokens_scan, cy_emision, scan_h, scan_x0)
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
        debug_dir: Path | None,
    ) -> FechaEmisionResult | None:
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
            fecha_crop = self._get_debug_crop(cheque_img, tokens_scan, cy_emision, scan_h, scan_x0)
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
