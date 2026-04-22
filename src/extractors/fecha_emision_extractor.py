"""Extractor de zona de fecha de emision de cheques.

Estrategia hibrida:
  1. OCR amplio sobre la mitad superior del cheque.
  2. Detectar el cy de la linea de emision usando el cluster superior de
     pares 'DE' (excluyendo boilerplate y linea de pago).
     Como refuerzo, si no se encuentra un par DE, se intenta localizar el
     token 'EL' (incluso fusionado: ELJ8, EL19DE...) y estimar la linea
     de emision como la linea inmediatamente anterior.
  3. Con el cy estimado, recortar el cheque original en esa banda vertical
     y re-ejecutar OCR sobre el crop — exactamente como hacia el enfoque
     original, pero con deteccion de linea mas robusta.
  Fallback: filtrar del scan amplio por tokens de mes/año conocido.
"""

import logging
import re
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
    if re.match(r'^20\d{2}$', t):
        return True
    return False


# Acepta 'DE', 'de', 'DE.', '_DE_', etc.
_DE_RE = re.compile(r'^[^a-zA-Z0-9]*[Dd][Ee][^a-zA-Z0-9]*$')

# Detecta 'EL' incluso fusionado con el dia siguiente (ELJ8, EL19DE, ELZo, EL.)
_EL_INICIO_RE = re.compile(r'^[Ee][Ll]')

# Tokens que identifican el boilerplate "La fecha de pago no puede exceder un plazo de 360 dias"
_BOILERPLATE_RE = re.compile(r'^(360|dias?)$', re.IGNORECASE)

_DEBUG_FECHA_ZONA = "fecha_zona.png"
_VENTANA_CY = 0.07


class FechaEmisionExtractor:
    """Lee los tokens OCR de la zona de fecha de emision."""

    def __init__(self, ocr_reader: OCRReader, crop_ocr_reader: OCRReader | None = None):
        self._ocr = ocr_reader
        self._crop_ocr = crop_ocr_reader or ocr_reader

    def leer_tokens(self, cheque_img: np.ndarray, debug_dir: Path | None = None) -> list[OCRResult]:
        """Devuelve tokens OCR de la linea de fecha de emision.

        Paso 1: OCR sobre una franja amplia para detectar posicion de la linea.
        Paso 2: Crop + re-OCR sobre la banda de la linea de emision.
        Fallback: filtrar del scan amplio por tokens de mes/año conocido.
        """
        h, w = cheque_img.shape[:2]
        scan_h = int(h * 0.55)
        scan_x0 = int(w * 0.10)
        zona = cheque_img[0:scan_h, scan_x0:w]
        tokens_scan = self._ocr.read(zona)
        logger.info("Tokens scan: %s", [(t.text, round(t.cy, 3)) for t in tokens_scan])

        cy_emision = self._detectar_cy_emision(tokens_scan)
        if cy_emision is not None:
            tokens = self._crop_por_cy(cheque_img, tokens_scan, cy_emision, w, scan_h, scan_x0, debug_dir)
            if tokens:
                return tokens

        return self._fallback_fecha(tokens_scan, zona, debug_dir)

    def _detectar_cy_emision(self, tokens_scan: list[OCRResult]) -> float | None:
        """Detecta el cy (normalizado al scan) de la linea de emision."""
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

    def _crop_por_cy(
        self,
        cheque_img: np.ndarray,
        tokens_scan: list[OCRResult],
        cy_emision: float,
        w: int,
        scan_h: int,
        scan_x0: int,
        debug_dir: Path | None,
    ) -> list[OCRResult]:
        """Recorta la banda de la linea de emision y re-ejecuta OCR."""
        # Estimar altura de linea a partir del token EL o de un token DE cercano
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
        scan_x1 = self._detectar_limite_derecho(tokens_scan, cy_emision, scan_h, scan_x0, w, token_h_norm)

        fecha_crop = cheque_img[y0:y1, scan_x0:scan_x1]
        logger.info(
            "Fecha crop [y=%d:%d, x=%d:%d, token_h=%dpx]",
            y0, y1, scan_x0, scan_x1, token_h_px,
        )

        if fecha_crop.size == 0 or y1 <= y0:
            return []

        if debug_dir is not None:
            Image.fromarray(fecha_crop).save(debug_dir / _DEBUG_FECHA_ZONA)
        tokens = self._crop_ocr.read(fecha_crop)
        logger.info("Fecha crop -> %d tokens", len(tokens))
        return tokens

    @staticmethod
    def _detectar_limite_derecho(
        tokens_scan: list[OCRResult],
        cy_emision: float,
        scan_h: int,
        scan_x0: int,
        w: int,
        token_h_norm: float,
    ) -> int:
        """Devuelve el x absoluto donde comienza el identificador del cheque en la fila de emision."""
        fila = [t for t in tokens_scan if abs(t.cy - cy_emision) < token_h_norm]
        id_tokens = [t for t in fila if re.match(r'^\d{6,}$', t.text.strip()) and t.cx > 0.4]
        if id_tokens:
            leftmost = min(id_tokens, key=lambda t: t.cx)
            x1 = scan_x0 + int(leftmost.cx * (w - scan_x0))
            logger.info("Limite derecho: token %r en cx=%.2f -> x=%d", leftmost.text, leftmost.cx, x1)
            return x1
        return w

    def _fallback_fecha(
        self,
        tokens_scan: list[OCRResult],
        zona: np.ndarray,
        debug_dir: Path | None,
    ) -> list[OCRResult]:
        """Fallback: filtra tokens del scan amplio por mes/año conocido."""
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
