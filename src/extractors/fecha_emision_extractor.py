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
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ..ocr.ocr_readers import OCRReader, OCRResult

if TYPE_CHECKING:
    from ..llm.llm_backends import OllamaBackend

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

# Acepta 'DE', 'de', 'DE.', '_DE_', etc. (solo Mayúsculas)
_DE_MAYUS_RE = re.compile(r'^[^a-zA-Z0-9]*DE[^a-zA-Z0-9]*$')

# Detecta 'EL' incluso fusionado con el dia siguiente (ELJ8, EL19DE, ELZo, EL.)
_EL_INICIO_RE = re.compile(r'^[Ee][Ll]')

# Tokens que identifican el boilerplate "La fecha de pago no puede exceder un plazo de 360 dias"
_BOILERPLATE_RE = re.compile(r'^(360|dias?)$', re.IGNORECASE)

_DEBUG_FECHA_ZONA = "fecha_zona.png"
_VENTANA_CY = 0.07
_ISO_DATE_RE = re.compile(r'\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])')
# Matches a token that is a city name ending with a comma, e.g. 'FEDERAL,' 'CUATIA,'
_CIUDAD_COMA_RE = re.compile(r'^[A-Za-z\u00C0-\u024F][A-Za-z\u00C0-\u024F]+,$')


# llava works best with images at least this tall; short crops get upscaled.
_VISION_MIN_HEIGHT_PX = 200


def _build_vision_fecha_prompt(ocr_tokens: list[str], debug: bool) -> str:
    """Builds the vision prompt with OCR tokens + image as dual sources."""
    token_str = " ".join(ocr_tokens) if ocr_tokens else "(none)"
    debug_block = (
        "SINCE THIS IS DEBUG MODE: before the ISO date, add exactly one line starting with "
        "'REASONING:' explaining, for each field (day / month / year), which source you used "
        "and why.\n\n"
    ) if debug else ""
    return """\
You are reading the emission date line of an Argentine bank cheque (CPD).
The date is handwritten in the format: DD DE MES DE YYYY
The line usually starts with a city name (e.g. QUILMES, DON TORCUATO).

You have TWO sources of information:
  1. OCR TOKENS from the date area (left to right, may contain noise): {token_str}
  2. The attached IMAGE of the same date area.

TASK — determine the best reading for each field independently:
  - DAY  (01-31): which source gives you the clearest number?
  - MONTH (Spanish name): which source gives you the clearest text?
  - YEAR (2024-2027): which source gives you the clearest number?

For each field, prefer the source where the evidence is unambiguous.
If the OCR tokens contain a clear month name (e.g. "enero", "Nov", "Fe") that is
recognisable, trust that over the image. If they contain a clear 4-digit year or a
clear 1-2 digit day, trust those too. Only fall back to the image for fields where
the tokens are garbled, missing, or contradictory.

READING RULES (apply to both sources):
- The month is always one of exactly 12 Spanish names. Match partial or garbled text to
  the ONLY month it could belong to:
    Fe / Feb              → Febrero   (02) — only month starting Fe
    En / Ene              → Enero     (01) — only month starting En
    Ma (short)            → Marzo     (03); Ma + y / longer → Mayo (05)
    Ab / Abr              → Abril     (04) — only month starting Ab
    Ju + n                → Junio     (06); Ju + l → Julio (07)
    Ag / Ago              → Agosto    (08) — only month starting Ag
    Se / Sep              → Septiembre(09) — only month starting Se
    Oc / Oct              → Octubre   (10) — only month starting Oc
    No / Nov              → Noviembre (11) — only month starting No
    Di / Dic              → Diciembre (12) — only month starting Di
- Digits 0/O and 1/l are often confused — use context (valid day 01-31,
  valid year 2024-2027) to resolve them. Example: Z026 → 2026, l6 → 16.
- Commit to your best reading. Do not refuse because of imperfect handwriting or noise.

{debug_block}OUTPUT RULES (be strict):
- Reply with the date in ISO format: YYYY-MM-DD (e.g. 2026-01-15)
- No extra text or punctuation around the ISO date.
- Reply null ONLY if neither source lets you determine any part of the date.\
""".format(token_str=token_str, debug_block=debug_block)

def _es_fecha_valida(text: str) -> bool:
    return bool(re.match(r'\d{2} DE [A-Za-z]+ DE \d{4}', text))

def _limpiar_dia(text: str) -> str:
    """Extrae solo los dígitos que forman un día válido (01-31) del texto sucio."""
    # Busca secuencias de dígitos
    digits = re.findall(r'\d+', text)
    for d in digits:
        num = int(d)
        if 1 <= num <= 31:
            return str(num).zfill(2)  # Retorna con 0 si es necesario
    return text  # Fallback: devuelve el original


def _limpiar_mes(text: str) -> str:
    """Extrae el mes en español válido del texto sucio.
    
    Busca en la lista de meses válidos, incluso parcialmente en el texto.
    """
    text_lower = text.lower().strip()
    # Busca match con cada mes (orden importa para evitar confusiones)
    for mes in ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]:
        if mes in text_lower:
            return mes.capitalize()
    return text  # Fallback: devuelve el original


def _limpiar_ano(text: str) -> str:
    """Extrae solo los dígitos que forman un año válido (2020-2030) del texto sucio."""
    # Busca secuencias de 4 dígitos
    years = re.findall(r'\d{4}', text)
    for year in years:
        num = int(year)
        if 2020 <= num <= 2030:  # Rango realista para cheques
            return year
    return text  # Fallback: devuelve el original


def _filtrar_tokens_fecha_estructura(tokens: list[OCRResult]) -> list[OCRResult]:
    """Filtra tokens para mantener solo la estructura: DIA DE MES DE AÑO
    
    Encuentra los dos "DE" en MAYÚSCULAS y extrae:
    - Token anterior al primer DE → DIA
    - Tokens entre los dos DE → MES
    - Token posterior al segundo DE → AÑO
    
    Devuelve los tres componentes combinados en un único token.
    También descarta tokens que correspondan al boilerplate.
    """
    de_indices = [i for i, t in enumerate(tokens) if _DE_MAYUS_RE.match(t.text.strip())]
    
    if len(de_indices) < 2:
        logger.info("Estructura fecha: menos de 2 'DE' encontrados, retornando tokens originales")
        return tokens
    
    idx_de1 = de_indices[0]
    idx_de2 = de_indices[1]
    
    # Extraer DIA (token anterior al primer DE, descartando boilerplate)
    dia_token = None
    if idx_de1 > 0 and not _BOILERPLATE_RE.match(tokens[idx_de1 - 1].text.strip()):
        dia_token = tokens[idx_de1 - 1]
    
    # Extraer MES (tokens entre los dos DE, descartando boilerplate)
    mes_tokens = []
    for i in range(idx_de1 + 1, idx_de2):
        if not _BOILERPLATE_RE.match(tokens[i].text.strip()):
            mes_tokens.append(tokens[i])
    
    # Extraer AÑO (token posterior al segundo DE, descartando boilerplate)
    anio_token = None
    if idx_de2 + 1 < len(tokens) and not _BOILERPLATE_RE.match(tokens[idx_de2 + 1].text.strip()):
        anio_token = tokens[idx_de2 + 1]
    
    # Limpiar cada componente
    dia_raw = dia_token.text.strip() if dia_token else ""
    mes_raw = " ".join(t.text.strip() for t in mes_tokens) if mes_tokens else ""
    anio_raw = anio_token.text.strip() if anio_token else ""
    
    dia_text = _limpiar_dia(dia_raw)
    mes_text = _limpiar_mes(mes_raw)
    anio_text = _limpiar_ano(anio_raw)
    
    fecha_completa = f"{dia_text} DE {mes_text} DE {anio_text}"
    logger.info(
        "Estructura fecha: DIA=%r (limpio: %r), MES=%r (limpio: %r), AÑO=%r (limpio: %r) -> %r",
        dia_raw, dia_text, mes_raw, mes_text, anio_raw, anio_text, fecha_completa,
    )
    
    # Devolver un único token con la fecha completa
    resultado_token = OCRResult(
        text=fecha_completa,
        confidence=1.0,
        cx=0.5,
        cy=0.5,
        height=0.1,
    )
    return [resultado_token]


def make_vision_fecha_fn(backend: "OllamaBackend") -> "Callable[[np.ndarray, list[str], bool], str | None]":
    """Retorna una vision_fn lista para inyectar en FechaEmisionExtractor."""
    def _fn(img: np.ndarray, ocr_tokens: list[str], debug: bool = False) -> str | None:
        prompt = _build_vision_fecha_prompt(ocr_tokens, debug)
        messages = [{"role": "user", "content": prompt}]
        return backend.chat_vision(messages, [img])
    return _fn


class FechaEmisionExtractor:
    """Lee los tokens OCR de la zona de fecha de emision."""

    def __init__(
        self,
        ocr_reader: OCRReader,
        crop_ocr_reader: OCRReader | None = None,
        vision_fn: "Callable[[np.ndarray, list[str], bool], str | None] | None" = None,
    ):
        self._ocr = ocr_reader
        self._crop_ocr = crop_ocr_reader or ocr_reader
        self._vision_fn = vision_fn

    def leer_tokens(
        self,
        cheque_img: np.ndarray,
        debug_dir: Path | None = None,
    ) -> tuple[list[OCRResult], list[OCRResult]]:
        """Devuelve (crop_tokens, scan_window_tokens).

        crop_tokens: tokens del crop re-OCR (o fallback del scan amplio).
        scan_window_tokens: tokens del scan amplio dentro de ±_VENTANA_CY
            alrededor del cy de la linea de emision. Mas limpios que el crop
            re-OCR para inferencia LLM. Vacio si no se detecto cy_emision.
        """
        h, w = cheque_img.shape[:2]
        scan_h = int(h * 0.55)
        scan_x0 = int(w * 0.10)
        zona = cheque_img[0:scan_h, scan_x0:w]
        tokens_scan = self._ocr.read(zona)
        logger.info("Tokens scan: %s", [(t.text, round(t.cy, 3)) for t in tokens_scan])
        # 1. Ciudad-coma anchor: token like 'FEDERAL,' or 'CUATIA,' marks the
        #    start of the emission date line. Crop starts just after that token.
        ciudad_coma = self._encontrar_ciudad_coma(tokens_scan)
        if ciudad_coma is not None:
            cy_emision = ciudad_coma.cy
            x_left = scan_x0 + int(ciudad_coma.cx * (w - scan_x0))
            logger.info(
                "Ciudad-coma ancla: token=%r cy=%.3f cx=%.3f -> x_left=%d",
                ciudad_coma.text, cy_emision, ciudad_coma.cx, x_left,
            )
            scan_window = [t for t in tokens_scan if abs(t.cy - cy_emision) < _VENTANA_CY]
            logger.info(
                "Scan window (cy=%.3f \u00b1%.3f): %s",
                cy_emision, _VENTANA_CY,
                [(t.text, round(t.cy, 3)) for t in scan_window],
            )
            tokens = self._crop_por_cy(cheque_img, tokens_scan, cy_emision, w, scan_h, scan_x0, debug_dir, x_left=x_left)
            if tokens:
                return tokens, scan_window

        # 2. Existing DE-cluster / EL-ancla flow
        cy_emision = self._detectar_cy_emision(tokens_scan)

        scan_window: list[OCRResult] = []
        if cy_emision is not None:
            scan_window = [t for t in tokens_scan if abs(t.cy - cy_emision) < _VENTANA_CY]
            logger.info(
                "Scan window (cy=%.3f ±%.3f): %s",
                cy_emision, _VENTANA_CY,
                [(t.text, round(t.cy, 3)) for t in scan_window],
            )
            tokens = self._crop_por_cy(cheque_img, tokens_scan, cy_emision, w, scan_h, scan_x0, debug_dir)
            if tokens:
                return tokens, scan_window

        return self._fallback_fecha(tokens_scan, zona, debug_dir), scan_window

    @staticmethod
    def _encontrar_ciudad_coma(tokens_scan: list[OCRResult]) -> OCRResult | None:
        """Busca el token de ciudad con coma al final (e.g. 'FEDERAL,', 'CUATIA,')."""
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
        x_left: int | None = None,
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
        crop_x0 = x_left if x_left is not None else scan_x0

        fecha_crop = cheque_img[y0:y1, crop_x0:scan_x1]
        logger.info(
            "Fecha crop [y=%d:%d, x=%d:%d, token_h=%dpx]",
            y0, y1, crop_x0, scan_x1, token_h_px,
        )

        if fecha_crop.size == 0 or y1 <= y0:
            return []

        if debug_dir is not None:
            Image.fromarray(fecha_crop).save(debug_dir / _DEBUG_FECHA_ZONA)

        tokens = self._crop_ocr.read(fecha_crop)

        # Filtrar tokens para mantener solo la estructura DIA DE MES DE AÑO
        tokens = _filtrar_tokens_fecha_estructura(tokens)
        
        ocr_texts = [t.text for t in tokens]

        logger.info("Fecha crop -> %d tokens: %s", len(tokens), ocr_texts)

        if self._vision_fn is not None:
            result = self._llamar_vision_llm(fecha_crop, ocr_texts, debug_dir is not None)
            if result is not None:
                return result

        return tokens

    def _llamar_vision_llm(
        self,
        fecha_crop: np.ndarray,
        ocr_texts: list[str],
        debug_mode: bool,
    ) -> list[OCRResult] | None:
        """Llama al Vision LLM y retorna un OCRResult con la fecha ISO, o None si falla."""
        vision_input = self._upscale_for_vision(fecha_crop)
        raw = self._vision_fn(vision_input, ocr_texts, debug_mode)  # type: ignore[misc]
        logger.info("Vision LLM fecha raw: %r", raw)
        if not raw:
            logger.info("Vision LLM no retornó fecha ISO válida")
            return None
        if debug_mode:
            reasoning_match = re.search(r'REASONING:\s*(.+)', raw, re.IGNORECASE)
            if reasoning_match:
                logger.info("Vision LLM reasoning: %s", reasoning_match.group(1).strip())
        match = _ISO_DATE_RE.search(raw)
        if match:
            candidate = match.group(0)
            logger.info("Vision LLM fecha aceptada: %s", candidate)
            return [OCRResult(candidate, 1.0, 0.5, 0.5, 0.1)]
        logger.info("Vision LLM no retornó fecha ISO válida")
        return None

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

    @staticmethod
    def _upscale_for_vision(img: np.ndarray) -> np.ndarray:
        """Escala el crop para que tenga al menos _VISION_MIN_HEIGHT_PX de alto.

        Las tiras de fecha son muy anchas y bajas (~100×1500px). llava las
        comprime a 336×336 y pierde detalle. Escalar la altura ayuda.
        """
        h, w = img.shape[:2]
        if h >= _VISION_MIN_HEIGHT_PX:
            return img
        scale = _VISION_MIN_HEIGHT_PX / h
        new_w = int(w * scale)
        new_h = _VISION_MIN_HEIGHT_PX
        pil = Image.fromarray(img)
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
        logger.info("Vision crop upscaled %dx%d -> %dx%d", w, h, new_w, new_h)
        return np.array(pil)

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
