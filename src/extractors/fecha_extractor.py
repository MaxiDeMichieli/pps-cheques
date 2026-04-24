"""Utilidades OCR compartidas para FechaEmisionExtractor y FechaPagoExtractor."""

import logging
import re
from dataclasses import dataclass

from ..ocr.ocr_readers import OCRResult

logger = logging.getLogger(__name__)

_MESES_NOMBRES = {
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
}

# Acepta 'DE', 'de', 'De', 'DE.', etc. (case-insensitive, whole-token match)
_DE_RE = re.compile(r'^[^a-zA-Z0-9]*[Dd][Ee][^a-zA-Z0-9]*$')

# Detecta 'EL' incluso fusionado con el dia siguiente (ELJ8, EL19DE, ELZo, EL.)
_EL_INICIO_RE = re.compile(r'^[Ee][Ll]')

# Tokens que identifican el boilerplate "La fecha de pago no puede exceder un plazo de 360 dias"
_BOILERPLATE_RE = re.compile(r'^(360|dias?)$', re.IGNORECASE)
# Ancla fiable para el limite inferior del fallback: 'plazo' de la leyenda 'plazo de 360 dias'
_PLAZO_360_RE = re.compile(r'^plazo$', re.IGNORECASE)

_ANNO_RE = re.compile(r'^20\d{2}$')

_VENTANA_CY = 0.07

_MES_A_NUM = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
}

_DE_EMBEDDED_SPLIT_RE = re.compile(r'(DE)', re.IGNORECASE)
_EL_STRIP_RE = re.compile(r'^[Ee][Ll]')


@dataclass
class FechaResult:
    fecha_iso: str | None
    tokens: list[OCRResult]


def _es_token_fecha(text: str) -> bool:
    t = text.strip().lower()
    return t in _MESES_NOMBRES or bool(_ANNO_RE.match(t))


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
    for d in re.findall(r'\d+', text):
        num = int(d)
        if 1 <= num <= 31:
            return str(num).zfill(2)
    return text


def _limpiar_mes(text: str) -> str:
    text_lower = text.lower().strip()
    for mes in _MESES_NOMBRES:
        if mes in text_lower:
            return mes.capitalize()
    return text


def _limpiar_ano(text: str) -> str:
    for year in re.findall(r'\d{4}', text):
        if 2020 <= int(year) <= 2030:
            return year
    return text


def _expandir_tokens_de(tokens: list[OCRResult]) -> list[OCRResult]:
    """Split tokens with embedded 'DE' into sub-tokens.

    Only splits when DE is adjacent to a digit or at the token start
    (e.g. '16DE'→['16','DE'], 'DEZ025'→['DE','Z025']).
    Pure-word tokens like 'FEDERAL' are left untouched.
    """
    result: list[OCRResult] = []
    for t in tokens:
        text = t.text.strip()
        if _DE_RE.match(text):
            result.append(t)
            continue
        has_embedded = bool(
            re.search(r'\dDE|DE\d', text, re.IGNORECASE)
            or re.match(r'^DE.', text, re.IGNORECASE)
        )
        if not has_embedded:
            result.append(t)
            continue
        for p in (p for p in _DE_EMBEDDED_SPLIT_RE.split(text) if p):
            result.append(OCRResult(text=p, confidence=t.confidence, cx=t.cx, cy=t.cy, height=t.height))
    return result


def _extraer_dia(
    tokens: list[OCRResult],
    idx_de1: int,
    skip_el_prefix: bool,
) -> tuple[OCRResult | None, list[OCRResult], str]:
    """Returns (el_token, dia_parts, dia_raw) for the day portion of a date line.

    When skip_el_prefix is True and the first token starts with EL, it is treated
    as the fecha_pago marker; digits fused with it (e.g. 'EL19') plus any following
    tokens before the first DE are all concatenated as the day text.
    """
    if skip_el_prefix and idx_de1 > 0 and _EL_INICIO_RE.match(tokens[0].text.strip()):
        el_token = tokens[0]
        el_suffix = _EL_STRIP_RE.sub('', el_token.text.strip()).strip()
        dia_parts = [
            tokens[i] for i in range(1, idx_de1)
            if not _BOILERPLATE_RE.match(tokens[i].text.strip())
        ]
        return el_token, dia_parts, el_suffix + "".join(t.text.strip() for t in dia_parts)
    if idx_de1 > 0 and not _BOILERPLATE_RE.match(tokens[idx_de1 - 1].text.strip()):
        dia = tokens[idx_de1 - 1]
        return None, [dia], dia.text.strip()
    return None, [], ""


def _buscar_anno_fallback(
    tokens: list[OCRResult],
    assigned_ids: set[int],
) -> tuple[OCRResult | None, str, str]:
    """Scan unassigned tokens for a valid 4-digit year. Returns (token, raw, text) or Nones."""
    for t in tokens:
        if id(t) in assigned_ids:
            continue
        y = _limpiar_ano(t.text.strip())
        if _ANNO_RE.match(y):
            return t, t.text.strip(), y
    return None, "", ""


def _filtrar_tokens_fecha_estructura(
    tokens: list[OCRResult],
    skip_el_prefix: bool = False,
) -> tuple[list[OCRResult], list[OCRResult]]:
    """Extrae DIA/MES/ANNO de los tokens.

    Returns:
        (combined, source_tokens): combined is a single-element list with the assembled
        date string; source_tokens contains only the tokens that formed the structure.
        When fewer than 2 DE are found, returns (tokens, tokens).

    skip_el_prefix: pass True for fecha_pago lines that begin with 'EL'.
    """
    tokens = _expandir_tokens_de(tokens)
    de_indices = [i for i, t in enumerate(tokens) if _DE_RE.match(t.text.strip())]

    if len(de_indices) < 2:
        logger.info("Estructura fecha: menos de 2 'DE' encontrados, retornando tokens originales")
        return tokens, tokens

    idx_de1, idx_de2 = de_indices[0], de_indices[1]

    el_token, dia_parts, dia_raw = _extraer_dia(tokens, idx_de1, skip_el_prefix)

    mes_tokens = [
        tokens[i] for i in range(idx_de1 + 1, idx_de2)
        if not _BOILERPLATE_RE.match(tokens[i].text.strip())
    ]

    anio_token: OCRResult | None = None
    if idx_de2 + 1 < len(tokens) and not _BOILERPLATE_RE.match(tokens[idx_de2 + 1].text.strip()):
        anio_token = tokens[idx_de2 + 1]

    mes_raw = " ".join(t.text.strip() for t in mes_tokens) if mes_tokens else ""
    anio_raw = anio_token.text.strip() if anio_token else ""

    dia_text = _limpiar_dia(dia_raw)
    mes_text = _limpiar_mes(mes_raw)
    anio_text = _limpiar_ano(anio_raw)

    # Fallback: year may appear before the DE...DE structure due to OCR ordering.
    if not _ANNO_RE.match(anio_text):
        assigned_ids = (
            ({id(el_token)} if el_token else set())
            | {id(t) for t in dia_parts}
            | {id(tokens[idx_de1]), id(tokens[idx_de2])}
            | {id(t) for t in mes_tokens}
        )
        fb_token, fb_raw, fb_text = _buscar_anno_fallback(tokens, assigned_ids)
        if fb_token is not None:
            anio_token, anio_raw, anio_text = fb_token, fb_raw, fb_text

    fecha_completa = f"{dia_text} DE {mes_text} DE {anio_text}"
    logger.info(
        "Estructura fecha: DIA=%r (limpio: %r), MES=%r (limpio: %r), ANNO=%r (limpio: %r) -> %r",
        dia_raw, dia_text, mes_raw, mes_text, anio_raw, anio_text, fecha_completa,
    )

    source_tokens: list[OCRResult] = []
    if el_token:
        source_tokens.append(el_token)
    source_tokens.extend(dia_parts)
    source_tokens.append(tokens[idx_de1])
    source_tokens.extend(mes_tokens)
    source_tokens.append(tokens[idx_de2])
    if anio_token:
        source_tokens.append(anio_token)

    return (
        [OCRResult(text=fecha_completa, confidence=1.0, cx=0.5, cy=0.5, height=0.1)],
        [t for t in source_tokens if t.text.strip()],
    )


def _agrupar_de_clusters(de_tokens: list[OCRResult]) -> list[list[OCRResult]]:
    clusters: list[list[OCRResult]] = [[de_tokens[0]]]
    for tok in de_tokens[1:]:
        if tok.cy - clusters[-1][-1].cy < 0.06:
            clusters[-1].append(tok)
        else:
            clusters.append([tok])
    return clusters
