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
_FOUR_DIGITS_RE = re.compile(r'\d{4}')

_VENTANA_CY = 0.07

_MES_A_NUM = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
}

_DE_EMBEDDED_SPLIT_RE = re.compile(r'(DE)', re.IGNORECASE)
_EL_STRIP_RE = re.compile(r'^[Ee][Ll]')

# OCR character confusion rules: (ocr_char, true_char, field_context)
# day/year: letters/symbols that OCR reads instead of digits (forward substitution)
# month: digits that OCR reads instead of letters (inverse substitution)
_OCR_CHAR_RULES: list[tuple[str, str, str]] = [
    # Day
    ('Z', '2', 'day'), ('z', '2', 'day'),
    ('S', '5', 'day'), ('s', '5', 'day'),
    ('O', '0', 'day'), ('o', '0', 'day'),
    ('I', '1', 'day'), ('i', '1', 'day'),
    ('/', '1', 'day'),
    # Year
    ('Z', '2', 'year'), ('z', '2', 'year'),
    ('O', '0', 'year'), ('o', '0', 'year'),
    ('7', '2', 'year'),
    # Month (inverse: digit → letter look-alike)
    ('0', 'o', 'month'), ('1', 'i', 'month'),
    ('2', 'z', 'month'), ('3', 'e', 'month'),
]

_DAY_OCR_SUBS  = str.maketrans({c: t for c, t, ctx in _OCR_CHAR_RULES if ctx == 'day'})
_YEAR_OCR_SUBS = str.maketrans({c: t for c, t, ctx in _OCR_CHAR_RULES if ctx == 'year'})
_MES_OCR_SUBS  = str.maketrans({c: t for c, t, ctx in _OCR_CHAR_RULES if ctx == 'month'})


def _trigrams(text: str) -> set[str]:
    return {text[i:i + 3] for i in range(len(text) - 2)}


_MES_TRIGRAMS: dict[str, set[str]] = {mes: _trigrams(mes) for mes in _MESES_NOMBRES}


def _mes_por_trigrams(alpha_only: str) -> str | None:
    token_tgrams = _trigrams(alpha_only)
    scores = {m: len(token_tgrams & tgrams) for m, tgrams in _MES_TRIGRAMS.items()}
    best = max(scores.values())
    if best >= 2:
        winners = [m for m, s in scores.items() if s == best]
        if len(winners) == 1:
            return winners[0].capitalize()
    return None


@dataclass
class Fecha:
    """Date components, each validated or None if ambiguous/unrecognized by OCR.

    Validated slots (dia/mes/anno) hold clean values the code confirmed; raw slots
    hold the original OCR text so the LLM can reason about unrecognized components.
    """
    dia: str | None      # zero-padded day "01"-"31", None if unrecognized
    mes: str | None      # month number "01"-"12", None if unrecognized
    anno: str | None     # 4-digit year "2020"-"2030", None if out of range
    dia_raw: str | None = None   # raw OCR text for day slot
    mes_raw: str | None = None   # raw OCR text for month slot
    anno_raw: str | None = None  # raw OCR text for year slot

    def to_iso(self) -> str | None:
        if self.dia and self.mes and self.anno:
            return f"{self.anno}-{self.mes}-{self.dia}"
        return None

    def any_known(self) -> bool:
        return any(v is not None for v in (self.dia, self.mes, self.anno))

    def all_known(self) -> bool:
        return all(v is not None for v in (self.dia, self.mes, self.anno))


# Keep alias for any external code that still references PartialFecha
PartialFecha = Fecha


@dataclass
class FechaResult:
    fecha_iso: str | None
    tokens: list[OCRResult]
    partial: "Fecha | None" = None


def _validar_componentes(
    dia_raw: str | None,
    mes_raw: str | None,
    anno_raw: str | None,
) -> Fecha:
    """Validates raw OCR text per slot. Returns Fecha with clean values where unambiguous.

    Only accepts values the code can confirm with certainty:
      - day: numeric 1-31
      - month: recognized Spanish month name
      - year: exactly 20XX in range 2020-2030 (e.g. "7076" normalized to "2076" → valid)
    """
    dia = None
    if dia_raw:
        for d in re.findall(r'\d+', _limpiar_dia(dia_raw)):
            n = int(d)
            if 1 <= n <= 31:
                dia = str(n).zfill(2)
                break

    mes = None
    if mes_raw:
        for nombre, num in _MES_A_NUM.items():
            if nombre in mes_raw.lower():
                mes = num
                break

    anno = None
    if anno_raw:
        for y_str in _FOUR_DIGITS_RE.findall(anno_raw):
            if 2020 <= int(y_str) <= 2030:
                anno = y_str
                break

    return Fecha(
        dia=dia, mes=mes, anno=anno,
        dia_raw=dia_raw, mes_raw=mes_raw, anno_raw=anno_raw,
    )


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
    normalized = text.translate(_DAY_OCR_SUBS)
    for d in re.findall(r'\d+', normalized):
        num = int(d)
        if 1 <= num <= 31:
            return str(num).zfill(2)
    return text


def _limpiar_mes(text: str) -> str:
    text_lower = re.sub(r'\s+', '', text.lower().strip())
    for mes in _MESES_NOMBRES:
        if mes in text_lower:
            return mes.capitalize()
    normalized = text_lower.translate(_MES_OCR_SUBS)
    for mes in _MESES_NOMBRES:
        if mes in normalized:
            return mes.capitalize()
    for prefix_len in (2, 3):
        if len(normalized) >= prefix_len:
            matches = [m for m in _MESES_NOMBRES if m.startswith(normalized[:prefix_len])]
            if len(matches) == 1:
                return matches[0].capitalize()
    alpha_only = re.sub(r'[^a-z]', '', normalized)
    if len(alpha_only) >= 3:
        result = _mes_por_trigrams(alpha_only)
        if result is not None:
            return result
    return normalized.upper() if normalized != text_lower else text


def _limpiar_ano(text: str) -> str:
    for year in _FOUR_DIGITS_RE.findall(text):
        if 2020 <= int(year) <= 2030:
            return year
    normalized = text.translate(_YEAR_OCR_SUBS)
    for year in _FOUR_DIGITS_RE.findall(normalized):
        if 2020 <= int(year) <= 2030:
            return year
    for partial in re.findall(r'\d{3}', normalized):
        candidate = '20' + partial[-2:]
        if 2020 <= int(candidate) <= 2030:
            return candidate
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
        check_text = text.translate(_DAY_OCR_SUBS)
        has_embedded = bool(
            re.search(r'\dDE|DE\d', check_text, re.IGNORECASE)
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


def _filtrar_tokens_fecha_un_de(
    tokens: list[OCRResult],
    idx_de: int,
) -> tuple[list[OCRResult], list[OCRResult], "PartialFecha | None"]:
    """Parse a date line with only one DE (the year separator): [DIA] [MES...] DE [ANNO].

    The first digit-containing token before DE is treated as the day; remaining
    non-boilerplate tokens before DE are the month; first token after DE is the year.
    """
    before = [t for t in tokens[:idx_de] if not _BOILERPLATE_RE.match(t.text.strip())]
    after = [t for t in tokens[idx_de + 1:] if not _BOILERPLATE_RE.match(t.text.strip())]

    dia_token: OCRResult | None = None
    mes_tokens: list[OCRResult] = []
    for t in before:
        if dia_token is None and re.search(r'\d', t.text):
            dia_token = t
        else:
            mes_tokens.append(t)

    anio_token: OCRResult | None = None
    for t in after:
        candidate = _limpiar_ano(t.text.strip())
        if _ANNO_RE.match(candidate):
            anio_token = t
            break

    dia_raw = dia_token.text.strip() if dia_token else ""
    mes_raw = " ".join(t.text.strip() for t in mes_tokens)
    anio_raw = anio_token.text.strip() if anio_token else ""

    dia_text = _limpiar_dia(dia_raw)
    mes_text = _limpiar_mes(mes_raw)
    anio_text = _limpiar_ano(anio_raw)

    fecha_completa = f"{dia_text} DE {mes_text} DE {anio_text}"
    logger.info(
        "Estructura fecha (1 DE): DIA=%r (limpio: %r), MES=%r (limpio: %r), ANNO=%r (limpio: %r) -> %r",
        dia_raw, dia_text, mes_raw, mes_text, anio_raw, anio_text, fecha_completa,
    )

    source_tokens = [t for t in ([dia_token] + mes_tokens + [tokens[idx_de]] + ([anio_token] if anio_token else [])) if t is not None and t.text.strip()]

    fecha = _validar_componentes(
        dia_raw if dia_raw else None,
        mes_raw if mes_raw else None,
        anio_raw if anio_raw else None,
    )

    return (
        [OCRResult(text=fecha_completa, confidence=1.0, cx=0.5, cy=0.5, height=0.1)],
        source_tokens,
        fecha,
    )


def _filtrar_tokens_fecha_estructura(
    tokens: list[OCRResult],
    skip_el_prefix: bool = False,
    allow_single_de: bool = False,
) -> tuple[list[OCRResult], list[OCRResult], "PartialFecha | None"]:
    """Extrae DIA/MES/ANNO de los tokens.

    Returns:
        (combined, source_tokens, partial): combined is a single-element list with the
        assembled date string; source_tokens contains only the tokens that formed the
        structure; partial holds whichever components OCR did recognize (None = not found).
        When fewer than 2 DE are found and allow_single_de is False, returns (tokens, tokens, None).

    skip_el_prefix: pass True for fecha_pago lines that begin with 'EL'.
    allow_single_de: pass True for fecha_emision, where the first DE (day-month separator)
        may be absent. Format: [DIA] [MES...] DE [ANNO].
    """
    tokens = _expandir_tokens_de(tokens)
    de_indices = [i for i, t in enumerate(tokens) if _DE_RE.match(t.text.strip())]

    if len(de_indices) < 2:
        if allow_single_de and len(de_indices) == 1:
            return _filtrar_tokens_fecha_un_de(tokens, de_indices[0])
        logger.info("Estructura fecha: menos de 2 'DE' encontrados, retornando tokens originales")
        return tokens, tokens, None

    idx_de1, idx_de2 = de_indices[0], de_indices[1]

    el_token, dia_parts, dia_raw = _extraer_dia(tokens, idx_de1, skip_el_prefix)

    mes_tokens = [
        tokens[i] for i in range(idx_de1 + 1, idx_de2)
        if not _BOILERPLATE_RE.match(tokens[i].text.strip())
        and re.search(r'[a-zA-Z0-9]', tokens[i].text)
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

    fecha = _validar_componentes(
        dia_raw if dia_raw else None,
        mes_raw if mes_raw else None,
        anio_raw if anio_raw else None,
    )

    return (
        [OCRResult(text=fecha_completa, confidence=1.0, cx=0.5, cy=0.5, height=0.1)],
        [t for t in source_tokens if t.text.strip()],
        fecha,
    )


def _agrupar_de_clusters(de_tokens: list[OCRResult]) -> list[list[OCRResult]]:
    clusters: list[list[OCRResult]] = [[de_tokens[0]]]
    for tok in de_tokens[1:]:
        if tok.cy - clusters[-1][-1].cy < 0.06:
            clusters[-1].append(tok)
        else:
            clusters.append([tok])
    return clusters
