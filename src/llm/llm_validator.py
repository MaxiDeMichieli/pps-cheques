"""Validacion y extraccion de campos de cheques usando LLM via backend intercambiable.

El LLM recibe los tokens OCR del cheque (texto + posiciones normalizadas) y
el contexto del lote (montos ya vistos) para extraer campos estructurados con
un score de confianza calibrado (0.0-1.0).

Campos soportados actualmente:
- monto: importe numerico en formato argentino (ej. "4.000.000")
- fecha_emision: fecha de emision (ej. "11 DE Febrero DE 2026" -> "2026-02-11")
"""

import json
import re
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from .llm_backends import LLMBackend
from ..ocr.ocr_readers import OCRResult
from ..extractors.fecha_extractor import Fecha


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
Sos un experto en lectura de cheques bancarios argentinos (Cheques de Pago Diferido - CPD).
Tu tarea es extraer campos especificos a partir de los tokens detectados por OCR en la imagen del cheque.

### Campos a extraer

1. **monto**: El importe del cheque. Esta escrito a mano en el recuadro que esta a la derecha del signo "$".
   - Formato argentino: punto = separador de miles, coma = decimales. Ej: "4.000.000", "802.470,20"
   - Nunca contiene letras. Si el OCR confunde un digito con una letra (ej. "l" por "1"), corregilo.
   - Ignorar el numero de cheque (tipicamente 8 digitos sin puntos, ej. "14193346").

2. **fecha_emision**: La fecha de emision del cheque. Aparece en la linea que comienza con el nombre
   de una ciudad (ej. "QUILMES") seguido de la fecha en formato "DD DE MES DE AAAA".
   - Normalizar al formato ISO: "YYYY-MM-DD".
   - Meses en espanol: Enero=01, Febrero=02, Marzo=03, Abril=04, Mayo=05, Junio=06,
     Julio=07, Agosto=08, Septiembre=09, Octubre=10, Noviembre=11, Diciembre=12.

### Calibracion de confianza
- 0.95-1.00: valor inequivoco, formato estandar reconocible
- 0.80-0.94: legible con algo de ruido OCR, reconstruccion segura
- 0.60-0.79: parcialmente reconstruido con ayuda del contexto del lote
- 0.00-0.59: el LLM esta adivinando, resultado no confiable

### Formato de respuesta
Responde UNICAMENTE con un objeto JSON valido, sin texto adicional, sin markdown:
{
  "monto": {
    "value": "<string tal como aparece en el cheque, o null>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<explicacion breve en una oracion>"
  },
  "fecha_emision": {
    "value": "<YYYY-MM-DD normalizado, o null>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<explicacion breve en una oracion>"
  }
}
"""


@dataclass
class LLMExtractionResult:
    """Resultado de extraccion de un campo por el LLM."""
    value: str | None
    normalized: Any | None
    confidence: float
    reasoning: str


_FAILED_RESULT = LLMExtractionResult(
    value=None, normalized=None, confidence=0.0, reasoning="llm_unavailable"
)

_FECHA_SYSTEM_PROMPT = """\
Sos un experto en inferir fechas de emisión de cheques bancarios argentinos (CPD).
Recibís exactamente los caracteres que el OCR capturó del texto manuscrito.
Tu tarea es inferir la fecha más probable dado esos caracteres imperfectos.\
"""

_FECHA_USER_TEMPLATE = """\
Los siguientes tokens son exactamente lo que el OCR leyó de la línea de fecha \
manuscrita del cheque (pueden estar distorsionados por la escritura a mano):

  {tokens}

La fecha está en formato "DD DE MES DE YYYY".

El mes es SIEMPRE uno de estos 12, sin excepción:
  Enero=01  Febrero=02  Marzo=03    Abril=04
  Mayo=05   Junio=06    Julio=07    Agosto=08
  Septiembre=09  Octubre=10  Noviembre=11  Diciembre=12

Reglas de inferencia para el MES — identificá por el prefijo visible, \
cada prefijo es único salvo los indicados:
  Fe / Feb              → Febrero   (único con Fe)
  En / Ene              → Enero     (único con En)
  Ma (corto)            → Marzo; Ma + y o más letras → Mayo
  Ab / Abr              → Abril     (único con Ab)
  Ju + n                → Junio; Ju + l → Julio
  Ag / Ago              → Agosto    (único con Ag)
  Se / Sep              → Septiembre (único con Se)
  Oc / Oct              → Octubre   (único con Oc)
  No / Nov / Nor / Norr → Noviembre (ÚNICO mes que empieza con "No"; si ves cualquier "No..." el mes ES Noviembre, no existe otro)
  Di / Dic              → Diciembre (único con Di)

Reglas para DÍGITOS (confusiones OCR frecuentes):
  Z↔2, o/O↔0, l↔1, S↔5, G↔6, B↔8
  Ejemplos: "Z025" → 2025, "ZoZ6" → 2026, "l6" → 16

Reglas específicas para el DÍA — confusión "I" (letra) ↔ "1" (dígito):
  "I"         → 1   (ej: "I" solo → día 1)
  "II"        → 11
  "I1" o "1I" → 11
  "I" + dígito N → 1N  (ej: "I2"→12, "I3"→13, "I4"→14, ... "I9"→19)
  dígito N + "I" → N1  (ej: "3I"→31, "2I"→21, "1I"→11)
  Regla general: en el campo día, toda "I" mayúscula es el dígito "1".
- Si un token parece ruido sin letras ni dígitos reconocibles (ej. "DEZOZS"), ignoralo.

- {date_constraint}
- Si el año inferido es mayor a {max_year}, es un error OCR (probablemente Z→2).
- Comprometete con tu mejor lectura. No te niegues por escritura imperfecta o ruido.
{partial_hint}
Respondé ÚNICAMENTE con la fecha en formato ISO YYYY-MM-DD.
Sin texto adicional. Si es imposible inferir cualquier componente de la fecha, \
respondé null.\
"""

_MESES = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
}


def _normalizar_monto(value: str | None) -> float | None:
    """Convierte string de monto argentino a float."""
    if not value:
        return None
    limpio = value.strip().rstrip('-').rstrip(',')
    limpio = re.sub(r'^[\$Ss]\s*', '', limpio)
    if not limpio:
        return None
    try:
        if ',' in limpio:
            partes = limpio.split(',')
            entero = partes[0].replace('.', '')
            decimal = partes[1] if len(partes) > 1 else '0'
            return float(f"{entero}.{decimal}")
        return float(limpio.replace('.', ''))
    except ValueError:
        return None


def _normalizar_fecha(value: str | None) -> str | None:
    """Intenta parsear la fecha ya normalizada o en formato legible."""
    if not value:
        return None
    # Ya esta en ISO
    if re.match(r'^\d{4}-\d{2}-\d{2}$', value.strip()):
        return value.strip()
    # Formato "DD DE MES DE YYYY" o variantes
    m = re.search(
        r'(\d{1,2})\s+(?:DE\s+)?(\w+)\s+(?:DE\s+)?(\d{4})',
        value, re.IGNORECASE
    )
    if m:
        dia = m.group(1).zfill(2)
        mes_str = m.group(2).lower()
        anio = m.group(3)
        mes = _MESES.get(mes_str)
        if mes:
            return f"{anio}-{mes}-{dia}"
    return None


def _build_partial_hint(fecha: "Fecha | None") -> str:
    """Returns a prompt snippet differentiating confirmed OCR slots from those needing inference."""
    if fecha is None or not (fecha.any_known() or fecha.dia_raw or fecha.mes_raw or fecha.anno_raw):
        return ""
    lines = ["- El OCR procesó la estructura DIA DE MES DE AÑO:"]
    for label, validated, raw in [
        ("Día", fecha.dia, fecha.dia_raw),
        ("Mes", fecha.mes, fecha.mes_raw),
        ("Año", fecha.anno, fecha.anno_raw),
    ]:
        if validated is not None:
            lines.append(f"    {label}: CONFIRMADO = {validated}  (OCR raw: {raw!r})")
        elif raw:
            lines.append(f"    {label}: INFERIR desde token OCR = {raw!r}")
        else:
            lines.append(f"    {label}: sin datos")
    lines.append(
        "  Usá exactamente el valor de los slots CONFIRMADOS. "
        "Aplicá las reglas de confusión OCR solo para los slots a INFERIR."
    )
    return "\n".join(lines)


def _tokens_a_texto(ocr_tokens: list[OCRResult]) -> str:
    """Convierte lista de OCRResult a texto plano ordenado por posicion.

    Omite coordenadas y confianza para reducir el largo del prompt.
    Los tokens se ordenan fila a fila (cy redondeado) para que el LLM
    los lea en orden natural de lectura.
    """
    tokens_ordenados = sorted(ocr_tokens, key=lambda t: (round(t.cy, 1), t.cx))
    return " ".join(t.text for t in tokens_ordenados)


class LLMValidator:
    """Extrae y valida campos de cheques usando un LLM via backend intercambiable."""

    def __init__(self, backend: LLMBackend):
        self._backend = backend

    def extract_fields(
        self,
        ocr_tokens: list[OCRResult],
        batch_context: list[str] | None = None,
    ) -> dict[str, LLMExtractionResult]:
        """Extrae monto y fecha_emision de los tokens OCR del cheque.

        Args:
            ocr_tokens: Tokens OCR del cheque completo (texto + posiciones).
            batch_context: Montos raw de otros cheques del mismo lote, para
                           ayudar al LLM a desambiguar valores dudosos.

        Returns:
            Dict con claves "monto" y "fecha_emision", cada una con un
            LLMExtractionResult.
        """
        tokens_txt = _tokens_a_texto(ocr_tokens)
        logger.info("OCR tokens: %s", tokens_txt)

        user_message = self._build_user_message(ocr_tokens, batch_context or [])
        raw_response = self._call_llm(user_message)
        if raw_response is None:
            return {"monto": _FAILED_RESULT, "fecha_emision": _FAILED_RESULT}
        return self._parse_response(raw_response)

    def _build_user_message(
        self, ocr_tokens: list[OCRResult], batch_context: list[str]
    ) -> str:
        tokens_txt = _tokens_a_texto(ocr_tokens)
        partes = [
            "### Texto OCR del cheque (tokens en orden de lectura)\n",
            tokens_txt,
        ]
        if batch_context:
            ctx = ", ".join(f'"{m}"' for m in batch_context if m)
            partes.append(
                f"\n### Contexto del lote\nOtros cheques en este lote tienen montos: [{ctx}]. "
                "Usa esta informacion para detectar si un valor parece anomalo o si el OCR "
                "confundio digitos."
            )
        partes.append(
            "\nExtrae los campos solicitados y responde SOLO con el JSON indicado."
        )
        return "\n".join(partes)

    def _call_llm(self, user_message: str) -> str | None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        return self._backend.chat(messages)

    def infer_fecha(
        self,
        fecha_tokens: list[OCRResult],
        today_max: str | None = None,
        max_future_days: int | None = None,
        partial_fecha: "Fecha | None" = None,
    ) -> LLMExtractionResult:
        """Infiere una fecha a partir de los tokens crudos del crop de fecha.

        Args:
            fecha_tokens: Tokens OCR de la línea de fecha.
            today_max: Fecha máxima en formato ISO (para fecha_emision, debe ser <= hoy).
            max_future_days: Si se indica, la fecha puede ser futura hasta este número
                de días desde hoy (para fecha_pago; típicamente 365).
                Si se pasan ambos, today_max tiene precedencia.
            partial_fecha: Componentes que el OCR ya reconoció con certeza. El LLM
                debe usarlos como base y solo inferir los que falten.
        """
        if not fecha_tokens:
            return _FAILED_RESULT

        tokens_txt = " ".join(
            t.text for t in sorted(fecha_tokens, key=lambda t: t.cx) if t.text.strip()
        )
        logger.info("infer_fecha tokens: %s", tokens_txt)
        if partial_fecha is not None and (partial_fecha.any_known() or partial_fecha.dia_raw):
            logger.info(
                "infer_fecha fecha: dia=%r(raw=%r) mes=%r(raw=%r) anno=%r(raw=%r)",
                partial_fecha.dia, partial_fecha.dia_raw,
                partial_fecha.mes, partial_fecha.mes_raw,
                partial_fecha.anno, partial_fecha.anno_raw,
            )
        else:
            logger.info("infer_fecha partial: none")

        # Compute the effective upper bound for the date
        if today_max is not None:
            fecha_tope = today_max
        elif max_future_days is not None:
            fecha_tope = (date.today() + timedelta(days=max_future_days)).isoformat()
        else:
            fecha_tope = None

        if fecha_tope is not None:
            date_constraint = f"La fecha NO puede ser posterior a {fecha_tope}."
            max_year = fecha_tope[:4]
        else:
            date_constraint = "No hay restricción de fecha futura para este campo."
            max_year = str(date.today().year + 2)

        partial_hint = _build_partial_hint(partial_fecha)

        user_msg = _FECHA_USER_TEMPLATE.format(
            tokens=tokens_txt,
            date_constraint=date_constraint,
            max_year=max_year,
            partial_hint=partial_hint,
        )
        messages = [
            {"role": "system", "content": _FECHA_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw = self._backend.chat(messages)
        if raw is None:
            return _FAILED_RESULT

        candidate = raw.strip()
        logger.info("infer_fecha raw: %r", candidate)

        if candidate.lower() in ("null", "none", ""):
            return LLMExtractionResult(value=None, normalized=None, confidence=0.0, reasoning="llm_null")

        normalized = _normalizar_fecha(candidate)
        if normalized is None:
            logger.warning("infer_fecha: formato inesperado del LLM: %r", candidate[:80])
            return _FAILED_RESULT

        if fecha_tope is not None and normalized > fecha_tope:
            logger.warning("infer_fecha: fecha rechazada: %s > %s", normalized, fecha_tope)
            return _FAILED_RESULT

        return LLMExtractionResult(
            value=candidate,
            normalized=normalized,
            confidence=0.88,
            reasoning="inferida de tokens OCR con reglas de mes cerrado",
        )

    def _parse_response(self, raw: str) -> dict[str, LLMExtractionResult]:
        # Extraer JSON aunque el LLM agregue texto extra
        match = re.search(r'\{[\s\S]*\}', raw)
        if not match:
            logger.warning("LLM no devolvio JSON valido: %s", raw[:200])
            return {"monto": _FAILED_RESULT, "fecha_emision": _FAILED_RESULT}

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning("JSON invalido del LLM: %s", exc)
            return {"monto": _FAILED_RESULT, "fecha_emision": _FAILED_RESULT}

        monto_data = data.get("monto") or {}
        monto_value = monto_data.get("value")
        monto_result = LLMExtractionResult(
            value=monto_value,
            normalized=_normalizar_monto(monto_value),
            confidence=float(monto_data.get("confidence", 0.0)),
            reasoning=monto_data.get("reasoning", ""),
        )

        fecha_data = data.get("fecha_emision") or {}
        fecha_value = fecha_data.get("value")
        fecha_result = LLMExtractionResult(
            value=fecha_value,
            normalized=_normalizar_fecha(fecha_value),
            confidence=float(fecha_data.get("confidence", 0.0)),
            reasoning=fecha_data.get("reasoning", ""),
        )

        return {"monto": monto_result, "fecha_emision": fecha_result}
