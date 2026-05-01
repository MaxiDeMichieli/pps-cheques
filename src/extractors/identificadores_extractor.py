"""Extractor de identificadores del cheque (sucursal, numero de cheque, cuenta).

Estos tres datos aparecen en el recuadro derecho del cheque, en ese orden,
con un digito verificador al final de cada uno (que descartamos).

Como el texto del recuadro es molde y bien legible, este extractor depende
solo del OCR sin necesidad de validacion via LLM.

Estrategia:
1. Recortar el recuadro derecho del cheque.
2. OCR sobre la zona.
3. Filtrar tokens del borde izquierdo (ruido del cuerpo del cheque).
4. Agrupar tokens en filas por posicion vertical (cy).
5. Tomar las 3 primeras filas (sucursal, cheque, cuenta).
6. Para cada fila: concatenar digitos y descartar el verificador final.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..ocr.ocr_readers import OCRReader, OCRResult


# Longitudes esperadas de cada identificador SIN el digito verificador.
LARGOS_ESPERADOS = {
    "sucursal": 10,        # formato XXX-XXX-XXXX
    "numero_cheque": 8,
    "cuenta": 11,
}


@dataclass
class IdentificadoresResult:
    """Resultado de la extraccion del recuadro de identificadores."""
    sucursal: str | None
    numero_cheque: str | None
    cuenta: str | None
    zona_tokens: list[OCRResult]


class IdentificadoresExtractor:
    """Extrae sucursal, numero de cheque y cuenta del recuadro derecho."""

    def __init__(self, ocr_reader: OCRReader):
        self._ocr = ocr_reader

    def extraer(self, cheque_img: np.ndarray, debug_dir: Path | None = None) -> IdentificadoresResult:
        h, w = cheque_img.shape[:2]
        x0 = int(w * 0.79)
        y0 = int(h * 0.15)
        y1 = int(h * 0.65)
        zona = cheque_img[y0:y1, x0:w]

        if debug_dir is not None:
            Image.fromarray(zona).save(debug_dir / "identificadores_zona.png")

        tokens = self._ocr.read(zona)
        # Excluir ruido a la izquierda de la zona (columna central de cheque repetido,
        # anotaciones manuscritas, etc.)
        candidatos = [t for t in tokens if t.cx > 0.20]
        filas = self._agrupar_filas(candidatos, threshold=0.05)
        # Solo filas que tengan al menos un token con >= 4 digitos (descarta rumas
        # de tokens sueltos como dashes o caracteres aislados)
        filas = [f for f in filas if any(self._cuenta_digitos(t.text) >= 4 for t in f)]
        filas.sort(key=lambda f: sum(t.cy for t in f) / len(f))
        filas = filas[:3]

        campos = ["sucursal", "numero_cheque", "cuenta"]
        valores: dict[str, str | None] = {c: None for c in campos}
        for i, campo in enumerate(campos):
            if i < len(filas):
                valores[campo] = self._extraer_valor(filas[i], LARGOS_ESPERADOS[campo])

        return IdentificadoresResult(
            sucursal=valores["sucursal"],
            numero_cheque=valores["numero_cheque"],
            cuenta=valores["cuenta"],
            zona_tokens=tokens,
        )

    @staticmethod
    def _cuenta_digitos(text: str) -> int:
        return sum(1 for c in text if c.isdigit())

    @staticmethod
    def _agrupar_filas(tokens: list[OCRResult], threshold: float = 0.05) -> list[list[OCRResult]]:
        """Agrupa tokens por cy: tokens dentro de `threshold` del centro de la fila se agregan."""
        if not tokens:
            return []
        tokens_sorted = sorted(tokens, key=lambda t: t.cy)
        filas: list[list[OCRResult]] = [[tokens_sorted[0]]]
        for t in tokens_sorted[1:]:
            cy_fila = sum(x.cy for x in filas[-1]) / len(filas[-1])
            if abs(t.cy - cy_fila) < threshold:
                filas[-1].append(t)
            else:
                filas.append([t])
        return filas

    @staticmethod
    def _extraer_valor(fila: list[OCRResult], largo_esperado: int) -> str | None:
        """Concatena los tokens de la fila por cx, extrae digitos y descarta el verificador.

        Si el conteo de digitos supera al largo esperado, asumimos que el ultimo
        es el verificador y lo descartamos. Si coincide con el largo esperado,
        asumimos que el verificador no fue reconocido (ej: leido como letra) y
        devolvemos los digitos tal cual.
        """
        if not fila:
            return None
        fila_sorted = sorted(fila, key=lambda t: t.cx)
        texto = "".join(t.text for t in fila_sorted)
        digitos = re.sub(r"\D", "", texto)
        if len(digitos) > largo_esperado:
            digitos = digitos[:-1]
        if len(digitos) < 4:
            return None
        return digitos
