"""Extractor de monto manuscrito del campo 'LA CANTIDAD DE PESOS' en cheques.

Realiza los siguientes pasos:
1. Detecta el label "LA CANTIDAD DE PESOS" (case-insensitive)
2. Recorta la zona de renglones manuscritos alrededor del label
3. Lee el texto manuscrito con OCR
4. Normaliza palabras a números usando text2number
5. Compara con monto numérico y calcula confianza
6. Retorna resultado serializable a JSON
"""

import cv2
import numpy as np
import base64
import re
from dataclasses import dataclass
from typing import Optional

from ..ocr.ocr_readers import OCRReader, OCRResult


@dataclass
class ManuscritoOCRResult:
    """Resultado del paso OCR de extraccion de monto manuscrito."""
    monto_manuscrito: Optional[float] = None
    monto_manuscrito_raw: Optional[str] = None
    monto_manuscrito_score: float = 2.0
    monto_manuscrito_confidence_ocr: float = 0.0
    monto_inconsistencia_pct: Optional[float] = None
    monto_manuscrito_zona_base64: Optional[str] = None
    validacion_alineada: bool = False
    zona_tokens: list = None  # Para debugging

    def __post_init__(self):
        if self.zona_tokens is None:
            self.zona_tokens = []


class ManuscritoExtractor:
    """Extrae y valida el monto manuscrito del campo 'LA CANTIDAD DE PESOS'."""

    def __init__(self, ocr_reader: OCRReader):
        """Inicializa con un lector OCR.

        Args:
            ocr_reader: Implementación de OCRReader (DocTR, Tesseract, etc.)
        """
        self._ocr = ocr_reader

    def extraer(
        self,
        cheque_img: np.ndarray,
        monto_numerico: Optional[float] = None,
    ) -> ManuscritoOCRResult:
        """Extrae monto manuscrito de un cheque.

        Args:
            cheque_img: Imagen RGB del cheque recortado.
            monto_numerico: Monto numérico del cheque (para validación cruzada).

        Returns:
            ManuscritoOCRResult con campos manuscritos, confianza y validación.
        """
        resultado = ManuscritoOCRResult()

        # ---- Paso 1: Detectar "LA CANTIDAD DE PESOS" ----
        zona_busqueda = self._definir_zona_busqueda(cheque_img)
        tokens_zona = self._ocr_read(zona_busqueda)
        resultado.zona_tokens = tokens_zona

        label_pos = self._encontrar_label(tokens_zona)
        if not label_pos:
            return resultado

        # ---- Paso 2: Recortar área de manuscrito ----
        crop = self._recortar_zona_manuscrito(cheque_img, zona_busqueda, label_pos)
        if crop is None or crop.size == 0:
            return resultado

        resultado.monto_manuscrito_zona_base64 = self._imagen_a_base64(crop)

        # ---- Paso 3: Leer OCR del manuscrito ----
        texto_manuscrito, confianza_ocr = self._leer_manuscrito(crop)
        if not texto_manuscrito:
            resultado.monto_manuscrito_score = 2.0
            return resultado

        resultado.monto_manuscrito_raw = texto_manuscrito
        resultado.monto_manuscrito_confidence_ocr = confianza_ocr

        # ---- Paso 4: Normalizar texto a número ----
        try:
            monto_normalizado = self._normalizar_texto_a_numero(texto_manuscrito)
        except Exception:
            monto_normalizado = None

        if monto_normalizado is None:
            resultado.monto_manuscrito_score = 4.0
            return resultado

        resultado.monto_manuscrito = monto_normalizado

        # ---- Paso 5: Comparar con monto numérico ----
        if monto_numerico is not None:
            resultado.monto_inconsistencia_pct = self._calcular_inconsistencia(
                monto_normalizado, monto_numerico
            )
            score, validacion_alineada = self._calcular_confianza(
                resultado.monto_inconsistencia_pct,
                confianza_ocr,
            )
            resultado.monto_manuscrito_score = score
            resultado.validacion_alineada = validacion_alineada
        else:
            resultado.monto_manuscrito_score = min(9.5, confianza_ocr * 10)
            resultado.validacion_alineada = True if confianza_ocr > 0.8 else False

        return resultado

    # ---- Detección de zona y label ----

    @staticmethod
    def _definir_zona_busqueda(cheque_img):
        """Define zona de búsqueda para 'LA CANTIDAD DE PESOS'.

        Returns:
            Subimagen con el área donde buscar el label (15%-45% vertical, 10%-60% horizontal).
        """
        h, w = cheque_img.shape[:2]
        y_inicio = int(h * 0.25)
        y_fin = int(h * 0.55)
        x_fin = int(w * 0.50)
        return cheque_img[y_inicio:y_fin, 0:x_fin]

    def _encontrar_label(self, tokens: list) -> Optional[tuple]:
        """Busca el label 'LA CANTIDAD DE PESOS' en tokens OCR.

        Returns:
            Tupla (cx, cy) del label en coordenadas normalizadas, o None.
        """
        texto_acumulado = ""
        posiciones = []

        for token in tokens:
            texto_acumulado += token.text + " "
            posiciones.append((token.cx, token.cy))

        texto_upper = texto_acumulado.upper()
        if "LA CANTIDAD DE PESOS" in texto_upper:
            if posiciones:
                cx_promedio = sum(p[0] for p in posiciones) / len(posiciones)
                cy_promedio = sum(p[1] for p in posiciones) / len(posiciones)
                return cx_promedio, cy_promedio

        if all(x in texto_upper for x in ["CANTIDAD", "PESOS"]):
            if posiciones:
                cx_promedio = sum(p[0] for p in posiciones) / len(posiciones)
                cy_promedio = sum(p[1] for p in posiciones) / len(posiciones)
                return cx_promedio, cy_promedio

        return None

    @staticmethod
    def _recortar_zona_manuscrito(
        cheque_img: np.ndarray,
        zona_busqueda: np.ndarray,
        label_pos: tuple,
    ) -> Optional[np.ndarray]:
        """Recorta área de renglones manuscritos alrededor del label.

        Args:
            cheque_img: Imagen original del cheque.
            zona_busqueda: Subimagen donde se encontró el label.
            label_pos: Tupla (cx, cy) normalizado en zona_busqueda.

        Returns:
            Subimagen con los renglones manuscritos, o None.
        """
        h_busqueda, w_busqueda = zona_busqueda.shape[:2]
        cx, cy = label_pos

        h_cheque, w_cheque = cheque_img.shape[:2]
        y_zona_inicio = int(h_cheque * 0.25)
        x_zona_inicio = 0

        x_abs = int(x_zona_inicio + cx * w_busqueda)
        y_abs = int(y_zona_inicio + cy * h_busqueda)

        margen_y_arriba = int(h_busqueda * 0.20)
        margen_y_abajo = int(h_busqueda * 0.70)
        margen_x_izq = int(w_busqueda * 0.25)
        margen_x_der = int(w_busqueda * 0.88)

        y_inicio = max(0, y_abs - margen_y_arriba)
        y_fin = min(h_cheque, y_abs + margen_y_abajo)
        x_inicio = max(0, x_abs - margen_x_izq)
        x_fin = min(w_cheque, x_abs + margen_x_der)

        crop = cheque_img[y_inicio:y_fin, x_inicio:x_fin]
        return crop if crop.size > 0 else None

    # ---- OCR y lectura de manuscrito ----

    def _ocr_read(self, img: np.ndarray) -> list:
        """Lee tokens OCR de una imagen."""
        return self._ocr.read(img)

    def _leer_manuscrito(self, crop: np.ndarray) -> tuple:
        """Lee texto manuscrito de un crop de imagen.

        Prueba múltiples preprocesados (OTSU, escalado).

        Returns:
            Tupla (texto, confianza_promedio) o ("", 0.0) si no hay texto.
        """
        candidatos = []

        for prep_fn in [self._noop, self._otsu, self._x2_otsu]:
            img_prep = prep_fn(crop)
            tokens = self._ocr_read(img_prep)

            if tokens:
                texto = " ".join(t.text for t in tokens)
                confianza = np.mean([t.confidence for t in tokens]) if tokens else 0.0
                if texto.strip():
                    candidatos.append((texto.strip(), confianza))

        if not candidatos:
            return "", 0.0

        candidatos.sort(key=lambda x: x[1], reverse=True)
        return candidatos[0]

    @staticmethod
    def _noop(img: np.ndarray) -> np.ndarray:
        """Sin preprocesamiento."""
        return img

    @staticmethod
    def _otsu(img: np.ndarray) -> np.ndarray:
        """Binarización OTSU."""
        gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, b = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _x2_otsu(img: np.ndarray) -> np.ndarray:
        """Escalado x2 + OTSU."""
        esc = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gris = cv2.cvtColor(esc, cv2.COLOR_RGB2GRAY)
        _, b = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)

    # ---- Normalización texto -> número ----

    @staticmethod
    def _normalizar_texto_a_numero(texto: str) -> Optional[float]:
        if not texto or not texto.strip():
            return None

        texto = texto.strip().lower()

        try:
            from text2number import text2num
            resultado = text2num(texto, lang="es")
            if isinstance(resultado, (int, float)):
                return float(resultado)
        except Exception:
            pass

        numeros = re.findall(r"\d+", texto)
        if numeros:
            try:
                numero_puro = int("".join(numeros[:3]))
                return float(numero_puro)
            except Exception:
                pass

        match = re.search(r"(\d+)[\.,](\d+)", texto)
        if match:
            try:
                entero = match.group(1)
                decimal = match.group(2)
                return float(f"{entero}.{decimal}")
            except Exception:
                pass

        return None

    # ---- Validación y scoring ----

    @staticmethod
    def _calcular_inconsistencia(monto_manuscrito: float, monto_numerico: float) -> float:
        if monto_manuscrito == 0:
            return 0.0
        diferencia_pct = ((monto_numerico - monto_manuscrito) / monto_manuscrito) * 100
        return round(diferencia_pct, 2)

    @staticmethod
    def _calcular_confianza(
        inconsistencia_pct: Optional[float],
        confianza_ocr: float,
    ) -> tuple:
        if inconsistencia_pct is None:
            return 6.0, False

        abs_inconsistencia = abs(inconsistencia_pct)

        if abs_inconsistencia <= 10:
            score = 9.5 + (0.5 if confianza_ocr > 0.8 else 0)
            validacion_alineada = True
        elif abs_inconsistencia <= 20:
            score = 6.0 + (confianza_ocr * 2)
            validacion_alineada = False
        else:
            score = 2.0 + confianza_ocr
            validacion_alineada = False

        return min(10.0, score), validacion_alineada

    @staticmethod
    def _imagen_a_base64(img: np.ndarray) -> str:
        _, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode("utf-8")