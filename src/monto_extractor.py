"""Extractor de monto de cheques escaneados.

Usa un lector OCR abstracto para leer el monto numerico de la esquina superior derecha.
Estrategia simple:
1. Buscar el $ con OCR en la zona superior
2. Recortar alrededor del $ y leer con OCR (crudo, otsu, x2)
3. Si no hay buen resultado, probar zonas fijas
4. Elegir el candidato con mejor formato de monto argentino
5. Normalizar a float

El lector OCR se inyecta via OCRReader, permitiendo usar docTR, Tesseract, EasyOCR, etc.
"""

import cv2
import numpy as np
import re
from dataclasses import dataclass

from .ocr_readers import OCRReader, OCRResult


@dataclass
class MontoOCRResult:
    """Resultado del paso OCR de extraccion de monto."""
    monto: float | None
    monto_raw: str
    monto_score: float
    zona_tokens: list[OCRResult]  # tokens de zona_sup, para uso del validador LLM


class MontoExtractor:
    """Extrae el monto numerico de un cheque escaneado via OCR heuristico."""

    def __init__(self, ocr_reader: OCRReader):
        self._ocr = ocr_reader

    def extraer(self, cheque_img: np.ndarray) -> MontoOCRResult:
        """Extrae el monto de un cheque usando OCR + heuristicas.

        Args:
            cheque_img: Imagen RGB del cheque recortado.

        Returns:
            MontoOCRResult con monto normalizado, raw, score y tokens de zona.
        """
        h, w = cheque_img.shape[:2]
        candidatos = []

        # ---- Paso 1: Encontrar $ y recortar alrededor ----
        zona_sup = cheque_img[0:int(h * 0.40), int(w * 0.40):w]
        zona_sup_tokens = self._ocr.read(zona_sup)
        textos_sup = [(r.text, r.confidence, r.cx, r.cy) for r in zona_sup_tokens]
        dolar_pos = self._encontrar_dolar(textos_sup)

        if dolar_pos:
            cx, cy = dolar_pos
            zh, zw = zona_sup.shape[:2]
            x_abs = int(cx * zw)
            y_abs = int(cy * zh)

            margen_y = int(zh * 0.30)
            margen_x_izq = int(zw * 0.10)
            crop = zona_sup[
                max(0, y_abs - margen_y):min(zh, y_abs + margen_y),
                max(0, x_abs - margen_x_izq):
            ]

            if crop.size > 0:
                for prep_fn in [self._noop, self._otsu, self._x2_otsu]:
                    img = prep_fn(crop)
                    for txt in self._extraer_montos(self._ocr_read(img), cerca_dolar=True):
                        candidatos.append((txt, True))

        # ---- Paso 2: Zonas fijas (siempre, complementa el paso 1) ----
        for x_pct, y_fin in [(0.63, 0.25), (0.58, 0.32), (0.50, 0.40)]:
            zona = cheque_img[0:int(h * y_fin), int(w * x_pct):w]
            for prep_fn in [self._noop, self._otsu]:
                img = prep_fn(zona)
                for txt in self._extraer_montos(self._ocr_read(img), cerca_dolar=False):
                    candidatos.append((txt, False))
            if any(self._score(t, d) >= 5.0 for t, d in candidatos):
                break

        # ---- Elegir mejor candidato ----
        if candidatos:
            candidatos.sort(key=lambda c: self._score(c[0], c[1]), reverse=True)
            mejor_txt, mejor_dolar = candidatos[0]
            ocr_score = self._score(mejor_txt, mejor_dolar)
            ocr_valor = self._normalizar(mejor_txt)
        else:
            mejor_txt, ocr_score, ocr_valor = "", -1, None

        return MontoOCRResult(
            monto=ocr_valor,
            monto_raw=mejor_txt,
            monto_score=ocr_score,
            zona_tokens=zona_sup_tokens,
        )

    # ---- OCR ----

    def _ocr_read(self, img):
        """Lee texto de imagen usando el reader inyectado."""
        return [(r.text, r.confidence, r.cx, r.cy) for r in self._ocr.read(img)]

    @staticmethod
    def _encontrar_dolar(textos):
        for txt, conf, cx, cy in textos:
            if '$' in txt:
                return cx, cy
        return None

    # ---- Preprocesamiento ----

    @staticmethod
    def _noop(img):
        return img

    @staticmethod
    def _otsu(img):
        gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, b = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _x2_otsu(img):
        esc = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gris = cv2.cvtColor(esc, cv2.COLOR_RGB2GRAY)
        _, b = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)

    # ---- Extraccion de candidatos ----

    def _extraer_montos(self, textos, cerca_dolar=False):
        """Extrae candidatos a monto de textos OCR."""
        dolar_cy = None
        for txt, conf, cx, cy in textos:
            if '$' in txt:
                dolar_cy = cy
                break

        candidatos = []
        for txt, conf, cx, cy in textos:
            limpio = txt.rstrip('-').rstrip(',').strip()
            # Limpiar $ o S al inicio
            limpio = re.sub(r'^[\$Ss]\s*[\[\(]?', '', limpio)
            limpio = limpio.rstrip('-').rstrip(',').strip()
            if not limpio:
                continue

            # Formato con puntos de miles (X.XXX.XXX)
            if re.match(r'^\d{1,4}(\.\d{3})+(,\d{1,2})?$', limpio):
                candidatos.append(limpio)
                continue

            # Numero puro 6+ digitos, solo si cerca del $
            if re.match(r'^\d{6,}$', limpio):
                if cerca_dolar or (dolar_cy is not None and abs(cy - dolar_cy) < 0.25):
                    candidatos.append(limpio)
                continue

            # Monto pegado al $
            m = re.search(r'[\$]\s*[\[\(]?([\d.,]+)', txt)
            if m:
                monto = m.group(1).rstrip('-').rstrip(',')
                digitos = re.sub(r'[^0-9]', '', monto)
                if len(digitos) >= 3:
                    candidatos.append(monto)

        return candidatos

    # ---- Scoring ----

    @staticmethod
    def _score(txt, cerca_dolar=False):
        if not txt:
            return -1
        digitos = re.sub(r'[^0-9]', '', txt)
        score = len(digitos) * 0.05

        # Formato con puntos de miles = alta confianza
        if re.match(r'^\d{1,4}(\.\d{3})+(,\d{1,2})?$', txt):
            score += 5.0
        elif re.match(r'^\d{6,}$', txt):
            score += 1.0
            if len(digitos) >= 9:
                score -= 2.0
            elif len(digitos) == 8:
                score -= 0.8  # 8 digitos = tipico nro de cheque
            elif len(digitos) == 7 and not cerca_dolar:
                score -= 0.3
        return score

    # ---- Normalizacion ----

    @staticmethod
    def _normalizar(txt):
        """Convierte monto argentino a float.

        Formato argentino: puntos = miles, coma = decimales.
        '2.500.000' -> 2500000.0
        '802.470,20' -> 802470.20
        '4000000' -> 4000000.0
        """
        if not txt:
            return None
        limpio = txt.rstrip('-').rstrip(',').strip()

        if ',' in limpio:
            partes = limpio.split(',')
            entero = partes[0].replace('.', '')
            decimal = partes[1] if len(partes) > 1 else '0'
            return float(f"{entero}.{decimal}")

        return float(limpio.replace('.', ''))
