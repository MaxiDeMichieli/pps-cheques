"""Extractor de sucursal de cheques escaneados.

Usa un lector OCR abstracto para leer el código de sucursal.
Estrategia simple:
1. Buscar las anclas "SUC" o "SUCURSAL" con OCR en la zona del 20% superior
2. Recortar alrededor de la ancla y leer con OCR (crudo, otsu, x2)
3. Extraer el valor numerico despues de la ancla
4. Elegir el candidato con mejor formato
5. Normalizar a string/int

El lector OCR se inyecta via OCRReader, permitiendo usar docTR, Tesseract, EasyOCR, etc.
"""

import cv2
import numpy as np
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..ocr.ocr_readers import OCRReader, OCRResult


@dataclass
class SucursalOCRResult:
    """Resultado del paso OCR de extraccion de sucursal."""
    sucursal: str | None
    sucursal_raw: str
    sucursal_score: float
    zona_tokens: list[OCRResult]  # tokens de zona_sucursal, para uso del validador LLM


class SucursalExtractor:
    """Extrae el codigo de sucursal de un cheque escaneado via OCR heuristico."""

    def __init__(self, ocr_reader: OCRReader):
        self._ocr = ocr_reader

    def extraer(self, cheque_img: np.ndarray, debug_dir: Path | None = None) -> SucursalOCRResult:
        """Extrae la sucursal de un cheque usando OCR + heuristicas.

        Args:
            cheque_img: Imagen RGB del cheque recortado.
            debug_dir: Si se provee, guarda imagenes intermedias en ese directorio.

        Returns:
            SucursalOCRResult con sucursal normalizada, raw, score y tokens de zona.
        """
        h, w = cheque_img.shape[:2]
        candidatos = []

        def _save(img: np.ndarray, name: str):
            if debug_dir is not None:
                Image.fromarray(img).save(debug_dir / name)

        # ---- Paso 1: Zona del 20% superior para buscar ancla SUC/SUCURSAL ----
        zona_sucursal = cheque_img[0:int(h * 0.20), :]
        _save(zona_sucursal, "sucursal_zona_completa.png")
        zona_tokens = self._ocr.read(zona_sucursal)
        textos = [(r.text, r.confidence, r.cx, r.cy) for r in zona_tokens]

        ancla_pos = self._encontrar_ancla(textos)

        if ancla_pos:
            cx, cy = ancla_pos
            zh, zw = zona_sucursal.shape[:2]
            x_abs = int(cx * zw)
            y_abs = int(cy * zh)

            margen_y = int(zh * 0.30)
            margen_x_der = int(zw * 0.30)
            crop = zona_sucursal[
                max(0, y_abs - margen_y):min(zh, y_abs + margen_y),
                min(zw, x_abs):min(zw, x_abs + margen_x_der),
            ]

            if crop.size > 0:
                for prep_fn, label in [
                    (self._noop, "sucursal_ancla_crop_raw.png"),
                    (self._otsu, "sucursal_ancla_crop_otsu.png"),
                    (self._x2_otsu, "sucursal_ancla_crop_x2otsu.png"),
                ]:
                    img = prep_fn(crop)
                    _save(img, label)
                    for txt in self._extraer_sucursales(self._ocr_read(img), cerca_ancla=True):
                        candidatos.append((txt, True))

        # ---- Paso 2: Zonas fijas (siempre, complementa el paso 1) ----
        for i, (x_inicio, x_fin) in enumerate([(0.35, 0.65), (0.30, 0.70), (0.20, 0.80)], 1):
            zona = cheque_img[0:int(h * 0.20), int(w * x_inicio):int(w * x_fin)]
            _save(zona, f"sucursal_fixed_z{i}.png")
            for prep_fn in [self._noop, self._otsu]:
                img = prep_fn(zona)
                for txt in self._extraer_sucursales(self._ocr_read(img), cerca_ancla=False):
                    candidatos.append((txt, False))
            if any(self._score(t, d) >= 3.0 for t, d in candidatos):
                break

        # ---- Elegir mejor candidato ----
        if candidatos:
            candidatos.sort(key=lambda c: self._score(c[0], c[1]), reverse=True)
            mejor_txt, mejor_ancla = candidatos[0]
            ocr_score = self._score(mejor_txt, mejor_ancla)
            ocr_valor = self._normalizar(mejor_txt)
        else:
            mejor_txt, ocr_score, ocr_valor = "", -1, None

        return SucursalOCRResult(
            sucursal=ocr_valor,
            sucursal_raw=mejor_txt,
            sucursal_score=ocr_score,
            zona_tokens=zona_tokens,
        )

    # ---- OCR ----

    def _ocr_read(self, img):
        """Lee texto de imagen usando el reader inyectado."""
        return [(r.text, r.confidence, r.cx, r.cy) for r in self._ocr.read(img)]

    @staticmethod
    def _encontrar_ancla(textos):
        """Busca 'SUC' o 'SUCURSAL' en los textos OCR."""
        for txt, conf, cx, cy in textos:
            if re.search(r'\bsuc(ursal)?\b', txt, re.IGNORECASE):
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

    def _extraer_sucursales(self, textos, cerca_ancla=False):
        """Extrae candidatos a sucursal de textos OCR."""
        candidatos = []
        ancla_cx = None

        # Encontrar posicion de la ancla
        for txt, conf, cx, cy in textos:
            if re.search(r'\bsuc(ursal)?\b', txt, re.IGNORECASE):
                ancla_cx = cx
                break

        # Buscar numeros despues de la ancla
        for txt, conf, cx, cy in textos:
            if ancla_cx is not None and cx <= ancla_cx:
                continue

            limpio = txt.strip()
            if not limpio:
                continue

            # Numeros puros (codigo de sucursal tipicamente 3-5 digitos)
            if re.match(r'^\d{1,5}$', limpio):
                candidatos.append(limpio)
                continue

            # Extraer numero al inicio
            m = re.match(r'^(\d{1,5})', limpio)
            if m:
                candidatos.append(m.group(1))

        return candidatos

    # ---- Scoring ----

    @staticmethod
    def _score(txt, cerca_ancla=False):
        if not txt:
            return -1

        digitos = re.sub(r'[^0-9]', '', txt)
        score = len(digitos) * 0.5

        # Numero de 3-5 digitos = formato tipico de sucursal
        if re.match(r'^\d{3,5}$', txt):
            score += 3.0
        elif re.match(r'^\d{1,2}$', txt):
            score -= 1.0  # Muy pocos digitos
        elif len(digitos) > 5:
            score -= 2.0  # Demasiados digitos

        # Bonus si esta cerca de la ancla
        if cerca_ancla:
            score += 1.0

        return score

    # ---- Normalizacion ----

    @staticmethod
    def _normalizar(txt):
        """Convierte sucursal a string limpio.

        Extrae solo los digitos, valida longitud tipica (3-5 caracteres).
        """
        if not txt:
            return None

        limpio = re.sub(r'[^0-9]', '', txt).strip()

        if not limpio or len(limpio) == 0:
            return None

        # Sucursal tipicamente 3-5 digitos
        if 1 <= len(limpio) <= 5:
            return limpio

        return None
