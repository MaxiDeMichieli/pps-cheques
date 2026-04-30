"""Extractor de sucursal de cheques escaneados.

Usa un lector OCR abstracto para leer el código de sucursal.
Estrategia simple:
1. Buscar las anclas "SUC", "SUCURSAL" o "FILIAL" con OCR en la zona del 40% inferior
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


_DEBUG_SUCURSAL_ZONA = "sucursal_zona.png"


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

        # ---- Paso 1: Zona del 40% inferior para buscar ancla SUC/SUCURSAL/FILIAL ----
        zona_sucursal = cheque_img[int(h * 0.60):h, :]
        if debug_dir is not None:
            Image.fromarray(zona_sucursal).save(debug_dir / _DEBUG_SUCURSAL_ZONA)
        _save(zona_sucursal, "sucursal_zona_completa.png")
        zona_tokens = self._ocr.read(zona_sucursal)
        textos = [(r.text, r.confidence, r.cx, r.cy) for r in zona_tokens]

        ancla_pos = self._encontrar_ancla(textos)
        candidatos_ancla = []

        if ancla_pos:
            cx, cy = ancla_pos
            zh, zw = zona_sucursal.shape[:2]
            x_abs = int(cx * zw)
            y_abs = int(cy * zh)

            offset_x = int(zw * 0.01) # pequeño margen para recortar parte del ancla: "SUC" o "SUCURSAL"
            x0 = min(zw, x_abs + offset_x)
            margen_y = int(zh * 0.05)
            margen_x_der = int(zw * 0.40)  # recortar hasta 40% del ancho a la derecha de la ancla, buscando el nombre de ciudad
            crop = zona_sucursal[
                max(0, y_abs - margen_y):min(zh, y_abs + margen_y),
                x0:min(zw, x0 + margen_x_der),
            ]

            if crop.size > 0:
                if debug_dir is not None:
                    Image.fromarray(crop).save(debug_dir / "sucursal_crop.png")
                for prep_fn, label in [
                    (self._noop, "sucursal_ancla_crop_raw.png"),
                    (self._otsu, "sucursal_ancla_crop_otsu.png"),
                    (self._x2_otsu, "sucursal_ancla_crop_x2otsu.png"),
                ]:
                    img = prep_fn(crop)
                    _save(img, label)
                    for txt in self._extraer_sucursales(self._ocr_read(img), cerca_ancla=True):
                        candidatos_ancla.append(txt)

        # ---- Paso 2: Zonas fijas (siempre, complementa el paso 1) ----
        candidatos_fijos = []
        for i, (x_inicio, x_fin) in enumerate([(0.35, 0.65), (0.30, 0.70), (0.20, 0.80)], 1):
            zona = cheque_img[int(h * 0.60):h, int(w * x_inicio):int(w * x_fin)]
            _save(zona, f"sucursal_fixed_z{i}.png")
            for prep_fn, label in [
                (self._noop, f"sucursal_fixed_z{i}_raw.png"),
                (self._otsu, f"sucursal_fixed_z{i}_otsu.png"),
            ]:
                img = prep_fn(zona)
                _save(img, label)
                for txt in self._extraer_sucursales(self._ocr_read(img), cerca_ancla=False):
                    candidatos_fijos.append(txt)
            if any(self._score(t, False) >= 3.0 for t in candidatos_fijos):
                break

        # ---- Elegir mejor candidato ----
        # Priorizar candidatos de la ancla si tienen score decente (>= 3.0)
        if candidatos_ancla:
            candidatos_ancla.sort(key=lambda t: self._score(t, True), reverse=True)
            mejor_ancla = candidatos_ancla[0]
            score_ancla = self._score(mejor_ancla, True)
            if score_ancla >= 3.0:
                mejor_txt, ocr_score, ocr_valor = mejor_ancla, score_ancla, self._normalizar(mejor_ancla)
            else:
                # Usar fijos si ancla no es buena
                if candidatos_fijos:
                    candidatos_fijos.sort(key=lambda t: self._score(t, False), reverse=True)
                    mejor_fijo = candidatos_fijos[0]
                    mejor_txt, ocr_score, ocr_valor = mejor_fijo, self._score(mejor_fijo, False), self._normalizar(mejor_fijo)
                else:
                    mejor_txt, ocr_score, ocr_valor = mejor_ancla, score_ancla, self._normalizar(mejor_ancla)
        elif candidatos_fijos:
            candidatos_fijos.sort(key=lambda t: self._score(t, False), reverse=True)
            mejor_fijo = candidatos_fijos[0]
            mejor_txt, ocr_score, ocr_valor = mejor_fijo, self._score(mejor_fijo, False), self._normalizar(mejor_fijo)
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
        """Busca 'SUC', 'SUCURSAL' o 'FILIAL' en los textos OCR."""
        for txt, conf, cx, cy in textos:
            if re.search(r'\b(suc(ursal)?|filial)\b', txt, re.IGNORECASE):
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
        """Extrae candidatos a sucursal (nombres de ciudades) de textos OCR."""
        candidatos = []
        ancla_cx = None

        # Encontrar posicion de la ancla
        for txt, conf, cx, cy in textos:
            if re.search(r'\b(suc(ursal)?|filial)\b', txt, re.IGNORECASE):
                ancla_cx = cx
                break

        # Buscar nombres de ciudades despues de la ancla
        for txt, conf, cx, cy in textos:
            if ancla_cx is not None and cx <= ancla_cx:
                continue

            limpio = txt.strip()
            if not limpio:
                continue

            # Descartar si es principalmente numerico
            digitos = re.sub(r'[^0-9]', '', limpio)
            if len(digitos) >= len(limpio) * 0.7:  # Mas de 70% digitos
                continue

            # Nombre de ciudad: al menos 3 letras, puede tener espacios/guiones
            if re.search(r'[a-zA-Z]', limpio) and len(limpio) >= 3:
                candidatos.append(limpio)

        return candidatos

    # ---- Scoring ----

    @staticmethod
    def _score(txt, cerca_ancla=False):
        if not txt:
            return -1

        # Favor palabras más largas (típicamente nombres de ciudades)
        letras = re.sub(r'[^a-zA-Z]', '', txt)
        score = len(letras) * 0.3

        # Nombre de ciudad bien formado: 5+ letras
        if len(letras) >= 5:
            score += 2.0
        elif len(letras) >= 3:
            score += 0.5
        else:
            score -= 1.0  # Muy corto

        # Bonus si tiene mayuscula inicial (formato tipico de ciudad)
        if txt[0].isupper():
            score += 1.0

        # Bonus si esta cerca de la ancla
        if cerca_ancla:
            score += 1.0

        return score

    # ---- Normalizacion ----

    @staticmethod
    def _normalizar(txt):
        """Convierte sucursal a string limpio.

        Extrae el nombre de ciudad, elimina espacios/caracteres especiales innecesarios.
        """
        if not txt:
            return None

        limpio = txt.strip()

        # Descartar si no tiene letras
        if not re.search(r'[a-zA-Z]', limpio):
            return None

        # Sucursal es un nombre de ciudad (al menos 3 letras)
        letras = re.sub(r'[^a-zA-Z]', '', limpio)
        if len(letras) >= 3:
            return limpio

        return None
