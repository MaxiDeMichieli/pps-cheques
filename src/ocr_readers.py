"""Abstraccion de lectores OCR.

Define una interfaz comun para diferentes librerias OCR (docTR, Tesseract, EasyOCR, etc.)
permitiendo intercambiarlas sin modificar la logica de extraccion de montos.
"""

from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np


class OCRResult:
    """Resultado de una palabra detectada por OCR."""
    __slots__ = ('text', 'confidence', 'cx', 'cy')

    def __init__(self, text: str, confidence: float, cx: float, cy: float):
        self.text = text
        self.confidence = confidence
        self.cx = cx  # Centro X normalizado (0-1)
        self.cy = cy  # Centro Y normalizado (0-1)

    def __iter__(self):
        """Permite desempaquetar como tupla: text, conf, cx, cy = result"""
        return iter((self.text, self.confidence, self.cx, self.cy))


class OCRReader(ABC):
    """Interfaz base para lectores OCR."""

    @abstractmethod
    def read(self, img: np.ndarray) -> list[OCRResult]:
        """Lee texto de una imagen.

        Args:
            img: Imagen RGB como numpy array.

        Returns:
            Lista de OCRResult con texto detectado y posiciones normalizadas.
        """
        pass


class DocTRReader(OCRReader):
    """Implementacion OCR usando docTR."""

    def __init__(self, model=None):
        """
        Args:
            model: Modelo docTR ya inicializado, o None para cargar el default.
        """
        if model is None:
            from doctr.models import ocr_predictor
            model = ocr_predictor(
                det_arch='db_resnet50',
                reco_arch='crnn_vgg16_bn',
                pretrained=True
            )
        self._model = model

    def read(self, img: np.ndarray) -> list[OCRResult]:
        result = self._model([img])
        palabras = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        geo = word.geometry
                        cx = (geo[0][0] + geo[1][0]) / 2
                        cy = (geo[0][1] + geo[1][1]) / 2
                        palabras.append(OCRResult(word.value, word.confidence, cx, cy))
        return palabras


# ---- Implementaciones alternativas (ejemplos) ----

class TesseractReader(OCRReader):
    """Implementacion OCR usando pytesseract (ejemplo)."""

    def __init__(self, lang: str = 'spa'):
        self.lang = lang

    def read(self, img: np.ndarray) -> list[OCRResult]:
        import pytesseract
        data = pytesseract.image_to_data(img, lang=self.lang, output_type=pytesseract.Output.DICT)
        h, w = img.shape[:2]
        palabras = []
        for i, text in enumerate(data['text']):
            if text.strip():
                conf = float(data['conf'][i]) / 100.0 if data['conf'][i] != -1 else 0.0
                x = data['left'][i] + data['width'][i] / 2
                y = data['top'][i] + data['height'][i] / 2
                palabras.append(OCRResult(text, conf, x / w, y / h))
        return palabras


class EasyOCRReader(OCRReader):
    """Implementacion OCR usando EasyOCR (ejemplo)."""

    def __init__(self, langs: list[str] = None):
        import easyocr
        self._reader = easyocr.Reader(langs or ['es', 'en'])

    def read(self, img: np.ndarray) -> list[OCRResult]:
        results = self._reader.readtext(img)
        h, w = img.shape[:2]
        palabras = []
        for bbox, text, conf in results:
            # bbox es [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            cx = sum(p[0] for p in bbox) / 4 / w
            cy = sum(p[1] for p in bbox) / 4 / h
            palabras.append(OCRResult(text, conf, cx, cy))
        return palabras
