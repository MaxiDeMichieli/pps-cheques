"""Abstraccion de lectores OCR.

Define una interfaz comun para diferentes librerias OCR (docTR, Tesseract, EasyOCR, etc.)
permitiendo intercambiarlas sin modificar la logica de extraccion de montos.
"""

from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np


class OCRResult:
    """Resultado de una palabra detectada por OCR."""
    __slots__ = ('text', 'confidence', 'cx', 'cy', 'height')

    def __init__(self, text: str, confidence: float, cx: float, cy: float, height: float = 0.0):
        self.text = text
        self.confidence = confidence
        self.cx = cx      # Centro X normalizado (0-1)
        self.cy = cy      # Centro Y normalizado (0-1)
        self.height = height  # Alto del bounding box normalizado (0-1)

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
                        height = geo[1][1] - geo[0][1]
                        palabras.append(OCRResult(word.value, word.confidence, cx, cy, height))
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
                palabras.append(OCRResult(text, conf, x / w, y / h, data['height'][i] / h))
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
            height = (bbox[2][1] - bbox[0][1]) / h
            palabras.append(OCRResult(text, conf, cx, cy, height))
        return palabras


class TrOCRReader(OCRReader):
    """Lector OCR usando Microsoft TrOCR (solo reconocimiento, sin deteccion).

    Pensado para usarse sobre crops ya recortados (ej. la linea de fecha de
    emision), no sobre imagenes completas, ya que no devuelve bounding boxes.
    Cada palabra del texto reconocido se devuelve como un OCRResult con
    posicion horizontal aproximada y cy fijo en 0.5.
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self._processor = TrOCRProcessor.from_pretrained(model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def read(self, img: np.ndarray) -> list[OCRResult]:
        import torch
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(img).convert("RGB")
        pixel_values = self._processor(images=pil_img, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        words = text.split()
        if not words:
            return []
        n = len(words)
        return [
            OCRResult(word, 1.0, (i + 0.5) / n, 0.5, 0.1)
            for i, word in enumerate(words)
        ]


class SuryaReader(OCRReader):
    """Lector OCR usando Surya (deteccion + reconocimiento multilingue).

    Reemplaza completamente a DocTRReader: hace deteccion y reconocimiento
    en una sola pasada, soporta mas de 90 idiomas incluyendo espanol, y
    devuelve resultados a nivel de palabra con bounding boxes normalizados.
    """

    def __init__(self):
        from surya.recognition import RecognitionPredictor, FoundationPredictor
        from surya.detection import DetectionPredictor
        foundation = FoundationPredictor()
        self._rec = RecognitionPredictor(foundation)
        self._det = DetectionPredictor()

    def read(self, img: np.ndarray) -> list[OCRResult]:
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(img).convert("RGB")
        w, h = pil_img.size

        results = self._rec(
            [pil_img],
            det_predictor=self._det,
            return_words=True,
            math_mode=False,
        )

        palabras = []
        for line in results[0].text_lines:
            words = line.words or []
            if words:
                for word in words:
                    poly = word.polygon
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    cx = (min(xs) + max(xs)) / 2 / w
                    cy = (min(ys) + max(ys)) / 2 / h
                    height = (max(ys) - min(ys)) / h
                    palabras.append(OCRResult(word.text, word.confidence or 1.0, cx, cy, height))
            elif line.text.strip():
                # sin palabras individuales: usar la linea completa con posicion del poligono
                poly = line.polygon
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                cx = (min(xs) + max(xs)) / 2 / w
                cy = (min(ys) + max(ys)) / 2 / h
                height = (max(ys) - min(ys)) / h
                palabras.append(OCRResult(line.text, line.confidence or 1.0, cx, cy, height))
        return palabras
