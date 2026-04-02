"""docTR-based OCR implementation."""

import numpy as np
from typing import List
from .interfaces import OCRReader, OCRResult


class DocTRReader(OCRReader):
    """docTR-based OCR implementation."""

    def __init__(self, doctr_model):
        self.model = doctr_model

    def read(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using docTR model."""
        result = self.model([image])
        texts = []

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # Convert docTR geometry to normalized coordinates
                        geo = word.geometry
                        cx = (geo[0][0] + geo[1][0]) / 2
                        cy = (geo[0][1] + geo[1][1]) / 2

                        texts.append(OCRResult(
                            text=word.value,
                            confidence=word.confidence,
                            bbox=geo,
                            cx=cx,
                            cy=cy
                        ))

        return texts

    def get_name(self) -> str:
        return "docTR"