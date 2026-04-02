"""OCR abstraction layer interfaces and implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class OCRResult:
    """Standardized OCR result structure."""

    def __init__(self, text: str, confidence: float, bbox: tuple = None, cx: float = None, cy: float = None):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2) normalized coordinates
        self.cx = cx      # center x coordinate (normalized)
        self.cy = cy      # center y coordinate (normalized)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'cx': self.cx,
            'cy': self.cy
        }


class OCRReader(ABC):
    """Base interface for OCR implementations (Strategy Pattern)."""

    @abstractmethod
    def read(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text from image.

        Args:
            image: RGB numpy array

        Returns:
            List of OCRResult objects with text, confidence, and position data
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this OCR implementation."""
        pass