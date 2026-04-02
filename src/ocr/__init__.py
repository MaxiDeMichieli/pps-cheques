"""OCR abstraction layer."""

from .interfaces import OCRReader, OCRResult
from .doctr_reader import DocTRReader

__all__ = ['OCRReader', 'OCRResult', 'DocTRReader']