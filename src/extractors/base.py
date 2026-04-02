"""Base classes for field extractors (Template Method Pattern)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

from ..ocr.interfaces import OCRReader


class FieldExtractor(ABC):
    """Base class for all field extractors using Template Method pattern.

    Defines the standard extraction pipeline:
    1. Extract raw data from image region
    2. Parse OCR output to structured data
    3. Validate against business rules
    4. Normalize to final format
    """

    def __init__(self, ocr_reader: OCRReader):
        self.ocr_reader = ocr_reader

    def extract(self, check_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Template method: complete extraction pipeline.

        Args:
            check_image: RGB numpy array of the check

        Returns:
            Dict with extracted field data, or None if extraction failed
        """
        try:
            # Step 1: Extract raw OCR data from specific region
            raw_result = self._extract_raw(check_image)

            # Step 2: Parse the OCR output
            parsed = self._parse(raw_result)

            # Step 3: Validate the parsed data
            if not self._validate(parsed):
                return None

            # Step 4: Normalize to final format
            return self._normalize(parsed)

        except Exception as e:
            # Log error but don't crash the pipeline
            print(f"Error extracting {self.__class__.__name__}: {e}")
            return None

    @abstractmethod
    def _extract_raw(self, check_image: np.ndarray) -> Any:
        """Extract raw OCR data from specific region of the check.

        Override this to define region selection logic.
        """
        pass

    @abstractmethod
    def _parse(self, raw_data: Any) -> Dict[str, Any]:
        """Parse OCR output to structured data.

        Override this to define parsing logic.
        """
        pass

    @abstractmethod
    def _validate(self, parsed_data: Dict[str, Any]) -> bool:
        """Validate parsed data against business rules.

        Override this to define validation logic.
        """
        pass

    @abstractmethod
    def _normalize(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return final normalized result.

        Override this to define output format.
        """
        pass

    @property
    @abstractmethod
    def field_name(self) -> str:
        """Return the name of the field this extractor handles."""
        pass