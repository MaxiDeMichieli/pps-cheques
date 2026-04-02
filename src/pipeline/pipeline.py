"""Check processing pipeline."""

from typing import List, Dict, Any
import numpy as np

from ..extractors.base import FieldExtractor


class CheckProcessingResult:
    """Result of processing a single check."""

    def __init__(self, check_image_path: str = None):
        self.check_image_path = check_image_path
        self.fields = {}  # field_name -> extracted_data
        self.success = True
        self.errors = []  # List of (field_name, error_message)

    def add_field_result(self, field_name: str, data: Dict[str, Any]):
        """Add successful field extraction result."""
        self.fields[field_name] = data

    def add_error(self, field_name: str, error: str):
        """Add extraction error."""
        self.errors.append((field_name, error))
        self.success = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'success': self.success,
            'fields': self.fields,
            'errors': self.errors,
            'check_image_path': self.check_image_path
        }


class CheckProcessingPipeline:
    """Orchestrates field extraction from check images.

    Uses the Pipeline pattern to coordinate multiple field extractors.
    """

    def __init__(self, extractors: List[FieldExtractor]):
        """Initialize pipeline with field extractors.

        Args:
            extractors: List of FieldExtractor instances to run
        """
        self.extractors = extractors

    def process_check(self, check_image: np.ndarray, check_path: str = None) -> CheckProcessingResult:
        """Process a single check image through all extractors.

        Args:
            check_image: RGB numpy array of the check
            check_path: Optional path to the check image file

        Returns:
            CheckProcessingResult with all extracted fields
        """
        result = CheckProcessingResult(check_path)

        for extractor in self.extractors:
            field_name = extractor.field_name

            try:
                field_data = extractor.extract(check_image)
                if field_data is not None:
                    result.add_field_result(field_name, field_data)
                else:
                    result.add_error(field_name, "Extraction returned None")

            except Exception as e:
                result.add_error(field_name, f"Unexpected error: {str(e)}")

        return result

    def process_batch(self, check_images: List[np.ndarray], check_paths: List[str] = None) -> List[CheckProcessingResult]:
        """Process multiple check images.

        Args:
            check_images: List of RGB numpy arrays
            check_paths: Optional list of paths corresponding to images

        Returns:
            List of CheckProcessingResult objects
        """
        results = []

        for i, image in enumerate(check_images):
            path = check_paths[i] if check_paths and i < len(check_paths) else None
            result = self.process_check(image, path)
            results.append(result)

        return results