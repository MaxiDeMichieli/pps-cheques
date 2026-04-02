"""Date field extractor (supports multiple regions)."""

import numpy as np
from typing import Dict, Any, List, Tuple
from ..base import FieldExtractor
from .parser import DateParser
from .validators import DateValidator
from ...ocr.interfaces import OCRResult


class DateExtractor(FieldExtractor):
    """Extracts date from checks.

    Supports multiple date regions as required for Argentine checks:
    - Primary date (usually bottom right)
    - Secondary date (may be in header or other location)

    TODO: Define exact coordinates and regions for different check types
    TODO: Implement region detection logic
    """

    def __init__(self, ocr_reader):
        super().__init__(ocr_reader)
        self.parser = DateParser()
        self.validator = DateValidator()

        # TODO: Define specific regions for different check types
        # For now, using placeholder regions
        self.date_regions = {
            'primary': {
                'description': 'Bottom right area (main date)',
                'coords': (0.7, 0.8, 1.0, 1.0),  # x1, y1, x2, y2 as percentages
            },
            'secondary': {
                'description': 'Top right area (header date)',
                'coords': (0.7, 0.0, 1.0, 0.2),  # x1, y1, x2, y2 as percentages
            }
        }

    @property
    def field_name(self) -> str:
        return "fecha"

    def _extract_raw(self, check_image: np.ndarray) -> List[OCRResult]:
        """Extract date from multiple regions."""
        all_candidates = []

        # Try each defined region
        for region_name, region_config in self.date_regions.items():
            region_results = self._extract_from_region(check_image, region_config)

            # Tag results with region info
            for result in region_results:
                result.region = region_name
                all_candidates.append(result)

        return all_candidates

    def _extract_from_region(self, check_image: np.ndarray, region_config: Dict) -> List[OCRResult]:
        """Extract OCR results from a specific region."""
        h, w = check_image.shape[:2]
        x1, y1, x2, y2 = region_config['coords']

        # Convert percentages to pixel coordinates
        x1_px, x2_px = int(x1 * w), int(x2 * w)
        y1_px, y2_px = int(y1 * h), int(y2 * h)

        # Extract region
        region = check_image[y1_px:y2_px, x1_px:x2_px]

        if region.size == 0:
            return []

        # TODO: Apply region-specific preprocessing
        # For now, just read the region directly
        return self.ocr_reader.read(region)

    def _parse(self, raw_data: List[OCRResult]) -> Dict[str, Any]:
        """Parse OCR results using DateParser."""
        # Group by region for separate parsing
        primary_results = [r for r in raw_data if getattr(r, 'region', None) == 'primary']
        secondary_results = [r for r in raw_data if getattr(r, 'region', None) == 'secondary']

        results = {
            'primary': self.parser.parse(primary_results),
            'secondary': self.parser.parse(secondary_results),
        }

        # Choose best overall candidate
        all_candidates = []
        for region_name, region_result in results.items():
            if region_result.get('best_match'):
                candidate = region_result['best_match'].copy()
                candidate['region'] = region_name
                all_candidates.append(candidate)

        all_candidates.sort(key=lambda x: x['score'], reverse=True)

        results['candidates'] = all_candidates
        results['best_match'] = all_candidates[0] if all_candidates else None

        return results

    def _validate(self, parsed_data: Dict[str, Any]) -> bool:
        """Validate using DateValidator."""
        return self.validator.is_valid(parsed_data)

    def _normalize(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return normalized date data."""
        best_match = validated_data['best_match']

        return {
            'fecha': best_match['date'].isoformat() if best_match['date'] else None,
            'fecha_raw': best_match['raw_text'],
            'fecha_score': best_match['score'],
            'fecha_region': best_match.get('region', 'unknown')
        }