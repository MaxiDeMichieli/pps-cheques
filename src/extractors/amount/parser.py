"""Amount parsing logic."""

import re
from typing import Dict, Any, List


class AmountParser:
    """Parses OCR results to extract monetary amounts."""

    def parse(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse OCR results to find amount candidates.

        Args:
            ocr_results: List of OCR result dictionaries

        Returns:
            Dict with 'candidates' list and 'best_match' info
        """
        # The actual parsing is now done in the extractor
        # This method is kept for interface consistency
        return {
            'candidates': ocr_results,
            'best_match': ocr_results[0] if ocr_results else None
        }

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x['score'], reverse=True)

        return {
            'candidates': candidates,
            'best_match': candidates[0] if candidates else None
        }

    def _clean_amount_string(self, amount_str: str) -> str:
        """Clean and standardize amount string."""
        # Remove currency symbols and extra spaces
        cleaned = re.sub(r'[^\d,.\s]', '', amount_str).strip()

        # Handle different decimal separators
        if ',' in cleaned and '.' in cleaned:
            # If both separators exist, assume European format (1.234,56)
            if cleaned.rfind(',') > cleaned.rfind('.'):
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                # American format (1,234.56)
                cleaned = cleaned.replace(',', '')

        return cleaned

    def _calculate_score(self, amount_text: str, ocr_confidence: float) -> float:
        """Calculate overall score for amount candidate."""
        score = ocr_confidence * 10  # Base score from OCR confidence

        # Bonus for proper decimal places
        if re.search(r'\.\d{2}$', amount_text):
            score += 2

        # Bonus for reasonable length
        if 3 <= len(re.sub(r'[^\d]', '', amount_text)) <= 12:
            score += 1

        # Penalty for very short amounts (likely false positives)
        if len(re.sub(r'[^\d]', '', amount_text)) < 3:
            score -= 3

        return max(0, score)