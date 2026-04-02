"""Amount validation rules."""

from typing import Dict, Any, List
import re


class AmountValidator:
    """Validates extracted amounts against business rules."""

    def __init__(self):
        # Configurable validation rules
        self.min_amount = 0.01
        self.max_amount = 99999999.99  # 8 digits before decimal

    def is_valid(self, parsed_data: Dict[str, Any]) -> bool:
        """Validate parsed amount data.

        Args:
            parsed_data: Result from AmountParser.parse()

        Returns:
            True if valid, False otherwise
        """
        if not parsed_data.get('best_match'):
            return False

        candidate = parsed_data['best_match']

        # Check if we can convert to float
        try:
            amount_float = float(candidate['amount_text']) if isinstance(candidate['amount_text'], str) else candidate['amount_text']
        except (ValueError, TypeError):
            return False

        # Business rule validations
        if not self._validate_amount_range(amount_float):
            return False

        if not self._validate_format(candidate['amount_text']):
            return False

        if not self._validate_confidence(candidate['score']):
            return False

        return True

    def _validate_amount_range(self, amount: float) -> bool:
        """Check if amount is within acceptable range."""
        return self.min_amount <= amount <= self.max_amount

    def _validate_format(self, amount_text: str) -> bool:
        """Validate amount format."""
        if not amount_text:
            return False

        # Allow Argentine format: dots for thousands, comma for decimals
        # Or plain numbers
        pattern = r'^\d{1,4}(\.\d{3})*(,\d{1,2})?$|^\d+$'
        return bool(re.match(pattern, str(amount_text)))

    def _validate_confidence(self, score: float) -> bool:
        """Validate confidence score."""
        return score >= 0.0  # Allow any positive score

    def _validate_format(self, amount_text: str) -> bool:
        """Check if amount format is valid."""
        # Must have exactly 2 decimal places
        if not re.match(r'^\d+(?:\.\d{2})?$', amount_text):
            return False

        # Check for reasonable number of digits
        digits_only = re.sub(r'[^\d]', '', amount_text)
        return 3 <= len(digits_only) <= 12  # Allow 1-10 digits before decimal

    def _validate_confidence(self, score: float) -> bool:
        """Check if extraction confidence is acceptable."""
        return score >= 5.0  # Minimum acceptable score

    def get_validation_errors(self, parsed_data: Dict[str, Any]) -> List[str]:
        """Get detailed validation error messages."""
        errors = []

        if not parsed_data.get('best_match'):
            errors.append("No amount candidate found")
            return errors

        candidate = parsed_data['best_match']

        try:
            amount_float = float(candidate['amount_text'])
        except (ValueError, TypeError):
            errors.append("Cannot convert to numeric value")
            return errors

        if not self._validate_amount_range(amount_float):
            errors.append(f"Amount {amount_float} outside valid range [{self.min_amount}, {self.max_amount}]")

        if not self._validate_format(candidate['amount_text']):
            errors.append(f"Invalid amount format: {candidate['amount_text']}")

        if not self._validate_confidence(candidate['score']):
            errors.append(f"Low confidence score: {candidate['score']:.1f} (minimum: 5.0)")

        return errors