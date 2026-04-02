"""Date validation rules (placeholder implementation)."""

from datetime import datetime, date
from typing import Dict, Any, List
import re


class DateValidator:
    """Validates extracted dates against business rules.

    TODO: Define specific business rules for check dates:
    - Maximum age of check (e.g., not older than 1 year)
    - Cannot be future dates (beyond reasonable grace period)
    - Bank-specific date restrictions
    - Weekend/holiday validations
    """

    def __init__(self):
        # Configurable validation rules
        self.min_date = date(2020, 1, 1)  # Earliest acceptable date
        self.max_date = date.today()       # Latest acceptable date (today)
        self.max_age_days = 365           # Maximum check age in days

        # TODO: Add holiday calendar for Argentine bank holidays
        # TODO: Add weekend processing rules if needed

    def is_valid(self, parsed_data: Dict[str, Any]) -> bool:
        """Validate parsed date data.

        Args:
            parsed_data: Result from DateParser.parse()

        Returns:
            True if valid, False otherwise
        """
        if not parsed_data.get('best_match'):
            return False

        candidate = parsed_data['best_match']
        date_obj = candidate.get('date')

        if not isinstance(date_obj, date):
            return False

        # Business rule validations
        if not self._validate_date_range(date_obj):
            return False

        if not self._validate_date_format(candidate.get('raw_text', '')):
            return False

        if not self._validate_confidence(candidate.get('score', 0)):
            return False

        # TODO: Add weekend/holiday validation
        # TODO: Add bank-specific rules

        return True

    def _validate_date_range(self, date_obj: date) -> bool:
        """Check if date is within acceptable range."""
        today = date.today()

        # Cannot be too old
        if (today - date_obj).days > self.max_age_days:
            return False

        # Cannot be in the future (allow small grace period)
        if date_obj > today:
            return False

        # Within absolute bounds
        return self.min_date <= date_obj <= self.max_date

    def _validate_date_format(self, raw_text: str) -> bool:
        """Check if the raw text looks like a valid date format."""
        # Basic check - should contain numbers and separators
        if not re.search(r'\d', raw_text):
            return False

        # Should not be too long (avoid false positives)
        if len(raw_text) > 20:
            return False

        return True

    def _validate_confidence(self, score: float) -> bool:
        """Check if extraction confidence is acceptable."""
        return score >= 5.0  # Minimum acceptable score

    def get_validation_errors(self, parsed_data: Dict[str, Any]) -> List[str]:
        """Get detailed validation error messages."""
        errors = []

        if not parsed_data.get('best_match'):
            errors.append("No date candidate found")
            return errors

        candidate = parsed_data['best_match']
        date_obj = candidate.get('date')

        if not isinstance(date_obj, date):
            errors.append("Invalid date object")
            return errors

        if not self._validate_date_range(date_obj):
            today = date.today()
            age_days = (today - date_obj).days
            if age_days > self.max_age_days:
                errors.append(f"Date too old: {age_days} days (maximum: {self.max_age_days})")
            elif date_obj > today:
                errors.append(f"Future date not allowed: {date_obj}")

        if not self._validate_date_format(candidate.get('raw_text', '')):
            errors.append(f"Invalid date format: {candidate.get('raw_text', '')}")

        if not self._validate_confidence(candidate.get('score', 0)):
            errors.append(f"Low confidence score: {candidate.get('score', 0):.1f} (minimum: 5.0)")

        return errors