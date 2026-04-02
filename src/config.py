"""Configuration management for the check processing system."""

from typing import Dict, Any
import os


class Config:
    """Central configuration for the check processing system."""

    # OCR Configuration
    OCR_CONFIG = {
        'doctr': {
            'det_arch': 'db_resnet50',
            'reco_arch': 'crnn_vgg16_bn',
            'pretrained': True
        }
    }

    # Extractor Configuration
    EXTRACTOR_CONFIG = {
        'amount': {
            'min_amount': 0.01,
            'max_amount': 999999.99,
            'min_confidence': 5.0
        },
        'date': {
            'min_date': '2020-01-01',
            'max_age_days': 365,
            'min_confidence': 5.0
        }
    }

    # Pipeline Configuration
    PIPELINE_CONFIG = {
        'continue_on_error': True,  # Continue processing other fields if one fails
        'save_intermediate': False  # Save intermediate OCR results for debugging
    }

    @classmethod
    def get_ocr_config(cls, ocr_type: str = 'doctr') -> Dict[str, Any]:
        """Get OCR configuration."""
        return cls.OCR_CONFIG.get(ocr_type, {})

    @classmethod
    def get_extractor_config(cls, field_name: str) -> Dict[str, Any]:
        """Get extractor configuration."""
        return cls.EXTRACTOR_CONFIG.get(field_name, {})

    @classmethod
    def get_pipeline_config(cls) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return cls.PIPELINE_CONFIG

    @classmethod
    def update_from_env(cls):
        """Update configuration from environment variables."""
        # TODO: Implement environment variable loading
        # Example: cls.OCR_CONFIG['doctr']['pretrained'] = os.getenv('DOCTR_PRETRAINED', 'true').lower() == 'true'
        pass


# Initialize configuration
Config.update_from_env()