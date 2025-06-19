"""
MCP Training Service

A standalone system for training anomaly detection models from MCP service exports.
"""

__version__ = "1.0.0"
__author__ = "MCP Development Team"
__email__ = "dev@mcp-service.com"

from .core.config import TrainingConfig
from .core.feature_extractor import WiFiFeatureExtractor
from .core.model_trainer import ModelTrainer
from .core.export_validator import ExportValidator

__all__ = [
    "TrainingConfig",
    "WiFiFeatureExtractor", 
    "ModelTrainer",
    "ExportValidator",
] 