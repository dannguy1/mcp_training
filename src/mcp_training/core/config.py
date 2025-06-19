"""
Configuration management for MCP Training Service.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic_settings import BaseSettings
from pydantic import Field


class TrainingConfig(BaseSettings):
    """Configuration for the MCP Training Service."""
    
    # Service settings
    service_name: str = Field(default="mcp-training", env="TRAINING_SERVICE_NAME")
    service_version: str = Field(default="1.0.0", env="TRAINING_SERVICE_VERSION")
    debug: bool = Field(default=False, env="TRAINING_DEBUG")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", env="TRAINING_API_HOST")
    api_port: int = Field(default=8001, env="TRAINING_API_PORT")
    api_workers: int = Field(default=1, env="TRAINING_API_WORKERS")
    
    # Storage paths
    models_dir: str = Field(default="models", env="TRAINING_MODELS_DIR")
    exports_dir: str = Field(default="exports", env="TRAINING_EXPORTS_DIR")
    logs_dir: str = Field(default="logs", env="TRAINING_LOGS_DIR")
    
    # Training settings
    max_training_time: int = Field(default=3600, env="TRAINING_MAX_TRAINING_TIME")
    max_memory_usage: int = Field(default=4096, env="TRAINING_MAX_MEMORY_USAGE")
    enable_gpu: bool = Field(default=False, env="TRAINING_ENABLE_GPU")
    
    # Model settings
    default_model_type: str = Field(default="isolation_forest", env="TRAINING_DEFAULT_MODEL_TYPE")
    config_file: str = Field(default="config/model_config.yaml", env="TRAINING_MODEL_CONFIG_FILE")
    
    # Logging
    log_level: str = Field(default="INFO", env="TRAINING_LOG_LEVEL")
    log_file: str = Field(default="logs/training.log", env="TRAINING_LOG_FILE")
    
    # Monitoring
    enable_monitoring: bool = Field(default=True, env="TRAINING_ENABLE_MONITORING")
    prometheus_port: int = Field(default=9091, env="TRAINING_PROMETHEUS_PORT")
    
    # Pydantic v2 config
    model_config = dict(
        env_file=".env",
        case_sensitive=False,
        extra='allow'
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_config: Optional[Dict[str, Any]] = None
        self._training_config: Optional[Dict[str, Any]] = None
    
    @property
    def model_config_data(self) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        if self._model_config is None:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self._model_config = yaml.safe_load(f)
            else:
                self._model_config = {}
        return self._model_config
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Load training service configuration from YAML file."""
        if self._training_config is None:
            config_path = Path("config/training_config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self._training_config = yaml.safe_load(f)
            else:
                self._training_config = {}
        return self._training_config
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get parameters for a specific model type."""
        config = self.model_config_data
        models = config.get('models', {})
        return models.get(model_type, {}).get('parameters', {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration."""
        config = self.model_config_data
        return config.get('feature_extraction', {})
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training parameters."""
        config = self.model_config_data
        return config.get('training', {})
    
    def get_evaluation_params(self) -> Dict[str, Any]:
        """Get evaluation parameters."""
        config = self.model_config_data
        return config.get('evaluation', {})
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.models_dir,
            self.exports_dir,
            self.logs_dir,
            "config",
            "tests",
            "scripts"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Get absolute path for a relative path."""
        return Path.cwd() / relative_path


# Global configuration instance
config = TrainingConfig()

def get_config():
    return {
        "logging": {
            "level": config.log_level,
            "file": config.log_file,
            "format": "structured"
        },
        "storage": {
            "models_dir": config.models_dir,
            "exports_dir": config.exports_dir,
            "logs_dir": config.logs_dir
        },
        "cors": {
            "allowed_origins": ["*"],
            "allowed_methods": ["*"],
            "allowed_headers": ["*"],
            "allow_credentials": True
        },
        "performance": {
            "slow_request_threshold": 1.0
        },
        "auth": {
            "enabled": False
        }
    } 