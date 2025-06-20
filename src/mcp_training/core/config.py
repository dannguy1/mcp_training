"""
Configuration management for MCP Training Service.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class TrainingConfig(BaseSettings):
    """Configuration for the MCP Training Service."""
    
    # Service settings
    service_name: str = Field(default="MCP Training Service", env="TRAINING_SERVICE_NAME")
    service_version: str = Field(default="1.0.0", env="TRAINING_SERVICE_VERSION")
    debug: bool = Field(default=False, env="TRAINING_DEBUG")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", env="TRAINING_API_HOST")
    api_port: int = Field(default=8000, env="TRAINING_API_PORT")
    api_workers: int = Field(default=1, env="TRAINING_API_WORKERS")
    
    # Storage paths
    models_dir: str = Field(default="models", env="TRAINING_MODELS_DIR")
    exports_dir: str = Field(default="exports", env="TRAINING_EXPORTS_DIR")
    logs_dir: str = Field(default="logs", env="TRAINING_LOGS_DIR")
    
    # Training settings
    max_training_time: int = Field(default=3600, env="TRAINING_MAX_TRAINING_TIME")
    max_memory_usage: int = Field(default=4096, env="TRAINING_MAX_MEMORY_USAGE")
    enable_gpu: bool = Field(default=False, env="TRAINING_ENABLE_GPU")
    max_concurrent_jobs: int = Field(default=3, env="TRAINING_MAX_CONCURRENT_JOBS")
    default_max_iterations: int = Field(default=1000, env="TRAINING_DEFAULT_MAX_ITERATIONS")
    default_learning_rate: float = Field(default=0.01, env="TRAINING_DEFAULT_LEARNING_RATE")
    job_timeout: int = Field(default=24, env="TRAINING_JOB_TIMEOUT")
    
    # Model settings
    default_model_type: str = Field(default="isolation_forest", env="TRAINING_DEFAULT_MODEL_TYPE")
    config_file: str = Field(default="config/model_config.yaml", env="TRAINING_MODEL_CONFIG_FILE")
    
    # Logging
    log_level: str = Field(default="INFO", env="TRAINING_LOG_LEVEL")
    log_file: str = Field(default="logs/training.log", env="TRAINING_LOG_FILE")
    log_format: str = Field(default="structured", env="TRAINING_LOG_FORMAT")
    max_log_size_mb: int = Field(default=100, env="TRAINING_MAX_LOG_SIZE_MB")
    log_to_console: bool = Field(default=True, env="TRAINING_LOG_TO_CONSOLE")
    log_to_file: bool = Field(default=True, env="TRAINING_LOG_TO_FILE")
    
    # Monitoring
    enable_monitoring: bool = Field(default=True, env="TRAINING_ENABLE_MONITORING")
    prometheus_port: int = Field(default=9091, env="TRAINING_PROMETHEUS_PORT")
    
    # General UI settings
    timezone: str = Field(default="UTC", env="TRAINING_TIMEZONE")
    date_format: str = Field(default="YYYY-MM-DD", env="TRAINING_DATE_FORMAT")
    auto_refresh: bool = Field(default=True, env="TRAINING_AUTO_REFRESH")
    notifications: bool = Field(default=True, env="TRAINING_NOTIFICATIONS")
    
    # Storage management
    max_storage_gb: int = Field(default=10, env="TRAINING_MAX_STORAGE_GB")
    auto_cleanup: bool = Field(default=True, env="TRAINING_AUTO_CLEANUP")
    retention_days: int = Field(default=30, env="TRAINING_RETENTION_DAYS")
    
    # Security settings
    auth_enabled: bool = Field(default=False, env="TRAINING_AUTH_ENABLED")
    api_key: str = Field(default="", env="TRAINING_API_KEY")
    cors_origins: str = Field(default="http://localhost:3000,https://example.com", env="TRAINING_CORS_ORIGINS")
    rate_limit: int = Field(default=100, env="TRAINING_RATE_LIMIT")
    https_only: bool = Field(default=False, env="TRAINING_HTTPS_ONLY")
    secure_headers: bool = Field(default=True, env="TRAINING_SECURE_HEADERS")
    
    # Advanced settings
    performance_monitoring: bool = Field(default=True, env="TRAINING_PERFORMANCE_MONITORING")
    websocket_enabled: bool = Field(default=True, env="TRAINING_WEBSOCKET_ENABLED")
    auto_backup: bool = Field(default=False, env="TRAINING_AUTO_BACKUP")
    
    # Pydantic v2 config
    model_config = dict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_config: Optional[Dict[str, Any]] = None
        self._training_config: Optional[Dict[str, Any]] = None
    
    def get_cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(',') if origin.strip()]
        return []
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format for API responses."""
        return {
            "general": {
                "service_name": self.service_name,
                "service_version": self.service_version,
                "timezone": self.timezone,
                "date_format": self.date_format,
                "auto_refresh": self.auto_refresh,
                "notifications": self.notifications
            },
            "training": {
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "default_max_iterations": self.default_max_iterations,
                "default_learning_rate": self.default_learning_rate,
                "job_timeout": self.job_timeout,
                "default_config": self.get_model_params(self.default_model_type)
            },
            "storage": {
                "models_dir": self.models_dir,
                "exports_dir": self.exports_dir,
                "logs_dir": self.logs_dir,
                "max_storage_gb": self.max_storage_gb,
                "auto_cleanup": self.auto_cleanup,
                "retention_days": self.retention_days
            },
            "logging": {
                "log_level": self.log_level,
                "log_format": self.log_format,
                "log_file": self.log_file,
                "max_log_size_mb": self.max_log_size_mb,
                "log_to_console": self.log_to_console,
                "log_to_file": self.log_to_file
            },
            "security": {
                "auth_enabled": self.auth_enabled,
                "api_key": self.api_key,
                "cors_origins": self.cors_origins if isinstance(self.cors_origins, str) else ",".join(self.cors_origins),
                "rate_limit": self.rate_limit,
                "https_only": self.https_only,
                "secure_headers": self.secure_headers
            },
            "advanced": {
                "debug_mode": self.debug,
                "performance_monitoring": self.performance_monitoring,
                "websocket_enabled": self.websocket_enabled,
                "auto_backup": self.auto_backup,
                "custom_config": {}
            }
        }
    
    def update_from_dict(self, settings: Dict[str, Any]) -> None:
        """Update configuration from dictionary and save to environment file."""
        # Update general settings
        if "general" in settings:
            general = settings["general"]
            if "service_name" in general:
                self.service_name = general["service_name"]
            if "timezone" in general:
                self.timezone = general["timezone"]
            if "date_format" in general:
                self.date_format = general["date_format"]
            if "auto_refresh" in general:
                self.auto_refresh = general["auto_refresh"]
            if "notifications" in general:
                self.notifications = general["notifications"]
        
        # Update training settings
        if "training" in settings:
            training = settings["training"]
            if "max_concurrent_jobs" in training:
                self.max_concurrent_jobs = training["max_concurrent_jobs"]
            if "default_max_iterations" in training:
                self.default_max_iterations = training["default_max_iterations"]
            if "default_learning_rate" in training:
                self.default_learning_rate = training["default_learning_rate"]
            if "job_timeout" in training:
                self.job_timeout = training["job_timeout"]
        
        # Update storage settings
        if "storage" in settings:
            storage = settings["storage"]
            if "models_dir" in storage:
                self.models_dir = storage["models_dir"]
            if "exports_dir" in storage:
                self.exports_dir = storage["exports_dir"]
            if "logs_dir" in storage:
                self.logs_dir = storage["logs_dir"]
            if "max_storage_gb" in storage:
                self.max_storage_gb = storage["max_storage_gb"]
            if "auto_cleanup" in storage:
                self.auto_cleanup = storage["auto_cleanup"]
            if "retention_days" in storage:
                self.retention_days = storage["retention_days"]
        
        # Update logging settings
        if "logging" in settings:
            logging_config = settings["logging"]
            if "log_level" in logging_config:
                self.log_level = logging_config["log_level"]
            if "log_format" in logging_config:
                self.log_format = logging_config["log_format"]
            if "log_file" in logging_config:
                self.log_file = logging_config["log_file"]
            if "max_log_size_mb" in logging_config:
                self.max_log_size_mb = logging_config["max_log_size_mb"]
            if "log_to_console" in logging_config:
                self.log_to_console = logging_config["log_to_console"]
            if "log_to_file" in logging_config:
                self.log_to_file = logging_config["log_to_file"]
        
        # Update security settings
        if "security" in settings:
            security = settings["security"]
            if "auth_enabled" in security:
                self.auth_enabled = security["auth_enabled"]
            if "api_key" in security:
                self.api_key = security["api_key"]
            if "cors_origins" in security:
                self.cors_origins = security["cors_origins"]
            if "rate_limit" in security:
                self.rate_limit = security["rate_limit"]
            if "https_only" in security:
                self.https_only = security["https_only"]
            if "secure_headers" in security:
                self.secure_headers = security["secure_headers"]
        
        # Update advanced settings
        if "advanced" in settings:
            advanced = settings["advanced"]
            if "debug_mode" in advanced:
                self.debug = advanced["debug_mode"]
            if "performance_monitoring" in advanced:
                self.performance_monitoring = advanced["performance_monitoring"]
            if "websocket_enabled" in advanced:
                self.websocket_enabled = advanced["websocket_enabled"]
            if "auto_backup" in advanced:
                self.auto_backup = advanced["auto_backup"]
        
        # Save to .env file
        self.save_to_env_file()
    
    def save_to_env_file(self) -> None:
        """Save current configuration to .env file."""
        env_content = []
        
        # Service Configuration
        env_content.extend([
            f"# Service Configuration",
            f"TRAINING_SERVICE_NAME={self.service_name}",
            f"TRAINING_SERVICE_VERSION={self.service_version}",
            f"TRAINING_DEBUG={str(self.debug).lower()}",
            ""
        ])
        
        # API Configuration
        env_content.extend([
            f"# API Configuration",
            f"TRAINING_API_HOST={self.api_host}",
            f"TRAINING_API_PORT={self.api_port}",
            f"TRAINING_API_WORKERS={self.api_workers}",
            ""
        ])
        
        # Storage Configuration
        env_content.extend([
            f"# Storage Configuration",
            f"TRAINING_MODELS_DIR={self.models_dir}",
            f"TRAINING_EXPORTS_DIR={self.exports_dir}",
            f"TRAINING_LOGS_DIR={self.logs_dir}",
            f"TRAINING_MAX_STORAGE_GB={self.max_storage_gb}",
            f"TRAINING_AUTO_CLEANUP={str(self.auto_cleanup).lower()}",
            f"TRAINING_RETENTION_DAYS={self.retention_days}",
            ""
        ])
        
        # Training Configuration
        env_content.extend([
            f"# Training Configuration",
            f"TRAINING_MAX_TRAINING_TIME={self.max_training_time}",
            f"TRAINING_MAX_MEMORY_USAGE={self.max_memory_usage}",
            f"TRAINING_ENABLE_GPU={str(self.enable_gpu).lower()}",
            f"TRAINING_MAX_CONCURRENT_JOBS={self.max_concurrent_jobs}",
            f"TRAINING_DEFAULT_MAX_ITERATIONS={self.default_max_iterations}",
            f"TRAINING_DEFAULT_LEARNING_RATE={self.default_learning_rate}",
            f"TRAINING_JOB_TIMEOUT={self.job_timeout}",
            ""
        ])
        
        # Model Configuration
        env_content.extend([
            f"# Model Configuration",
            f"TRAINING_DEFAULT_MODEL_TYPE={self.default_model_type}",
            f"TRAINING_MODEL_CONFIG_FILE={self.config_file}",
            ""
        ])
        
        # Logging Configuration
        env_content.extend([
            f"# Logging Configuration",
            f"TRAINING_LOG_LEVEL={self.log_level}",
            f"TRAINING_LOG_FILE={self.log_file}",
            f"TRAINING_LOG_FORMAT={self.log_format}",
            f"TRAINING_MAX_LOG_SIZE_MB={self.max_log_size_mb}",
            f"TRAINING_LOG_TO_CONSOLE={str(self.log_to_console).lower()}",
            f"TRAINING_LOG_TO_FILE={str(self.log_to_file).lower()}",
            ""
        ])
        
        # General UI Configuration
        env_content.extend([
            f"# General UI Configuration",
            f"TRAINING_TIMEZONE={self.timezone}",
            f"TRAINING_DATE_FORMAT={self.date_format}",
            f"TRAINING_AUTO_REFRESH={str(self.auto_refresh).lower()}",
            f"TRAINING_NOTIFICATIONS={str(self.notifications).lower()}",
            ""
        ])
        
        # Security Configuration
        env_content.extend([
            f"# Security Configuration",
            f"TRAINING_AUTH_ENABLED={str(self.auth_enabled).lower()}",
            f"TRAINING_API_KEY={self.api_key}",
            f"TRAINING_CORS_ORIGINS={self.cors_origins if isinstance(self.cors_origins, str) else ','.join(self.cors_origins)}",
            f"TRAINING_RATE_LIMIT={self.rate_limit}",
            f"TRAINING_HTTPS_ONLY={str(self.https_only).lower()}",
            f"TRAINING_SECURE_HEADERS={str(self.secure_headers).lower()}",
            ""
        ])
        
        # Advanced Configuration
        env_content.extend([
            f"# Advanced Configuration",
            f"TRAINING_ENABLE_MONITORING={str(self.enable_monitoring).lower()}",
            f"TRAINING_PROMETHEUS_PORT={self.prometheus_port}",
            f"TRAINING_PERFORMANCE_MONITORING={str(self.performance_monitoring).lower()}",
            f"TRAINING_WEBSOCKET_ENABLED={str(self.websocket_enabled).lower()}",
            f"TRAINING_AUTO_BACKUP={str(self.auto_backup).lower()}"
        ])
        
        # Write to .env file
        with open(".env", "w") as f:
            f.write("\n".join(env_content))
    
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
            "format": config.log_format
        },
        "storage": {
            "models_dir": config.models_dir,
            "exports_dir": config.exports_dir,
            "logs_dir": config.logs_dir
        },
        "cors": {
            "allowed_origins": config.get_cors_origins_list(),
            "allowed_methods": ["*"],
            "allowed_headers": ["*"],
            "allow_credentials": True
        },
        "performance": {
            "slow_request_threshold": 1.0
        },
        "auth": {
            "enabled": config.auth_enabled,
            "api_key": config.api_key
        }
    } 