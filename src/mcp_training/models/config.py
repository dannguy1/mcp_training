"""
Model configuration management for MCP Training Service.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ModelParameters(BaseModel):
    """Model training parameters."""
    n_estimators: int = Field(default=100, description="Number of estimators")
    max_samples: str = Field(default="auto", description="Maximum samples")
    contamination: float = Field(default=0.1, description="Contamination ratio")
    random_state: int = Field(default=42, description="Random state")
    bootstrap: bool = Field(default=True, description="Bootstrap samples")
    max_features: float = Field(default=1.0, description="Maximum features")


class FeatureConfig(BaseModel):
    """Feature configuration."""
    numeric: List[str] = Field(default_factory=list, description="Numeric features")
    categorical: List[str] = Field(default_factory=list, description="Categorical features")
    temporal: List[str] = Field(default_factory=list, description="Temporal features")
    derived: List[str] = Field(default_factory=list, description="Derived features")


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""
    enabled: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=10, description="Early stopping patience")
    min_delta: float = Field(default=0.001, description="Minimum delta for improvement")


class TrainingConfig(BaseModel):
    """Training configuration."""
    test_size: float = Field(default=0.2, description="Test set size")
    validation_size: float = Field(default=0.1, description="Validation set size")
    random_state: int = Field(default=42, description="Random state")
    n_jobs: int = Field(default=-1, description="Number of jobs")
    cross_validation_folds: int = Field(default=5, description="CV folds")
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig, description="Early stopping config")


class StorageConfig(BaseModel):
    """Storage configuration."""
    directory: str = Field(default="models", description="Models directory")
    version_format: str = Field(default="%Y%m%d_%H%M%S", description="Version format")
    keep_last_n_versions: int = Field(default=5, description="Keep last N versions")
    backup_enabled: bool = Field(default=True, description="Enable backup")
    compression: bool = Field(default=True, description="Enable compression")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    metrics: List[str] = Field(default_factory=list, description="Evaluation metrics")
    thresholds: Dict[str, float] = Field(default_factory=dict, description="Metric thresholds")
    cross_validation: bool = Field(default=True, description="Enable cross-validation")


class DatabaseConfig(BaseModel):
    """Database configuration for training."""
    table_name: str = Field(default="log_entries", description="Main log table")
    wifi_programs: List[str] = Field(default=["hostapd", "wpa_supplicant"], description="WiFi-related programs")
    batch_size: int = Field(default=1000, description="Data fetching batch size")
    max_records: int = Field(default=100000, description="Maximum records to fetch")


class AlertingConfig(BaseModel):
    """Alerting configuration."""
    enabled: bool = Field(default=True, description="Enable alerting")
    email_notifications: bool = Field(default=False, description="Email notifications")
    slack_notifications: bool = Field(default=False, description="Slack notifications")


class MonitoringConfig(BaseModel):
    """Monitoring configuration for training."""
    enable_drift_detection: bool = Field(default=True, description="Enable drift detection")
    drift_threshold: float = Field(default=0.1, description="Drift detection threshold")
    performance_tracking: bool = Field(default=True, description="Enable performance tracking")
    resource_monitoring: bool = Field(default=True, description="Enable resource monitoring")
    alerting: AlertingConfig = Field(default_factory=AlertingConfig, description="Alerting configuration")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: str = Field(default="logs/training.log", description="Log file")
    max_size: str = Field(default="100MB", description="Max log size")
    backup_count: int = Field(default=10, description="Backup count")


class ModelConfig(BaseModel):
    """Complete model configuration."""
    version: str = Field(default="2.0.0", description="Configuration version")
    model: ModelParameters = Field(default_factory=ModelParameters)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def __init__(self, **data):
        super().__init__(**data)
        # Set default features if not provided
        if not self.features.numeric:
            self.features.numeric = [
                'auth_failure_ratio',
                'deauth_ratio',
                'beacon_ratio',
                'unique_mac_count',
                'unique_ssid_count',
                'mean_signal_strength',
                'std_signal_strength',
                'mean_data_rate',
                'mean_packet_loss',
                'error_ratio',
                'warning_ratio',
                'mean_hour_of_day',
                'mean_day_of_week',
                'mean_time_between_events',
                'total_devices',
                'max_device_activity',
                'mean_device_activity'
            ]
        
        if not self.features.temporal:
            self.features.temporal = [
                'mean_hour_of_day',
                'mean_day_of_week',
                'mean_time_between_events'
            ]
        
        if not self.features.derived:
            self.features.derived = [
                'signal_strength_trend',
                'connection_stability',
                'network_load'
            ]
        
        if not self.evaluation.metrics:
            self.evaluation.metrics = [
                'accuracy',
                'precision',
                'recall',
                'f1_score',
                'roc_auc',
                'average_precision'
            ]
        
        if not self.evaluation.thresholds:
            self.evaluation.thresholds = {
                'accuracy': 0.8,
                'precision': 0.7,
                'recall': 0.7,
                'f1_score': 0.7,
                'roc_auc': 0.8
            } 