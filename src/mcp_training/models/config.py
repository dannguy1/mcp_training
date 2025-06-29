"""
Model configuration for MCP Training Service.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import yaml


class ModelParameters(BaseModel):
    """Model parameters configuration."""
    type: str = Field(default="isolation_forest")
    n_estimators: int = Field(default=100)
    max_samples: str = Field(default="auto")
    contamination: float = Field(default=0.1)
    random_state: int = Field(default=42)
    bootstrap: bool = Field(default=True)
    max_features: float = Field(default=1.0)


class FeatureConfig(BaseModel):
    """Feature configuration."""
    numeric: List[str] = Field(default_factory=list)
    categorical: List[str] = Field(default_factory=list)
    temporal: List[str] = Field(default_factory=list)
    derived: List[str] = Field(default_factory=list)


class StorageConfig(BaseModel):
    """Storage configuration."""
    directory: str = Field(default="models")
    version_format: str = Field(default="%Y%m%d_%H%M%S")
    keep_last_n_versions: int = Field(default=5)
    backup_enabled: bool = Field(default=True)
    compression: bool = Field(default=True)


class ModelConfig(BaseModel):
    """Complete model configuration."""
    model: ModelParameters = Field(default_factory=ModelParameters)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    @classmethod
    def load_from_yaml(cls, config_path: str) -> 'ModelConfig':
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            # Return default config if file doesn't exist
            return cls()
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract relevant sections
        model_data = config_data.get('model', {})
        features_data = config_data.get('features', {})
        storage_data = config_data.get('storage', {})
        
        return cls(
            model=ModelParameters(**model_data),
            features=FeatureConfig(**features_data),
            storage=StorageConfig(**storage_data)
        )
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model': self.model.dict(),
            'features': self.features.dict(),
            'storage': self.storage.dict()
        } 