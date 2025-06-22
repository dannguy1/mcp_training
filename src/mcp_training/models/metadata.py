"""
Model metadata management for MCP Training Service.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from pathlib import Path
import json


class ModelInfo(BaseModel):
    """Model information."""
    model_config = ConfigDict(protected_namespaces=())
    
    version: str = Field(..., description="Model version")
    created_at: str = Field(..., description="Creation timestamp")
    model_type: str = Field(..., description="Type of model")
    training_source: str = Field(default="export_data", description="Training data source")
    export_file: Optional[str] = Field(None, description="Source export file")
    training_id: Optional[str] = Field(None, description="Training job ID")


class TrainingInfo(BaseModel):
    """Training information."""
    model_config = ConfigDict(protected_namespaces=())
    
    training_samples: int = Field(..., description="Number of training samples")
    feature_names: List[str] = Field(default_factory=list, description="Feature names")
    export_file_size: Optional[int] = Field(None, description="Export file size")
    training_duration: Optional[float] = Field(None, description="Training duration in seconds")
    model_parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")


class EvaluationInfo(BaseModel):
    """Evaluation information."""
    model_config = ConfigDict(protected_namespaces=())
    
    basic_metrics: Dict[str, Optional[float]] = Field(default_factory=dict, description="Basic metrics")
    score_distribution: Dict[str, Any] = Field(default_factory=dict, description="Score distribution")
    cross_validation_score: Optional[float] = Field(None, description="Cross-validation score")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance")


class DeploymentInfo(BaseModel):
    """Deployment information."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = Field(default="available", description="Deployment status")
    deployed_at: Optional[str] = Field(None, description="Deployment timestamp")
    deployed_by: Optional[str] = Field(None, description="Deployed by")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


class ModelMetadata(BaseModel):
    """Complete model metadata."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_info: ModelInfo
    training_info: TrainingInfo
    evaluation_info: EvaluationInfo
    deployment_info: DeploymentInfo = Field(default_factory=DeploymentInfo)

    @classmethod
    def create(cls, 
               version: str,
               model_type: str,
               training_samples: int,
               feature_names: List[str],
               export_file: Optional[str] = None,
               training_id: Optional[str] = None,
               model_parameters: Optional[Dict[str, Any]] = None) -> 'ModelMetadata':
        """Create new model metadata."""
        return cls(
            model_info=ModelInfo(
                version=version,
                created_at=datetime.now().isoformat(),
                model_type=model_type,
                training_source="export_data",
                export_file=export_file,
                training_id=training_id
            ),
            training_info=TrainingInfo(
                training_samples=training_samples,
                feature_names=feature_names,
                model_parameters=model_parameters or {}
            ),
            evaluation_info=EvaluationInfo()
        )

    def save(self, model_dir: Path) -> Path:
        """Save metadata to file."""
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.dict(), f, indent=2, default=str)
        return metadata_path

    @classmethod
    def load(cls, model_dir: Path) -> 'ModelMetadata':
        """Load metadata from file."""
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)

    def update_evaluation(self, evaluation_results: Dict[str, Any]):
        """Update evaluation information."""
        self.evaluation_info.basic_metrics = evaluation_results.get('basic_metrics', {})
        self.evaluation_info.score_distribution = evaluation_results.get('score_distribution', {})
        self.evaluation_info.cross_validation_score = evaluation_results.get('cross_validation_score')

    def update_training_duration(self, duration: float):
        """Update training duration."""
        self.training_info.training_duration = duration

    def update_export_file_size(self, file_size: int):
        """Update export file size."""
        self.training_info.export_file_size = file_size

    def deploy(self, deployed_by: Optional[str] = None):
        """Mark model as deployed."""
        self.deployment_info.status = "deployed"
        self.deployment_info.deployed_at = datetime.now().isoformat()
        self.deployment_info.deployed_by = deployed_by

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.dict() 