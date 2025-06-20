"""
Model training for anomaly detection.
"""

import os
import json
import pickle
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

from .config import get_global_config
from .feature_extractor import WiFiFeatureExtractor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train anomaly detection models from extracted features."""
    
    def __init__(self, model_type: str = "isolation_forest"):
        """Initialize the model trainer."""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = WiFiFeatureExtractor()
        config = get_global_config()
        self.model_params = config.get_model_params(model_type)
        self.training_params = config.get_training_params()
        self.evaluation_params = config.get_evaluation_params()
        
    def train(self, data: List[Dict[str, Any]], model_name: Optional[str] = None) -> Dict[str, Any]:
        """Train a model on the provided data."""
        logger.info(f"Starting training with {len(data)} data points")
        
        # Extract features
        features_df = self.feature_extractor.extract_features(data)
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(features_scaled)
        
        # Evaluate model
        evaluation_results = self._evaluate_model(features_scaled)
        
        # Save model
        model_name = model_name or f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self._save_model(model_name, features_df.columns.tolist())
        
        # Prepare results
        results = {
            'model_name': model_name,
            'model_type': self.model_type,
            'model_path': str(model_path),
            'training_date': datetime.now().isoformat(),
            'data_points': len(data),
            'features_count': len(features_df.columns),
            'feature_names': features_df.columns.tolist(),
            'model_params': self.model_params,
            'evaluation': evaluation_results,
            'metadata': {
                'scaler_fitted': True,
                'feature_extractor_config': self.feature_extractor.feature_config
            }
        }
        
        logger.info(f"Training completed. Model saved to {model_path}")
        return results
    
    def _create_model(self):
        """Create model instance based on type."""
        if self.model_type == "isolation_forest":
            return IsolationForest(**self.model_params)
        elif self.model_type == "local_outlier_factor":
            return LocalOutlierFactor(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature dataframe."""
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown')
        
        return df
    
    def _evaluate_model(self, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Evaluate the trained model."""
        # For unsupervised models, we'll use the model's decision function
        # In a real scenario, you might have labeled data for evaluation
        
        # Get anomaly scores
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(features_scaled)
        elif hasattr(self.model, 'score_samples'):
            scores = self.model.score_samples(features_scaled)
        else:
            scores = np.zeros(len(features_scaled))
        
        # Calculate basic statistics
        score_stats = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'percentiles': {
                '25': float(np.percentile(scores, 25)),
                '50': float(np.percentile(scores, 50)),
                '75': float(np.percentile(scores, 75)),
                '95': float(np.percentile(scores, 95)),
                '99': float(np.percentile(scores, 99))
            }
        }
        
        # Cross-validation score (if applicable)
        cv_score = None
        if hasattr(self.model, 'score'):
            try:
                cv_scores = cross_val_score(
                    self.model, features_scaled, 
                    cv=self.training_params.get('cross_validation_folds', 5),
                    scoring='neg_mean_squared_error'
                )
                cv_score = float(np.mean(cv_scores))
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
        
        return {
            'score_statistics': score_stats,
            'cross_validation_score': cv_score,
            'model_info': {
                'n_features': features_scaled.shape[1],
                'n_samples': features_scaled.shape[0]
            }
        }
    
    def _save_model(self, model_name: str, feature_names: List[str]) -> Path:
        """Save the trained model and metadata."""
        config = get_global_config()
        models_dir = Path(config.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_dir = models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_file = model_dir / "model.joblib"
        joblib.dump(self.model, model_file)
        
        # Save scaler
        scaler_file = model_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_file)
        
        # Save feature names
        features_file = model_dir / "features.json"
        with open(features_file, 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        # Save metadata
        metadata_file = model_dir / "metadata.json"
        metadata = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(feature_names),
            'feature_names': feature_names,
            'scaler_fitted': True
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_dir
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model from disk."""
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            logger.error(f"Model directory does not exist: {model_path}")
            return False
        
        try:
            # Load model
            model_file = model_dir / "model.joblib"
            self.model = joblib.load(model_file)
            
            # Load scaler
            scaler_file = model_dir / "scaler.joblib"
            self.scaler = joblib.load(scaler_file)
            
            # Load metadata
            metadata_file = model_dir / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.model_type = metadata.get('model_type', 'isolation_forest')
            self.model_params = metadata.get('model_params', {})
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def predict(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Extract features
        features_df = self.feature_extractor.extract_features(data)
        features_df = self._handle_missing_values(features_df)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make predictions
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(features_scaled)
        else:
            predictions = np.zeros(len(features_scaled))
        
        return predictions
    
    def get_anomaly_scores(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Get anomaly scores for data points."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Extract features
        features_df = self.feature_extractor.extract_features(data)
        features_df = self._handle_missing_values(features_df)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Get scores
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(features_scaled)
        elif hasattr(self.model, 'score_samples'):
            scores = self.model.score_samples(features_scaled)
        else:
            scores = np.zeros(len(features_scaled))
        
        return scores
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available trained models."""
        config = get_global_config()
        models_dir = Path(config.models_dir)
        if not models_dir.exists():
            return []
        
        models = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        models.append({
                            'name': model_dir.name,
                            'path': str(model_dir),
                            'model_type': metadata.get('model_type'),
                            'training_date': metadata.get('training_date'),
                            'feature_count': metadata.get('feature_count'),
                            'model_params': metadata.get('model_params')
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read metadata for {model_dir}: {e}")
        
        return sorted(models, key=lambda x: x['training_date'], reverse=True)
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a trained model."""
        config = get_global_config()
        model_dir = Path(config.models_dir) / model_name
        
        if not model_dir.exists():
            logger.error(f"Model does not exist: {model_name}")
            return False
        
        try:
            import shutil
            shutil.rmtree(model_dir)
            logger.info(f"Model deleted: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False 