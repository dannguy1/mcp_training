"""
Training pipeline for MCP Training Service.
"""

from typing import Dict, Any, Optional, List, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Pipeline for training unsupervised anomaly detection models."""
    
    def __init__(self, config=None):
        """Initialize training pipeline."""
        self.config = config
        self.scaler = None
        self.model = None
    
    async def validate_export_for_training(self, export_file: str) -> Dict[str, Any]:
        """Validate export file for training suitability.
        
        Args:
            export_file: Path to export file
            
        Returns:
            Validation results
        """
        try:
            with open(export_file, 'r') as f:
                data = json.load(f)
            
            # Check basic structure
            if 'data' not in data:
                return {
                    'is_valid': False,
                    'errors': ['Export file must contain "data" section']
                }
            
            records = data['data']
            if not records:
                return {
                    'is_valid': False,
                    'errors': ['Export file contains no data records']
                }
            
            # Check minimum record count
            if len(records) < 10:
                return {
                    'is_valid': False,
                    'errors': [f'Insufficient data: {len(records)} records (minimum 10 required)']
                }
            
            # Check data quality
            errors = []
            for i, record in enumerate(records[:100]):  # Check first 100 records
                if not isinstance(record, dict):
                    errors.append(f'Record {i} is not a dictionary')
                    continue
                
                # Check for required fields (basic validation)
                if 'timestamp' not in record:
                    errors.append(f'Record {i} missing timestamp')
                
                if len(errors) >= 10:  # Limit error reporting
                    errors.append('... (additional errors truncated)')
                    break
            
            return {
                'is_valid': len(errors) == 0,
                'errors': errors,
                'record_count': len(records),
                'sample_fields': list(records[0].keys()) if records else []
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [f'Error reading export file: {str(e)}']
            }
    
    async def run_training_pipeline(
        self,
        export_file_paths: List[str],
        model_type: str = "isolation_forest",
        model_name: Optional[str] = None,
        training_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """Run the complete training pipeline.
        
        Args:
            export_file_paths: List of export file paths
            model_type: Type of model to train
            model_name: Optional model name
            training_id: Training job ID
            progress_callback: Optional progress callback function
            
        Returns:
            Training results
        """
        try:
            if progress_callback:
                await progress_callback(15, 'Loading and preprocessing data')
            
            # Step 1: Load and preprocess data
            features_data = await self._load_and_preprocess_data(export_file_paths[0])
            
            if progress_callback:
                await progress_callback(30, 'Extracting features')
            
            # Step 2: Extract features
            X, feature_names = self._extract_features(features_data)
            
            if progress_callback:
                await progress_callback(50, 'Training model')
            
            # Step 3: Train model
            model = await self._train_model(X, model_type)
            
            if progress_callback:
                await progress_callback(70, 'Evaluating model')
            
            # Step 4: Evaluate model
            from .evaluation import ModelEvaluator
            evaluator = ModelEvaluator(self.config)
            evaluation_results = evaluator.evaluate_model(model, X, feature_names)
            
            if progress_callback:
                await progress_callback(90, 'Saving model and metadata')
            
            # Step 5: Save model and metadata
            model_path = await self._save_model_with_metadata(
                model, features_data, evaluation_results, export_file_paths[0], 
                training_id, model_type
            )
            
            if progress_callback:
                await progress_callback(100, 'Training completed')
            
            return {
                'model_version': model_path.name,
                'model_type': model_type,
                'training_samples': len(features_data),
                'evaluation_results': evaluation_results,
                'export_files': export_file_paths,
                'training_duration': 0.0,  # Will be calculated by training service
                'feature_names': feature_names,
                'export_files_size': Path(export_file_paths[0]).stat().st_size,
                'model_parameters': self._get_model_parameters(model),
                'model_path': str(model_path)
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    async def _load_and_preprocess_data(self, export_file: str) -> List[Dict[str, Any]]:
        """Load and preprocess export data."""
        try:
            with open(export_file, 'r') as f:
                data = json.load(f)
            
            records = data.get('data', [])
            logger.info(f"Loaded {len(records)} records from export file")
            
            # Basic preprocessing: filter out invalid records
            valid_records = []
            for record in records:
                if isinstance(record, dict) and 'timestamp' in record:
                    valid_records.append(record)
            
            logger.info(f"Preprocessed {len(valid_records)} valid records")
            return valid_records
            
        except Exception as e:
            logger.error(f"Error loading export data: {e}")
            raise
    
    def _extract_features(self, records: List[Dict[str, Any]]) -> tuple:
        """Extract features from records."""
        try:
            # Convert to DataFrame for easier feature extraction
            df = pd.DataFrame(records)
            
            # Extract basic features
            features = {}
            
            # Time-based features
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                features['hour_of_day'] = df['timestamp'].dt.hour
                features['day_of_week'] = df['timestamp'].dt.dayofweek
                features['minute_of_hour'] = df['timestamp'].dt.minute
                features['time_since_midnight'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
            
            # Boolean features
            features['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
            features['is_business_hours'] = ((df['timestamp'].dt.hour >= 9) & (df['timestamp'].dt.hour <= 17)).astype(int)
            
            # Event type features
            features['is_connection_event'] = df['message'].str.contains('connection|connect|join', case=False, na=False).astype(int)
            features['is_auth_event'] = df['message'].str.contains('auth|authentication|login', case=False, na=False).astype(int)
            features['is_error_event'] = df['message'].str.contains('error|fail|denied', case=False, na=False).astype(int)
            
            # Network features (if available)
            if 'mac_address' in df.columns:
                features['mac_count'] = df.groupby('timestamp').size().reindex(df['timestamp']).fillna(0)
            else:
                features['mac_count'] = 0
            
            if 'ip_address' in df.columns:
                features['ip_count'] = df.groupby('timestamp').size().reindex(df['timestamp']).fillna(0)
            else:
                features['ip_count'] = 0
            
            # Text features
            features['message_length'] = df['message'].str.len().fillna(0)
            features['word_count'] = df['message'].str.split().str.len().fillna(0)
            features['special_char_count'] = df['message'].str.count(r'[^a-zA-Z0-9\s]').fillna(0)
            features['uppercase_ratio'] = (df['message'].str.count(r'[A-Z]') / df['message'].str.len()).fillna(0)
            features['number_count'] = df['message'].str.count(r'\d').fillna(0)
            features['unique_chars'] = df['message'].apply(lambda x: len(set(str(x))) if pd.notna(x) else 0)
            
            # Process features (if available)
            if 'process' in df.columns:
                features['process_rank'] = df['process'].rank(method='dense').fillna(0)
                features['process_frequency'] = df.groupby('process').size().reindex(df['process']).fillna(0)
            else:
                features['process_rank'] = 0
                features['process_frequency'] = 0
            
            # Log level features
            if 'level' in df.columns:
                level_mapping = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
                features['log_level_numeric'] = df['level'].map(level_mapping).fillna(1)
            else:
                features['log_level_numeric'] = 1
            
            # Window-based features (simplified)
            features['window_5min_connection_count'] = 0  # Placeholder
            features['window_5min_unique_macs'] = 0
            features['window_5min_error_count'] = 0
            features['window_5min_process_diversity'] = 0
            features['window_15min_connection_count'] = 0
            features['window_15min_unique_macs'] = 0
            features['window_15min_error_count'] = 0
            features['window_15min_process_diversity'] = 0
            features['window_1hour_connection_count'] = 0
            features['window_1hour_unique_macs'] = 0
            features['window_1hour_error_count'] = 0
            features['window_1hour_process_diversity'] = 0
            
            # Convert to feature matrix
            feature_df = pd.DataFrame(features)
            X = feature_df.values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            feature_names = list(features.keys())
            logger.info(f"Extracted {len(feature_names)} features from {len(records)} records")
            
            return X_scaled, feature_names
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    async def _train_model(self, X: np.ndarray, model_type: str):
        """Train the specified model type."""
        try:
            if model_type == "isolation_forest":
                model = IsolationForest(
                    n_estimators=100,
                    max_samples='auto',
                    contamination=0.1,
                    random_state=42,
                    bootstrap=True,
                    max_features=1.0
                )
            else:
                # Default to Isolation Forest
                model = IsolationForest(
                    n_estimators=100,
                    max_samples='auto',
                    contamination=0.1,
                    random_state=42
                )
            
            # Train the model
            model.fit(X)
            self.model = model
            
            logger.info(f"Trained {model_type} model with {X.shape[0]} samples and {X.shape[1]} features")
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    async def _save_model_with_metadata(
        self,
        model: Any,
        features_data: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any],
        export_file: str,
        training_id: str,
        model_type: str
    ) -> Path:
        """Save model with metadata."""
        try:
            # Generate version
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create model metadata
            from .metadata import ModelMetadata
            metadata = ModelMetadata.create(
                version=version,
                model_type=model_type,
                training_samples=len(features_data),
                feature_names=list(features_data[0].keys()) if features_data else [],
                export_files=[export_file],
                training_id=training_id,
                model_parameters=self._get_model_parameters(model)
            )
            
            # Update evaluation results
            metadata.update_evaluation(evaluation_results)
            metadata.update_export_file_size(Path(export_file).stat().st_size)
            
            # Save model files
            import joblib
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save model
                model_file = temp_path / "model.joblib"
                joblib.dump(model, model_file)
                
                # Save scaler if available
                scaler_file = None
                if self.scaler:
                    scaler_file = temp_path / "scaler.joblib"
                    joblib.dump(self.scaler, scaler_file)
                
                # Save to registry
                from .registry import ModelRegistry
                project_root = Path(__file__).parent.parent.parent.parent
                models_dir = project_root / "models"
                registry = ModelRegistry(str(models_dir))
                
                model_path = registry.save_model(
                    version, metadata, model_file, scaler_file
                )
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def _get_model_parameters(self, model) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            if hasattr(model, 'get_params'):
                return model.get_params()
            else:
                return {'model_type': type(model).__name__}
        except Exception as e:
            logger.error(f"Error getting model parameters: {e}")
            return {'model_type': 'unknown'} 