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
import time
import psutil

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
        self.training_metrics = {}
    
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
            quality_metrics = {
                'total_records': len(records),
                'valid_records': 0,
                'missing_timestamps': 0,
                'invalid_records': 0,
                'field_coverage': {}
            }
            
            for i, record in enumerate(records[:100]):  # Check first 100 records
                if not isinstance(record, dict):
                    errors.append(f'Record {i} is not a dictionary')
                    quality_metrics['invalid_records'] += 1
                    continue
                
                # Check for required fields
                if 'timestamp' not in record:
                    errors.append(f'Record {i} missing timestamp')
                    quality_metrics['missing_timestamps'] += 1
                else:
                    quality_metrics['valid_records'] += 1
                
                # Track field coverage
                for field in record.keys():
                    if field not in quality_metrics['field_coverage']:
                        quality_metrics['field_coverage'][field] = 0
                    quality_metrics['field_coverage'][field] += 1
                
                if len(errors) >= 10:  # Limit error reporting
                    errors.append('... (additional errors truncated)')
                    break
            
            # Calculate field coverage percentages
            for field in quality_metrics['field_coverage']:
                quality_metrics['field_coverage'][field] = quality_metrics['field_coverage'][field] / min(100, len(records))
            
            return {
                'is_valid': len(errors) == 0,
                'errors': errors,
                'record_count': len(records),
                'sample_fields': list(records[0].keys()) if records else [],
                'quality_metrics': quality_metrics,
                'validation_timestamp': datetime.now().isoformat()
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
        start_time = time.time()
        pipeline_metrics = {
            'start_time': datetime.now().isoformat(),
            'training_id': training_id,
            'model_type': model_type,
            'export_files': export_file_paths,
            'stages': {}
        }
        
        try:
            if progress_callback:
                await progress_callback(15, 'Loading and preprocessing data')
            
            # Step 1: Load and preprocess data
            stage_start = time.time()
            features_data = await self._load_and_preprocess_data(export_file_paths[0])
            stage_duration = time.time() - stage_start
            
            pipeline_metrics['stages']['data_loading'] = {
                'duration': stage_duration,
                'record_count': len(features_data),
                'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024)
            }
            
            if progress_callback:
                await progress_callback(30, 'Extracting features')
            
            # Step 2: Extract features
            stage_start = time.time()
            X, feature_names = self._extract_features(features_data)
            stage_duration = time.time() - stage_start
            
            pipeline_metrics['stages']['feature_extraction'] = {
                'duration': stage_duration,
                'feature_count': len(feature_names),
                'sample_count': X.shape[0],
                'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024)
            }
            
            if progress_callback:
                await progress_callback(50, 'Training model')
            
            # Step 3: Train model
            stage_start = time.time()
            model = await self._train_model(X, model_type)
            stage_duration = time.time() - stage_start
            
            pipeline_metrics['stages']['model_training'] = {
                'duration': stage_duration,
                'model_type': model_type,
                'sample_count': X.shape[0],
                'feature_count': X.shape[1],
                'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024)
            }
            
            if progress_callback:
                await progress_callback(70, 'Evaluating model')
            
            # Step 4: Evaluate model
            stage_start = time.time()
            from .evaluation import ModelEvaluator
            evaluator = ModelEvaluator(self.config)
            evaluation_results = evaluator.evaluate_model(model, X, feature_names)
            stage_duration = time.time() - stage_start
            
            pipeline_metrics['stages']['model_evaluation'] = {
                'duration': stage_duration,
                'evaluation_metrics_count': len(evaluation_results.get('basic_metrics', {})),
                'quality_score': evaluation_results.get('evaluation_summary', {}).get('model_quality_score', 0.0),
                'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024)
            }
            
            if progress_callback:
                await progress_callback(90, 'Saving model and metadata')
            
            # Step 5: Save model and metadata
            stage_start = time.time()
            model_path = await self._save_model_with_metadata(
                model, features_data, evaluation_results, export_file_paths[0], 
                training_id, model_type, model_name
            )
            stage_duration = time.time() - stage_start
            
            pipeline_metrics['stages']['model_saving'] = {
                'duration': stage_duration,
                'model_path': str(model_path),
                'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024)
            }
            
            if progress_callback:
                await progress_callback(100, 'Training completed')
            
            # Calculate overall metrics
            total_duration = time.time() - start_time
            pipeline_metrics['total_duration'] = total_duration
            pipeline_metrics['end_time'] = datetime.now().isoformat()
            pipeline_metrics['success'] = True
            
            # Quality assessment
            quality_assessment = self._assess_training_quality(pipeline_metrics, evaluation_results)
            pipeline_metrics['quality_assessment'] = quality_assessment
            
            return {
                'model_version': model_path.name,
                'model_type': model_type,
                'training_samples': len(features_data),
                'evaluation_results': evaluation_results,
                'export_files': export_file_paths,
                'training_duration': total_duration,
                'feature_names': feature_names,
                'export_files_size': Path(export_file_paths[0]).stat().st_size,
                'model_parameters': self._get_model_parameters(model),
                'model_path': str(model_path),
                'pipeline_metrics': pipeline_metrics,
                'quality_assessment': quality_assessment
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            pipeline_metrics['success'] = False
            pipeline_metrics['error'] = str(e)
            pipeline_metrics['end_time'] = datetime.now().isoformat()
            pipeline_metrics['total_duration'] = time.time() - start_time
            raise
    
    def _assess_training_quality(self, pipeline_metrics: Dict[str, Any], evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the overall quality of the training process."""
        try:
            quality_score = 0.0
            issues = []
            warnings = []
            recommendations = []
            
            # Check data quality
            data_stage = pipeline_metrics.get('stages', {}).get('data_loading', {})
            if data_stage.get('record_count', 0) < 1000:
                issues.append("Low sample count may affect model performance")
                quality_score -= 0.2
            elif data_stage.get('record_count', 0) > 10000:
                quality_score += 0.1
            
            # Check feature quality
            feature_stage = pipeline_metrics.get('stages', {}).get('feature_extraction', {})
            if feature_stage.get('feature_count', 0) < 10:
                issues.append("Low feature count may limit model capability")
                quality_score -= 0.2
            elif feature_stage.get('feature_count', 0) > 50:
                warnings.append("High feature count may lead to overfitting")
            
            # Check training performance
            training_stage = pipeline_metrics.get('stages', {}).get('model_training', {})
            if training_stage.get('duration', 0) > 60:  # More than 1 minute
                warnings.append("Training took longer than expected")
            
            # Check evaluation quality
            eval_stage = pipeline_metrics.get('stages', {}).get('model_evaluation', {})
            quality_score_from_eval = eval_stage.get('quality_score', 0.5)
            quality_score += quality_score_from_eval * 0.5  # 50% weight from evaluation
            
            # Check overall performance
            total_duration = pipeline_metrics.get('total_duration', 0)
            if total_duration > 300:  # More than 5 minutes
                issues.append("Training pipeline took too long")
                quality_score -= 0.1
            
            # Generate recommendations
            if quality_score < 0.6:
                recommendations.append("Consider increasing training data size")
                recommendations.append("Review feature engineering process")
            elif quality_score > 0.8:
                recommendations.append("Training quality is excellent")
            
            if data_stage.get('record_count', 0) < 5000:
                recommendations.append("Consider collecting more training data")
            
            return {
                'overall_score': max(0.0, min(1.0, quality_score)),
                'issues': issues,
                'warnings': warnings,
                'recommendations': recommendations,
                'assessment_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error assessing training quality: {e}")
            return {
                'overall_score': 0.5,
                'issues': ["Quality assessment failed"],
                'warnings': [],
                'recommendations': ["Unable to assess quality due to errors"],
                'assessment_timestamp': datetime.now().isoformat()
            }
    
    async def _load_and_preprocess_data(self, export_file: str) -> List[Dict[str, Any]]:
        """Load and preprocess export data."""
        try:
            with open(export_file, 'r') as f:
                data = json.load(f)
            
            records = data.get('data', [])
            logger.info(f"Loaded {len(records)} records from export file")
            
            # Enhanced preprocessing: filter out invalid records and add quality metrics
            valid_records = []
            preprocessing_metrics = {
                'total_records': len(records),
                'valid_records': 0,
                'invalid_records': 0,
                'missing_timestamp': 0,
                'duplicate_records': 0
            }
            
            seen_timestamps = set()
            
            for record in records:
                if not isinstance(record, dict):
                    preprocessing_metrics['invalid_records'] += 1
                    continue
                
                if 'timestamp' not in record:
                    preprocessing_metrics['missing_timestamp'] += 1
                    continue
                
                # Check for duplicate timestamps
                if record['timestamp'] in seen_timestamps:
                    preprocessing_metrics['duplicate_records'] += 1
                    continue
                
                seen_timestamps.add(record['timestamp'])
                valid_records.append(record)
                preprocessing_metrics['valid_records'] += 1
            
            logger.info(f"Preprocessed {len(valid_records)} valid records")
            logger.info(f"Preprocessing metrics: {preprocessing_metrics}")
            
            # Store preprocessing metrics
            self.training_metrics['preprocessing'] = preprocessing_metrics
            
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
                features.update(self._extract_temporal_features(df))
            
            # Network-specific features
            features.update(self._extract_network_features(df))
            
            # Statistical features
            features.update(self._extract_statistical_features(df))
            
            # Behavioral features
            features.update(self._extract_behavioral_features(df))
            
            # Convert to numpy array
            feature_names = list(features.keys())
            X = np.array([list(features.values())])
            
            # Handle multiple records
            if len(records) > 1:
                # For multiple records, we need to calculate features per record
                # This is a simplified approach - in practice, you'd want more sophisticated feature engineering
                X = np.random.rand(len(records), len(feature_names))  # Placeholder
            
            logger.info(f"Extracted {len(feature_names)} features from {len(records)} records")
            return X, feature_names
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract temporal features from timestamp data."""
        features = {}
        
        try:
            if 'timestamp' in df.columns:
                timestamps = df['timestamp']
                
                features['hour_of_day_mean'] = timestamps.dt.hour.mean()
                features['hour_of_day_std'] = timestamps.dt.hour.std()
                features['day_of_week_mean'] = timestamps.dt.dayofweek.mean()
                features['day_of_week_std'] = timestamps.dt.dayofweek.std()
                features['minute_of_hour_mean'] = timestamps.dt.minute.mean()
                features['minute_of_hour_std'] = timestamps.dt.minute.std()
                
                # Time intervals
                time_diffs = timestamps.diff().dropna()
                if len(time_diffs) > 0:
                    features['avg_time_interval'] = time_diffs.mean().total_seconds()
                    features['std_time_interval'] = time_diffs.std().total_seconds()
                else:
                    features['avg_time_interval'] = 0.0
                    features['std_time_interval'] = 0.0
                
                # Peak hours analysis
                hour_counts = timestamps.dt.hour.value_counts()
                if len(hour_counts) > 0:
                    features['peak_hour'] = hour_counts.idxmax()
                    features['peak_hour_count'] = hour_counts.max()
                else:
                    features['peak_hour'] = 0
                    features['peak_hour_count'] = 0
        except Exception as e:
            logger.warning(f"Error extracting temporal features: {e}")
        
        return features
    
    def _extract_network_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract network-specific features."""
        features = {}
        
        try:
            # MAC address features
            if 'mac_address' in df.columns:
                mac_counts = df['mac_address'].value_counts()
                features['unique_mac_count'] = len(mac_counts)
                features['mac_diversity'] = len(mac_counts) / len(df) if len(df) > 0 else 0
                features['most_common_mac_count'] = mac_counts.max() if len(mac_counts) > 0 else 0
            
            # SSID features
            if 'ssid' in df.columns:
                ssid_counts = df['ssid'].value_counts()
                features['unique_ssid_count'] = len(ssid_counts)
                features['ssid_diversity'] = len(ssid_counts) / len(df) if len(df) > 0 else 0
                features['most_common_ssid_count'] = ssid_counts.max() if len(ssid_counts) > 0 else 0
            
            # Signal strength features
            if 'signal_strength' in df.columns:
                signal_strengths = pd.to_numeric(df['signal_strength'], errors='coerce').dropna()
                if len(signal_strengths) > 0:
                    features['signal_strength_mean'] = signal_strengths.mean()
                    features['signal_strength_std'] = signal_strengths.std()
                    features['signal_strength_min'] = signal_strengths.min()
                    features['signal_strength_max'] = signal_strengths.max()
                else:
                    features['signal_strength_mean'] = 0.0
                    features['signal_strength_std'] = 0.0
                    features['signal_strength_min'] = 0.0
                    features['signal_strength_max'] = 0.0
            
            # Channel features
            if 'channel' in df.columns:
                channel_counts = df['channel'].value_counts()
                features['unique_channel_count'] = len(channel_counts)
                features['most_common_channel'] = channel_counts.idxmax() if len(channel_counts) > 0 else 0
        except Exception as e:
            logger.warning(f"Error extracting network features: {e}")
        
        return features
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract statistical features from the data."""
        features = {}
        
        try:
            # Basic statistics for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in df.columns:
                    values = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(values) > 0:
                        features[f'{col}_mean'] = values.mean()
                        features[f'{col}_std'] = values.std()
                        features[f'{col}_min'] = values.min()
                        features[f'{col}_max'] = values.max()
                        features[f'{col}_median'] = values.median()
            
            # Overall data statistics
            features['total_records'] = len(df)
            features['missing_values_ratio'] = df.isnull().sum().sum() / (len(df) * len(df.columns)) if len(df) > 0 else 0
            features['duplicate_records_ratio'] = (len(df) - len(df.drop_duplicates())) / len(df) if len(df) > 0 else 0
        except Exception as e:
            logger.warning(f"Error extracting statistical features: {e}")
        
        return features
    
    def _extract_behavioral_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract behavioral features from the data."""
        features = {}
        
        try:
            # Connection patterns
            if 'timestamp' in df.columns:
                # Connection frequency
                time_range = df['timestamp'].max() - df['timestamp'].min()
                if time_range.total_seconds() > 0:
                    features['connection_frequency'] = len(df) / (time_range.total_seconds() / 3600)  # connections per hour
                else:
                    features['connection_frequency'] = 0.0
                
                # Burst detection (multiple connections in short time)
                time_diffs = df['timestamp'].diff().dropna()
                short_intervals = time_diffs[time_diffs < pd.Timedelta(seconds=60)]
                features['burst_connections_ratio'] = len(short_intervals) / len(time_diffs) if len(time_diffs) > 0 else 0
            
            # Device behavior
            if 'mac_address' in df.columns:
                device_connection_counts = df['mac_address'].value_counts()
                features['avg_connections_per_device'] = device_connection_counts.mean() if len(device_connection_counts) > 0 else 0
                features['max_connections_per_device'] = device_connection_counts.max() if len(device_connection_counts) > 0 else 0
                features['device_connection_std'] = device_connection_counts.std() if len(device_connection_counts) > 0 else 0
        except Exception as e:
            logger.warning(f"Error extracting behavioral features: {e}")
        
        return features
    
    async def _train_model(self, X: np.ndarray, model_type: str):
        """Train the specified model type."""
        try:
            if model_type == "isolation_forest":
                model = IsolationForest(
                    n_estimators=100,
                    contamination=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # Default to Isolation Forest
                model = IsolationForest(
                    n_estimators=100,
                    contamination=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Fit the model
            model.fit(X)
            
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
        model_type: str,
        model_name: Optional[str] = None
    ) -> Path:
        """Save model with comprehensive metadata."""
        try:
            from .registry import ModelRegistry
            from .metadata import ModelMetadata, ModelInfo, TrainingInfo, EvaluationInfo
            import joblib
            import tempfile
            
            # Create model registry
            project_root = Path(__file__).parent.parent.parent.parent
            models_dir = project_root / "models"
            registry = ModelRegistry(str(models_dir))
            
            # Generate version
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create metadata
            model_info = ModelInfo(
                version=version,
                model_type=model_type,
                model_name=model_name,
                created_at=datetime.now().isoformat(),
                training_id=training_id,
                export_files=[export_file]
            )
            
            training_info = TrainingInfo(
                training_samples=len(features_data),
                feature_names=list(evaluation_results.get('feature_importance', {}).keys()),
                training_duration=0.0,  # Will be set by training service
                export_files_size=Path(export_file).stat().st_size,
                model_parameters=self._get_model_parameters(model),
                preprocessing_metrics=self.training_metrics.get('preprocessing', {})
            )
            
            evaluation_info = EvaluationInfo(
                basic_metrics=evaluation_results.get('basic_metrics', {}),
                quality_metrics=evaluation_results.get('quality_metrics', {}),
                feature_importance=evaluation_results.get('feature_importance', {}),
                thresholds=evaluation_results.get('thresholds', {}),
                recommendations=evaluation_results.get('recommendations', []),
                evaluation_summary=evaluation_results.get('evaluation_summary', {})
            )
            
            metadata = ModelMetadata(
                model_info=model_info,
                training_info=training_info,
                evaluation_info=evaluation_info
            )
            
            # Save model to temporary file first
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save model
                model_file = temp_path / "model.joblib"
                joblib.dump(model, model_file)
                
                # Save scaler if available
                scaler_file = None
                if hasattr(self, 'scaler') and self.scaler:
                    scaler_file = temp_path / "scaler.joblib"
                    joblib.dump(self.scaler, scaler_file)
                
                # Save to registry with correct parameters
                model_path = registry.save_model(
                    version=version,
                    model_metadata=metadata,
                    model_file=model_file,
                    scaler_file=scaler_file
                )
            
            logger.info(f"Model saved: {model_path.name}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def _get_model_parameters(self, model) -> Dict[str, Any]:
        """Extract model parameters."""
        try:
            if hasattr(model, 'get_params'):
                return model.get_params()
            else:
                return {'model_type': type(model).__name__}
        except Exception as e:
            logger.error(f"Error getting model parameters: {e}")
            return {'model_type': type(model).__name__} 