"""
Comprehensive model evaluation system for MCP Training Service.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import logging

from .config import ModelConfig

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the model evaluator."""
        self.config = config
        self.metrics = {}
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Evaluate model performance comprehensively."""
        try:
            logger.info(f"Starting comprehensive model evaluation for {X.shape[0]} samples")
            
            # For unsupervised learning, create pseudo-labels
            if y is None:
                y_pred = model.predict(X)
                y = (y_pred == -1).astype(int)  # Convert to binary labels
            
            # Get anomaly scores
            scores = -model.score_samples(X)
            
            # Calculate all metrics
            evaluation_results = {
                'basic_metrics': self._calculate_basic_metrics(y, scores),
                'advanced_metrics': self._calculate_advanced_metrics(y, scores),
                'cross_validation': self._calculate_cross_validation(model, X, y),
                'feature_importance': self._calculate_feature_importance(model, X),
                'performance_analysis': self._analyze_performance(y, scores),
                'threshold_checks': {}  # Will be populated after basic metrics
            }
            
            # Check against thresholds
            evaluation_results['threshold_checks'] = self._check_thresholds(evaluation_results)
            
            # Store metrics
            self.metrics = evaluation_results
            
            logger.info("Model evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def _calculate_basic_metrics(self, y: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        # Use threshold to convert scores to predictions
        threshold = np.percentile(scores, 90)  # 90th percentile as threshold
        y_pred = (scores > threshold).astype(int)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, scores),
            'average_precision': average_precision_score(y, scores)
        }
    
    def _calculate_advanced_metrics(self, y: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced evaluation metrics."""
        threshold = np.percentile(scores, 90)
        y_pred = (scores > threshold).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        return {
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'threshold_analysis': self._analyze_thresholds(y, scores),
            'score_distribution': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'percentiles': {
                    '25': float(np.percentile(scores, 25)),
                    '50': float(np.percentile(scores, 50)),
                    '75': float(np.percentile(scores, 75)),
                    '90': float(np.percentile(scores, 90)),
                    '95': float(np.percentile(scores, 95)),
                    '99': float(np.percentile(scores, 99))
                }
            }
        }
    
    def _analyze_thresholds(self, y: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """Analyze performance across different thresholds."""
        thresholds = np.percentile(scores, [50, 75, 80, 85, 90, 95, 99])
        threshold_analysis = {}
        
        for threshold in thresholds:
            y_pred = (scores > threshold).astype(int)
            threshold_analysis[f'p{int(threshold)}'] = {
                'threshold': float(threshold),
                'precision': float(precision_score(y, y_pred, zero_division=0)),
                'recall': float(recall_score(y, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y, y_pred, zero_division=0)),
                'anomaly_ratio': float(np.mean(y_pred))
            }
        
        return threshold_analysis
    
    def _calculate_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate cross-validation scores."""
        try:
            cv_scores = cross_val_score(
                model, X, y, 
                cv=self.config.training.cross_validation_folds,
                scoring='neg_mean_squared_error'
            )
            
            return {
                'mean_cv_score': float(np.mean(cv_scores)),
                'std_cv_score': float(np.std(cv_scores)),
                'cv_scores': cv_scores.tolist()
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {
                'mean_cv_score': None,
                'std_cv_score': None,
                'cv_scores': []
            }
    
    def _calculate_feature_importance(self, model: Any, X: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for Isolation Forest."""
        try:
            # For Isolation Forest, we can use feature importances if available
            if hasattr(model, 'feature_importances_'):
                return dict(zip(range(X.shape[1]), model.feature_importances_))
            else:
                # Fallback: use permutation importance
                return self._calculate_permutation_importance(model, X)
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {}
    
    def _calculate_permutation_importance(self, model: Any, X: np.ndarray) -> Dict[str, float]:
        """Calculate permutation importance as fallback."""
        try:
            # Use a small sample for permutation importance to avoid long computation
            sample_size = min(1000, X.shape[0])
            sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[sample_indices]
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_sample, 
                n_repeats=5, 
                random_state=42,
                scoring='neg_mean_squared_error'
            )
            
            return dict(zip(range(X.shape[1]), perm_importance.importances_mean))
        except Exception as e:
            logger.warning(f"Permutation importance calculation failed: {e}")
            return {}
    
    def _analyze_performance(self, y: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """Analyze model performance characteristics."""
        return {
            'score_analysis': {
                'score_correlation_with_labels': float(np.corrcoef(y, scores)[0, 1]) if len(y) > 1 else 0.0,
                'score_separation': float(np.mean(scores[y == 1]) - np.mean(scores[y == 0])) if len(np.unique(y)) > 1 else 0.0,
                'score_variance': float(np.var(scores)),
                'score_skewness': float(self._calculate_skewness(scores))
            },
            'anomaly_detection_analysis': {
                'detected_anomalies': int(np.sum(y)),
                'total_samples': len(y),
                'anomaly_ratio': float(np.mean(y)),
                'score_distribution_by_class': {
                    'normal_mean': float(np.mean(scores[y == 0])) if np.sum(y == 0) > 0 else 0.0,
                    'anomaly_mean': float(np.mean(scores[y == 1])) if np.sum(y == 1) > 0 else 0.0,
                    'normal_std': float(np.std(scores[y == 0])) if np.sum(y == 0) > 0 else 0.0,
                    'anomaly_std': float(np.std(scores[y == 1])) if np.sum(y == 1) > 0 else 0.0
                }
            }
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return float(np.mean(((data - mean) / std) ** 3))
        except:
            return 0.0
    
    def _check_thresholds(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """Check if metrics meet configured thresholds."""
        thresholds = self.config.evaluation.thresholds
        basic_metrics = results['basic_metrics']
        
        checks = {}
        for metric, threshold in thresholds.items():
            if metric in basic_metrics:
                checks[metric] = basic_metrics[metric] >= threshold
            else:
                checks[metric] = True  # Default to True if metric not available
        
        return checks
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        if not self.metrics:
            return {}
        
        basic_metrics = self.metrics.get('basic_metrics', {})
        threshold_checks = self.metrics.get('threshold_checks', {})
        
        # Count passed thresholds
        passed_thresholds = sum(threshold_checks.values())
        total_thresholds = len(threshold_checks)
        
        return {
            'overall_performance': {
                'passed_thresholds': passed_thresholds,
                'total_thresholds': total_thresholds,
                'threshold_pass_rate': passed_thresholds / total_thresholds if total_thresholds > 0 else 0.0
            },
            'best_metrics': {
                'best_metric': max(basic_metrics.items(), key=lambda x: x[1]) if basic_metrics else None,
                'worst_metric': min(basic_metrics.items(), key=lambda x: x[1]) if basic_metrics else None
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        if not self.metrics:
            return recommendations
        
        basic_metrics = self.metrics.get('basic_metrics', {})
        threshold_checks = self.metrics.get('threshold_checks', {})
        
        # Check precision
        if basic_metrics.get('precision', 0) < 0.7:
            recommendations.append("Consider increasing contamination parameter to reduce false positives")
        
        # Check recall
        if basic_metrics.get('recall', 0) < 0.7:
            recommendations.append("Consider decreasing contamination parameter to catch more anomalies")
        
        # Check F1 score
        if basic_metrics.get('f1_score', 0) < 0.7:
            recommendations.append("Model may need hyperparameter tuning for better balance")
        
        # Check ROC AUC
        if basic_metrics.get('roc_auc', 0) < 0.8:
            recommendations.append("Consider feature engineering or model selection for better discrimination")
        
        # Check threshold pass rate
        passed_thresholds = sum(threshold_checks.values())
        total_thresholds = len(threshold_checks)
        if total_thresholds > 0 and passed_thresholds / total_thresholds < 0.5:
            recommendations.append("Model performance below requirements - consider retraining with different parameters")
        
        # Check score distribution
        score_dist = self.metrics.get('advanced_metrics', {}).get('score_distribution', {})
        if score_dist.get('std', 0) < 0.1:
            recommendations.append("Low score variance suggests model may not be discriminating well")
        
        return recommendations 