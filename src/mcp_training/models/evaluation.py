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
        """Evaluate unsupervised anomaly detection model performance."""
        try:
            logger.info(f"Starting anomaly detection model evaluation for {X.shape[0]} samples")
            
            # Get anomaly scores (higher scores = more anomalous)
            scores = -model.score_samples(X)
            
            # For anomaly detection, we focus on score distribution and model behavior
            # Ground truth labels are optional and only used for validation if available
            has_ground_truth = y is not None and len(np.unique(y)) > 1
            
            if has_ground_truth:
                logger.info("Evaluation with ground truth labels available")
            else:
                logger.info("Unsupervised evaluation - no ground truth labels")
            
            # Calculate metrics focused on anomaly detection
            evaluation_results = {
                'evaluation_mode': 'unsupervised',
                'has_ground_truth': has_ground_truth,
                'basic_metrics': self._calculate_anomaly_metrics(scores, y),
                'advanced_metrics': self._calculate_advanced_anomaly_metrics(scores, y),
                'cross_validation': self._calculate_anomaly_cross_validation(model, X),
                'feature_importance': self._calculate_feature_importance(model, X),
                'performance_analysis': self._analyze_anomaly_performance(scores, y),
                'threshold_checks': {}  # Will be populated after basic metrics
            }
            
            # Check against anomaly detection thresholds
            evaluation_results['threshold_checks'] = self._check_anomaly_thresholds(evaluation_results)
            
            # Store metrics
            self.metrics = evaluation_results
            
            logger.info("Anomaly detection model evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating anomaly detection model: {e}")
            raise
    
    def _calculate_anomaly_metrics(self, scores: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate basic anomaly detection metrics."""
        # Use threshold to convert scores to predictions
        threshold = np.percentile(scores, 90)  # 90th percentile as threshold
        y_pred = (scores > threshold).astype(int)
        
        return {
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'score_min': float(np.min(scores)),
            'score_max': float(np.max(scores)),
            'score_range': float(np.max(scores) - np.min(scores)),
            'score_median': float(np.median(scores)),
            'anomaly_ratio': float(np.mean(y_pred)),
            'threshold_value': float(threshold),
            'detected_anomalies': int(np.sum(y_pred)),
            'total_samples': len(scores)
        }
    
    def _calculate_advanced_anomaly_metrics(self, scores: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced anomaly detection metrics."""
        threshold = np.percentile(scores, 90)
        y_pred = (scores > threshold).astype(int)
        
        # Check if we have ground truth for validation
        has_ground_truth = y is not None and len(np.unique(y)) > 1
        
        # Always calculate score distribution - this is the core of anomaly detection
        score_distribution = {
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
        
        # Calculate threshold analysis for anomaly detection
        threshold_analysis = self._analyze_anomaly_thresholds(scores, y)
        
        # Only calculate classification metrics if we have ground truth
        classification_metrics = {}
        if has_ground_truth:
            try:
                # Confusion matrix
                cm = confusion_matrix(y, y_pred)
                classification_metrics['confusion_matrix'] = cm.tolist()
                
                # Classification report
                report = classification_report(y, y_pred, output_dict=True)
                classification_metrics['classification_report'] = report
                
                # Additional anomaly detection specific metrics
                classification_metrics['roc_auc'] = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else None
                classification_metrics['average_precision'] = average_precision_score(y, scores) if len(np.unique(y)) > 1 else None
                
            except Exception as e:
                logger.warning(f"Classification metrics calculation failed: {e}")
                classification_metrics = {
                    'confusion_matrix': None,
                    'classification_report': None,
                    'roc_auc': None,
                    'average_precision': None
                }
        else:
            logger.info("No ground truth available - skipping classification metrics")
        
        return {
            'score_distribution': score_distribution,
            'threshold_analysis': threshold_analysis,
            'classification_metrics': classification_metrics,
            'anomaly_detection_metrics': {
                'detected_anomalies': int(np.sum(y_pred)),
                'total_samples': len(scores),
                'anomaly_ratio': float(np.mean(y_pred)),
                'score_separation': float(np.std(scores)),  # Higher separation is better for anomaly detection
                'score_skewness': float(self._calculate_skewness(scores))
            }
        }
    
    def _analyze_anomaly_thresholds(self, scores: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze anomaly detection performance across different thresholds."""
        thresholds = np.percentile(scores, [50, 75, 80, 85, 90, 95, 99])
        threshold_analysis = {}
        
        # Check if we have ground truth for validation
        has_ground_truth = y is not None and len(np.unique(y)) > 1
        
        for threshold in thresholds:
            y_pred = (scores > threshold).astype(int)
            
            # Always calculate anomaly ratio - this is the core metric for anomaly detection
            threshold_data = {
                'threshold': float(threshold),
                'anomaly_ratio': float(np.mean(y_pred)),
                'detected_anomalies': int(np.sum(y_pred)),
                'total_samples': len(scores)
            }
            
            # Add classification metrics if ground truth is available
            if has_ground_truth:
                try:
                    threshold_data.update({
                        'precision': float(precision_score(y, y_pred, zero_division=0)),
                        'recall': float(recall_score(y, y_pred, zero_division=0)),
                        'f1_score': float(f1_score(y, y_pred, zero_division=0))
                    })
                except Exception as e:
                    logger.warning(f"Threshold analysis failed for threshold {threshold}: {e}")
                    threshold_data.update({
                        'precision': None,
                        'recall': None,
                        'f1_score': None
                    })
            
            threshold_analysis[f'p{int(threshold)}'] = threshold_data
        
        return threshold_analysis
    
    def _calculate_anomaly_cross_validation(self, model: Any, X: np.ndarray) -> Dict[str, float]:
        """Calculate cross-validation scores for anomaly detection models."""
        try:
            # For anomaly detection models, we use unsupervised cross-validation
            # focusing on the consistency of score distributions across folds
            logger.info("Using unsupervised cross-validation for anomaly detection")
            
            cv_scores = cross_val_score(
                model, X, 
                cv=self.config.training.cross_validation_folds,
                scoring='neg_mean_squared_error'
            )
            
            return {
                'mean_cv_score': float(np.mean(cv_scores)),
                'std_cv_score': float(np.std(cv_scores)),
                'cv_scores': cv_scores.tolist(),
                'cv_consistency': float(1.0 - np.std(cv_scores) / abs(np.mean(cv_scores))) if np.mean(cv_scores) != 0 else 0.0
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {
                'mean_cv_score': None,
                'std_cv_score': None,
                'cv_scores': [],
                'cv_consistency': None
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
        """Calculate permutation importance for unsupervised models."""
        try:
            # For unsupervised models, we can't use standard permutation importance
            # Instead, we'll calculate feature importance based on how much the model's
            # predictions change when we shuffle each feature
            
            # Use a small sample for efficiency
            sample_size = min(1000, X.shape[0])
            sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[sample_indices]
            
            # Get baseline scores
            baseline_scores = model.score_samples(X_sample)
            
            # Calculate importance for each feature
            feature_importance = {}
            for feature_idx in range(X_sample.shape[1]):
                # Create a copy of the data with this feature shuffled
                X_shuffled = X_sample.copy()
                np.random.shuffle(X_shuffled[:, feature_idx])
                
                # Get scores with shuffled feature
                shuffled_scores = model.score_samples(X_shuffled)
                
                # Calculate importance as the difference in score variance
                # Higher difference means the feature is more important
                baseline_var = np.var(baseline_scores)
                shuffled_var = np.var(shuffled_scores)
                importance = abs(baseline_var - shuffled_var)
                
                feature_importance[feature_idx] = float(importance)
            
            # Normalize importance scores
            if feature_importance:
                max_importance = max(feature_importance.values())
                if max_importance > 0:
                    feature_importance = {k: v / max_importance for k, v in feature_importance.items()}
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Unsupervised feature importance calculation failed: {e}")
            # Return uniform importance as fallback
            return {i: 1.0 / X.shape[1] for i in range(X.shape[1])}
    
    def _analyze_anomaly_performance(self, scores: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze anomaly detection model performance characteristics."""
        # Check if we have ground truth for validation
        has_ground_truth = y is not None and len(np.unique(y)) > 1
        
        # Core anomaly detection analysis - always available
        anomaly_analysis = {
            'score_analysis': {
                'score_variance': float(np.var(scores)),
                'score_skewness': float(self._calculate_skewness(scores)),
                'score_range': float(np.max(scores) - np.min(scores)),
                'score_median': float(np.median(scores)),
                'score_iqr': float(np.percentile(scores, 75) - np.percentile(scores, 25))
            },
            'anomaly_detection_analysis': {
                'total_samples': len(scores),
                'score_distribution': {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'q25': float(np.percentile(scores, 25)),
                    'q75': float(np.percentile(scores, 75))
                },
                'anomaly_score_characteristics': {
                    'high_score_ratio': float(np.mean(scores > np.percentile(scores, 90))),
                    'low_score_ratio': float(np.mean(scores < np.percentile(scores, 10))),
                    'score_separation': float(np.std(scores))  # Higher is better for anomaly detection
                }
            }
        }
        
        # Add ground truth validation if available
        if has_ground_truth:
            # Calculate how well the scores separate normal from anomalous samples
            normal_scores = scores[y == 0]
            anomaly_scores = scores[y == 1]
            
            if len(normal_scores) > 0 and len(anomaly_scores) > 0:
                anomaly_analysis['ground_truth_validation'] = {
                    'normal_score_stats': {
                        'mean': float(np.mean(normal_scores)),
                        'std': float(np.std(normal_scores)),
                        'count': len(normal_scores)
                    },
                    'anomaly_score_stats': {
                        'mean': float(np.mean(anomaly_scores)),
                        'std': float(np.std(anomaly_scores)),
                        'count': len(anomaly_scores)
                    },
                    'score_separation_quality': {
                        'mean_difference': float(np.mean(anomaly_scores) - np.mean(normal_scores)),
                        'separation_ratio': float((np.mean(anomaly_scores) - np.mean(normal_scores)) / np.std(scores)) if np.std(scores) > 0 else 0.0
                    }
                }
        
        return anomaly_analysis
    
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
    
    def _check_anomaly_thresholds(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """Check if anomaly detection metrics meet quality thresholds."""
        basic_metrics = results['basic_metrics']
        checks = {}
        
        # Anomaly detection specific thresholds
        anomaly_thresholds = {
            'score_std': 0.01,  # Minimum score variance for meaningful separation
            'anomaly_ratio': 0.15,  # Maximum reasonable anomaly ratio
            'score_range': 0.1,  # Minimum score range for good separation
        }
        
        # Check score-based metrics (core for anomaly detection)
        for metric, threshold in anomaly_thresholds.items():
            if metric in basic_metrics:
                metric_value = basic_metrics[metric]
                if metric_value is None:
                    checks[metric] = False
                    logger.warning(f"Anomaly detection metric {metric}: None value (FAILED)")
                else:
                    # For anomaly_ratio, we want it to be below threshold (few anomalies)
                    if metric == 'anomaly_ratio':
                        checks[metric] = metric_value <= threshold
                    else:
                        checks[metric] = metric_value >= threshold
                    
                    status = "PASSED" if checks[metric] else "FAILED"
                    logger.info(f"Anomaly detection metric {metric}: {metric_value:.4f} (threshold: {threshold}) - {status}")
            else:
                checks[metric] = True  # Default to True if metric not available
                logger.info(f"Anomaly detection metric {metric}: not available (PASSED by default)")
        
        # Check score distribution quality
        if 'score_std' in basic_metrics and basic_metrics['score_std'] is not None:
            checks['score_distribution'] = basic_metrics['score_std'] > 0.01
            status = "PASSED" if checks['score_distribution'] else "FAILED"
            logger.info(f"Score distribution check: std={basic_metrics['score_std']:.4f} > 0.01 - {status}")
        else:
            checks['score_distribution'] = False
            logger.warning("Score distribution check: score_std not available (FAILED)")
        
        # Check if we have reasonable anomaly detection capability
        if 'score_range' in basic_metrics and basic_metrics['score_range'] is not None:
            checks['anomaly_detection_capability'] = basic_metrics['score_range'] > 0.1
            status = "PASSED" if checks['anomaly_detection_capability'] else "FAILED"
            logger.info(f"Anomaly detection capability: range={basic_metrics['score_range']:.4f} > 0.1 - {status}")
        else:
            checks['anomaly_detection_capability'] = False
            logger.warning("Anomaly detection capability: score_range not available (FAILED)")
        
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
        
        # Filter out None values for best/worst metric calculation
        valid_metrics = {k: v for k, v in basic_metrics.items() if v is not None}
        
        return {
            'overall_performance': {
                'passed_thresholds': passed_thresholds,
                'total_thresholds': total_thresholds,
                'threshold_pass_rate': passed_thresholds / total_thresholds if total_thresholds > 0 else 0.0
            },
            'best_metrics': {
                'best_metric': max(valid_metrics.items(), key=lambda x: x[1]) if valid_metrics else None,
                'worst_metric': min(valid_metrics.items(), key=lambda x: x[1]) if valid_metrics else None
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for anomaly detection models."""
        recommendations = []
        
        if not self.metrics:
            return recommendations
        
        basic_metrics = self.metrics.get('basic_metrics', {})
        threshold_checks = self.metrics.get('threshold_checks', {})
        has_ground_truth = self.metrics.get('has_ground_truth', False)
        
        # Anomaly detection specific recommendations
        score_std = basic_metrics.get('score_std')
        anomaly_ratio = basic_metrics.get('anomaly_ratio')
        score_range = basic_metrics.get('score_range')
        
        # Score distribution recommendations
        if score_std is not None and score_std < 0.05:
            recommendations.append("Low score variance suggests model may not be discriminating well - consider feature engineering or different model parameters")
        
        if score_range is not None and score_range < 0.2:
            recommendations.append("Small score range indicates limited separation capability - consider adjusting contamination parameter or using different features")
        
        # Anomaly ratio recommendations
        if anomaly_ratio is not None:
            if anomaly_ratio > 0.2:
                recommendations.append("High anomaly ratio suggests model may be too sensitive - consider decreasing contamination parameter")
            elif anomaly_ratio < 0.01:
                recommendations.append("Very low anomaly ratio suggests model may be too conservative - consider increasing contamination parameter")
        
        # Check threshold pass rate
        passed_thresholds = sum(threshold_checks.values())
        total_thresholds = len(threshold_checks)
        if total_thresholds > 0 and passed_thresholds / total_thresholds < 0.7:
            recommendations.append("Model performance below requirements - consider retraining with different parameters or more diverse data")
        
        # Ground truth validation recommendations
        if has_ground_truth:
            performance_analysis = self.metrics.get('performance_analysis', {})
            ground_truth_validation = performance_analysis.get('ground_truth_validation', {})
            
            if ground_truth_validation:
                separation_quality = ground_truth_validation.get('score_separation_quality', {})
                separation_ratio = separation_quality.get('separation_ratio', 0)
                
                if separation_ratio < 1.0:
                    recommendations.append("Low score separation between normal and anomalous samples - consider feature engineering or model tuning")
                elif separation_ratio > 5.0:
                    recommendations.append("Very high score separation - model may be overfitting, consider regularization")
        
        # Cross-validation recommendations
        cv_results = self.metrics.get('cross_validation', {})
        cv_consistency = cv_results.get('cv_consistency')
        if cv_consistency is not None and cv_consistency < 0.8:
            recommendations.append("Low cross-validation consistency suggests model instability - consider more training data or simpler model")
        
        # Feature importance recommendations
        feature_importance = self.metrics.get('feature_importance', {})
        if feature_importance:
            # Check if some features have very low importance
            importance_values = list(feature_importance.values())
            if len(importance_values) > 0:
                min_importance = min(importance_values)
                if min_importance < 0.01:
                    recommendations.append("Some features have very low importance - consider feature selection to improve model performance")
        
        return recommendations 