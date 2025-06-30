"""
Model evaluation for MCP Training Service.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.metrics import silhouette_score
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for unsupervised anomaly detection models."""
    
    def __init__(self, config=None):
        """Initialize model evaluator."""
        self.config = config
    
    def evaluate_model(self, model, X: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Evaluate an unsupervised anomaly detection model.
        
        Args:
            model: Trained model (e.g., IsolationForest)
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            logger.info(f"Starting model evaluation with {X.shape[0]} samples and {X.shape[1]} features")
            
            # Get anomaly scores
            scores = model.score_samples(X)
            logger.info(f"Calculated anomaly scores with range [{np.min(scores):.4f}, {np.max(scores):.4f}]")
            
            # Calculate basic metrics
            basic_metrics = self._calculate_basic_metrics(scores)
            logger.info(f"Calculated basic metrics: {len(basic_metrics)} metrics")
            
            # Calculate score distribution
            score_distribution = self._calculate_score_distribution(scores)
            logger.info("Calculated score distribution")
            
            # Calculate feature importance (if available)
            try:
                feature_importance = self._calculate_feature_importance(model, X, feature_names)
                logger.info(f"Calculated feature importance for {len(feature_importance)} features")
            except Exception as e:
                logger.warning(f"Feature importance calculation failed: {e}")
                feature_importance = {}
            
            # Calculate cross-validation score
            try:
                cross_validation_score = self._calculate_cross_validation_score(model, X)
                if cross_validation_score is not None:
                    logger.info(f"Cross-validation score: {cross_validation_score:.4f}")
                else:
                    logger.warning("Cross-validation score calculation returned None")
            except Exception as e:
                logger.warning(f"Cross-validation score calculation failed: {e}")
                cross_validation_score = None
            
            # Calculate thresholds and recommendations
            thresholds = self._calculate_thresholds(scores)
            logger.info(f"Calculated {len(thresholds)} threshold values")
            
            # Generate recommendations
            recommendations = self._generate_recommendations(basic_metrics, thresholds)
            logger.info(f"Generated {len(recommendations)} recommendations")
            
            evaluation_results = {
                'basic_metrics': basic_metrics,
                'score_distribution': score_distribution,
                'feature_importance': feature_importance,
                'cross_validation_score': cross_validation_score,
                'thresholds': thresholds,
                'recommendations': recommendations,
                'evaluation_summary': {
                    'total_samples': X.shape[0],
                    'total_features': X.shape[1],
                    'score_range': float(np.max(scores) - np.min(scores)),
                    'score_mean': float(np.mean(scores)),
                    'score_std': float(np.std(scores)),
                    'evaluation_timestamp': datetime.now().isoformat()
                }
            }
            
            logger.info("Model evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'basic_metrics': {},
                'score_distribution': {},
                'feature_importance': {},
                'cross_validation_score': None,
                'thresholds': {},
                'recommendations': [],
                'error': str(e),
                'error_type': type(e).__name__,
                'evaluation_summary': {
                    'error_occurred': True,
                    'error_message': str(e),
                    'evaluation_timestamp': datetime.now().isoformat()
                }
            }
    
    def _calculate_basic_metrics(self, scores: np.ndarray) -> Dict[str, float]:
        """Calculate basic metrics for anomaly scores."""
        try:
            return {
                'score_mean': float(np.mean(scores)),
                'score_std': float(np.std(scores)),
                'score_min': float(np.min(scores)),
                'score_max': float(np.max(scores)),
                'score_range': float(np.max(scores) - np.min(scores)),
                'score_median': float(np.median(scores)),
                'anomaly_ratio': 0.1,  # Default contamination
                'threshold_value': float(np.percentile(scores, 90)),  # 90th percentile
                'detected_anomalies': float(np.sum(scores < np.percentile(scores, 90))),
                'total_samples': float(len(scores))
            }
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            return {}
    
    def _calculate_score_distribution(self, scores: np.ndarray) -> Dict[str, Any]:
        """Calculate score distribution statistics."""
        try:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            distribution = {
                'percentiles': {f'p{p}': float(np.percentile(scores, p)) for p in percentiles},
                'histogram': {
                    'bins': 20,
                    'counts': np.histogram(scores, bins=20)[0].tolist(),
                    'bin_edges': np.histogram(scores, bins=20)[1].tolist()
                }
            }
            return distribution
        except Exception as e:
            logger.error(f"Error calculating score distribution: {e}")
            return {}
    
    def _calculate_feature_importance(self, model, X: np.ndarray, feature_names: List[str] = None) -> Dict[str, float]:
        """Calculate feature importance for unsupervised models."""
        try:
            if hasattr(model, 'feature_importances_'):
                # For models with built-in feature importance
                importances = model.feature_importances_
            elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                # For ensemble models, calculate permutation importance
                importances = self._calculate_permutation_importance(model, X)
            else:
                # Fallback: use variance-based importance
                importances = np.var(X, axis=0)
            
            if feature_names and len(feature_names) == len(importances):
                return dict(zip(feature_names, importances.tolist()))
            else:
                return {f'feature_{i}': float(imp) for i, imp in enumerate(importances)}
                
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def _calculate_permutation_importance(self, model, X: np.ndarray) -> np.ndarray:
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
            feature_importance = np.zeros(X_sample.shape[1])
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
                
                feature_importance[feature_idx] = importance
            
            # Normalize importance scores
            if np.max(feature_importance) > 0:
                feature_importance = feature_importance / np.max(feature_importance)
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {e}")
            # Fallback to variance-based importance
            return np.var(X, axis=0)
    
    def _calculate_cross_validation_score(self, model, X: np.ndarray) -> Optional[float]:
        """Calculate cross-validation score for unsupervised models."""
        try:
            from sklearn.model_selection import cross_val_score
            
            # Use silhouette score as a proxy for model quality
            def scoring_function(estimator, X, y=None):
                try:
                    scores = estimator.score_samples(X)
                    # Convert scores to cluster labels for silhouette calculation
                    # Use percentile-based clustering
                    threshold = np.percentile(scores, 90)
                    labels = (scores < threshold).astype(int)
                    if len(np.unique(labels)) > 1:
                        return silhouette_score(X, labels)
                    else:
                        return 0.0
                except:
                    return 0.0
            
            cv_scores = cross_val_score(
                model, X,
                scoring=scoring_function,
                cv=5,
                n_jobs=-1
            )
            
            return float(np.mean(cv_scores))
            
        except Exception as e:
            logger.error(f"Error calculating cross-validation score: {e}")
            return None
    
    def _calculate_thresholds(self, scores: np.ndarray) -> Dict[str, float]:
        """Calculate various threshold values for anomaly detection."""
        try:
            return {
                'conservative': float(np.percentile(scores, 95)),  # 95th percentile
                'moderate': float(np.percentile(scores, 90)),      # 90th percentile
                'aggressive': float(np.percentile(scores, 85)),    # 85th percentile
                'mean_plus_2std': float(np.mean(scores) + 2 * np.std(scores)),
                'mean_plus_3std': float(np.mean(scores) + 3 * np.std(scores))
            }
        except Exception as e:
            logger.error(f"Error calculating thresholds: {e}")
            return {}
    
    def _generate_recommendations(self, metrics: Dict[str, float], thresholds: Dict[str, float]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        try:
            # Check score distribution
            if metrics.get('score_std', 0) < 0.01:
                recommendations.append("Low score variance detected. Consider feature engineering or different model parameters.")
            
            if metrics.get('score_range', 0) < 0.1:
                recommendations.append("Limited score range. Model may not be discriminating well between normal and anomalous patterns.")
            
            # Check threshold recommendations
            if thresholds.get('moderate', 0) > 0.8:
                recommendations.append("High threshold detected. Consider lowering contamination parameter for more sensitive detection.")
            
            if metrics.get('detected_anomalies', 0) < 1:
                recommendations.append("No anomalies detected. Consider adjusting contamination parameter or feature selection.")
            
            # General recommendations
            recommendations.append("Monitor model performance over time and retrain with new data periodically.")
            recommendations.append("Consider ensemble methods for improved robustness.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to evaluation errors.")
        
        return recommendations 