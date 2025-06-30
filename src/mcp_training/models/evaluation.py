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
            
            # Calculate model quality metrics
            quality_metrics = self._calculate_quality_metrics(model, X, scores)
            logger.info(f"Calculated quality metrics: {len(quality_metrics)} metrics")
            
            evaluation_results = {
                'basic_metrics': basic_metrics,
                'score_distribution': score_distribution,
                'feature_importance': feature_importance,
                'cross_validation_score': cross_validation_score,
                'thresholds': thresholds,
                'recommendations': recommendations,
                'quality_metrics': quality_metrics,
                'evaluation_summary': {
                    'total_samples': X.shape[0],
                    'total_features': X.shape[1],
                    'score_range': float(np.max(scores) - np.min(scores)),
                    'score_mean': float(np.mean(scores)),
                    'score_std': float(np.std(scores)),
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'model_quality_score': self._calculate_overall_quality_score(quality_metrics, basic_metrics)
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
                'quality_metrics': {},
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
                'total_samples': float(len(scores)),
                'score_variance': float(np.var(scores)),
                'score_skewness': float(self._calculate_skewness(scores)),
                'score_kurtosis': float(self._calculate_kurtosis(scores))
            }
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            return {}
    
    def _calculate_skewness(self, scores: np.ndarray) -> float:
        """Calculate skewness of scores."""
        try:
            mean = np.mean(scores)
            std = np.std(scores)
            if std == 0:
                return 0.0
            return float(np.mean(((scores - mean) / std) ** 3))
        except:
            return 0.0
    
    def _calculate_kurtosis(self, scores: np.ndarray) -> float:
        """Calculate kurtosis of scores."""
        try:
            mean = np.mean(scores)
            std = np.std(scores)
            if std == 0:
                return 0.0
            return float(np.mean(((scores - mean) / std) ** 4) - 3)
        except:
            return 0.0
    
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
                },
                'iqr': float(np.percentile(scores, 75) - np.percentile(scores, 25)),
                'outliers': self._detect_outliers(scores)
            }
            return distribution
        except Exception as e:
            logger.error(f"Error calculating score distribution: {e}")
            return {}
    
    def _detect_outliers(self, scores: np.ndarray) -> Dict[str, Any]:
        """Detect outliers in scores using IQR method."""
        try:
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = scores[(scores < lower_bound) | (scores > upper_bound)]
            
            return {
                'count': int(len(outliers)),
                'percentage': float(len(outliers) / len(scores) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
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
            
            # Custom scoring function for unsupervised learning
            def scoring_function(estimator, X, y=None):
                try:
                    scores = estimator.score_samples(X)
                    # Use negative mean absolute deviation as scoring
                    return -np.mean(np.abs(scores - np.mean(scores)))
                except:
                    return 0.0
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, scoring=scoring_function, cv=5)
            
            if len(cv_scores) > 0:
                return float(np.mean(cv_scores))
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error calculating cross-validation score: {e}")
            return None
    
    def _calculate_thresholds(self, scores: np.ndarray) -> Dict[str, float]:
        """Calculate various threshold values for anomaly detection."""
        try:
            return {
                'p90_threshold': float(np.percentile(scores, 90)),
                'p95_threshold': float(np.percentile(scores, 95)),
                'p99_threshold': float(np.percentile(scores, 99)),
                'mean_plus_2std': float(np.mean(scores) + 2 * np.std(scores)),
                'mean_plus_3std': float(np.mean(scores) + 3 * np.std(scores)),
                'iqr_upper': float(np.percentile(scores, 75) + 1.5 * (np.percentile(scores, 75) - np.percentile(scores, 25)))
            }
        except Exception as e:
            logger.error(f"Error calculating thresholds: {e}")
            return {}
    
    def _calculate_quality_metrics(self, model, X: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Calculate model quality metrics."""
        try:
            # Calculate silhouette score if possible
            try:
                # For anomaly detection, we can use the scores to create clusters
                # Use percentile-based clustering
                threshold = np.percentile(scores, 90)
                labels = (scores < threshold).astype(int)
                silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
            except:
                silhouette = 0.0
            
            # Calculate score stability
            score_stability = 1.0 - (np.std(scores) / (np.max(scores) - np.min(scores) + 1e-8))
            
            # Calculate feature utilization
            if hasattr(model, 'feature_importances_'):
                feature_utilization = np.sum(model.feature_importances_ > 0.01) / len(model.feature_importances_)
            else:
                feature_utilization = 1.0
            
            return {
                'silhouette_score': float(silhouette),
                'score_stability': float(score_stability),
                'feature_utilization': float(feature_utilization),
                'score_diversity': float(np.std(scores)),
                'model_complexity': float(X.shape[1])  # Number of features
            }
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {}
    
    def _calculate_overall_quality_score(self, quality_metrics: Dict[str, float], basic_metrics: Dict[str, float]) -> float:
        """Calculate overall model quality score."""
        try:
            scores = []
            
            # Silhouette score (0-1, higher is better)
            if 'silhouette_score' in quality_metrics:
                scores.append(quality_metrics['silhouette_score'])
            
            # Score stability (0-1, higher is better)
            if 'score_stability' in quality_metrics:
                scores.append(quality_metrics['score_stability'])
            
            # Feature utilization (0-1, higher is better)
            if 'feature_utilization' in quality_metrics:
                scores.append(quality_metrics['feature_utilization'])
            
            # Score diversity (normalized, moderate is better)
            if 'score_diversity' in quality_metrics and 'score_range' in basic_metrics:
                diversity_ratio = quality_metrics['score_diversity'] / (basic_metrics['score_range'] + 1e-8)
                # Penalize very low or very high diversity
                if diversity_ratio < 0.1 or diversity_ratio > 0.9:
                    scores.append(0.5)
                else:
                    scores.append(1.0)
            
            if scores:
                return float(np.mean(scores))
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            logger.error(f"Error calculating overall quality score: {e}")
            return 0.5
    
    def _generate_recommendations(self, metrics: Dict[str, float], thresholds: Dict[str, float]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        try:
            # Check score distribution
            if 'score_std' in metrics and metrics['score_std'] < 0.01:
                recommendations.append("Low score variance detected. Consider using more diverse features or adjusting model parameters.")
            
            # Check anomaly ratio
            if 'anomaly_ratio' in metrics and metrics['anomaly_ratio'] > 0.2:
                recommendations.append("High anomaly ratio detected. Consider adjusting contamination parameter or reviewing data quality.")
            
            # Check feature utilization
            if 'feature_utilization' in metrics and metrics['feature_utilization'] < 0.5:
                recommendations.append("Low feature utilization detected. Consider feature selection or engineering additional features.")
            
            # Check score stability
            if 'score_stability' in metrics and metrics['score_stability'] < 0.7:
                recommendations.append("Low score stability detected. Model may be sensitive to data variations.")
            
            # Add general recommendations
            if len(recommendations) == 0:
                recommendations.append("Model evaluation completed successfully. Consider monitoring performance on new data.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to evaluation errors.")
        
        return recommendations 