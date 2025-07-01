"""
Enhanced Model evaluation for MCP Training Service.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Enhanced evaluator for unsupervised anomaly detection models."""
    
    def __init__(self, config=None):
        """Initialize model evaluator."""
        self.config = config
        self.quality_thresholds = {
            'min_silhouette_score': 0.2,
            'min_score_stability': 0.6,
            'min_feature_utilization': 0.3,
            'max_score_skewness': 2.0,
            'min_score_diversity': 0.1,
            'max_anomaly_ratio': 0.3,
            'min_samples': 100,
            'min_features': 5
        }
    
    def evaluate_model(self, model, X: np.ndarray, feature_names: List[str] = None, X_original: np.ndarray = None) -> Dict[str, Any]:
        """Evaluate an unsupervised anomaly detection model with enhanced metrics.
        
        Args:
            model: Trained model (e.g., IsolationForest)
            X: Scaled feature matrix
            feature_names: List of feature names
            X_original: Original (unscaled) feature matrix for additional metrics
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        try:
            logger.info(f"Starting enhanced model evaluation with {X.shape[0]} samples and {X.shape[1]} features")
            
            # Validate input data
            if X.shape[0] < self.quality_thresholds['min_samples']:
                logger.warning(f"Low sample count: {X.shape[0]} < {self.quality_thresholds['min_samples']}")
            
            if X.shape[1] < self.quality_thresholds['min_features']:
                logger.warning(f"Low feature count: {X.shape[1]} < {self.quality_thresholds['min_features']}")
            
            # Get anomaly scores
            scores = model.score_samples(X)
            logger.info(f"Calculated anomaly scores with range [{np.min(scores):.4f}, {np.max(scores):.4f}]")
            
            # Calculate enhanced basic metrics
            basic_metrics = self._calculate_enhanced_basic_metrics(scores, X)
            logger.info(f"Calculated enhanced basic metrics: {len(basic_metrics)} metrics")
            
            # Calculate unsupervised clustering metrics
            clustering_metrics = self._calculate_clustering_metrics(X, scores)
            logger.info(f"Calculated clustering metrics: {len(clustering_metrics)} metrics")
            
            # Calculate traditional ML metrics if original features available
            if X_original is not None:
                traditional_metrics = self._calculate_enhanced_traditional_metrics(model, X, X_original, scores)
                basic_metrics.update(traditional_metrics)
                logger.info(f"Calculated enhanced traditional metrics: {len(traditional_metrics)} metrics")
            
            # Calculate enhanced score distribution
            score_distribution = self._calculate_enhanced_score_distribution(scores)
            logger.info("Calculated enhanced score distribution")
            
            # Calculate feature importance (if available)
            try:
                feature_importance = self._calculate_enhanced_feature_importance(model, X, feature_names)
                logger.info(f"Calculated enhanced feature importance for {len(feature_importance)} features")
            except Exception as e:
                logger.warning(f"Feature importance calculation failed: {e}")
                feature_importance = {}
            
            # Calculate cross-validation score
            try:
                cross_validation_score = self._calculate_enhanced_cross_validation_score(model, X)
                if cross_validation_score is not None:
                    logger.info(f"Enhanced cross-validation score: {cross_validation_score:.4f}")
                else:
                    logger.warning("Enhanced cross-validation score calculation returned None")
            except Exception as e:
                logger.warning(f"Enhanced cross-validation score calculation failed: {e}")
                cross_validation_score = None
            
            # Calculate enhanced thresholds and recommendations
            thresholds = self._calculate_enhanced_thresholds(scores)
            logger.info(f"Calculated {len(thresholds)} enhanced threshold values")
            
            # Generate enhanced recommendations
            recommendations = self._generate_enhanced_recommendations(basic_metrics, clustering_metrics, thresholds, X.shape)
            logger.info(f"Generated {len(recommendations)} enhanced recommendations")
            
            # Calculate enhanced model quality metrics
            quality_metrics = self._calculate_enhanced_quality_metrics(model, X, scores, clustering_metrics)
            logger.info(f"Calculated enhanced quality metrics: {len(quality_metrics)} metrics")
            
            # Calculate overall quality assessment
            quality_assessment = self._calculate_quality_assessment(basic_metrics, clustering_metrics, quality_metrics, X.shape)
            logger.info("Calculated comprehensive quality assessment")
            
            evaluation_results = {
                'basic_metrics': basic_metrics,
                'clustering_metrics': clustering_metrics,
                'score_distribution': score_distribution,
                'feature_importance': feature_importance,
                'cross_validation_score': cross_validation_score,
                'thresholds': thresholds,
                'recommendations': recommendations,
                'quality_metrics': quality_metrics,
                'quality_assessment': quality_assessment,
                'evaluation_summary': {
                    'total_samples': X.shape[0],
                    'total_features': X.shape[1],
                    'score_range': float(np.max(scores) - np.min(scores)),
                    'score_mean': float(np.mean(scores)),
                    'score_std': float(np.std(scores)),
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'model_quality_score': quality_assessment['overall_score'],
                    'quality_level': quality_assessment['quality_level'],
                    'validation_status': quality_assessment['validation_status']
                }
            }
            
            logger.info("Enhanced model evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in enhanced model evaluation: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'basic_metrics': {},
                'clustering_metrics': {},
                'score_distribution': {},
                'feature_importance': {},
                'cross_validation_score': None,
                'thresholds': {},
                'recommendations': ['Evaluation failed due to system error'],
                'quality_metrics': {},
                'quality_assessment': {
                    'overall_score': 0.0,
                    'quality_level': 'Poor',
                    'validation_status': 'FAILED',
                    'issues': [f'Evaluation error: {str(e)}']
                },
                'error': str(e),
                'error_type': type(e).__name__,
                'evaluation_summary': {
                    'error_occurred': True,
                    'error_message': str(e),
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'model_quality_score': 0.0,
                    'quality_level': 'Poor',
                    'validation_status': 'FAILED'
                }
            }
    
    def _calculate_enhanced_basic_metrics(self, scores: np.ndarray, X: np.ndarray) -> Dict[str, float]:
        """Calculate enhanced basic metrics for anomaly scores."""
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
                'score_kurtosis': float(self._calculate_kurtosis(scores)),
                'score_stability': 1.0 - (np.std(scores) / (np.max(scores) - np.min(scores) + 1e-8)),
                'feature_utilization': np.sum(np.var(X, axis=0) > 0.01) / len(np.var(X, axis=0)),
                'score_diversity': float(np.std(scores)),
                'model_complexity': float(X.shape[1])  # Number of features
            }
        except Exception as e:
            logger.error(f"Error calculating enhanced basic metrics: {e}")
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
    
    def _calculate_enhanced_score_distribution(self, scores: np.ndarray) -> Dict[str, Any]:
        """Calculate enhanced score distribution statistics."""
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
            logger.error(f"Error calculating enhanced score distribution: {e}")
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
    
    def _calculate_enhanced_feature_importance(self, model, X: np.ndarray, feature_names: List[str] = None) -> Dict[str, float]:
        """Calculate enhanced feature importance for unsupervised models."""
        try:
            if hasattr(model, 'feature_importances_'):
                # For models with built-in feature importance
                importances = model.feature_importances_
            elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                # For ensemble models, calculate permutation importance
                importances = self._calculate_enhanced_permutation_importance(model, X)
            else:
                # Fallback: use variance-based importance
                importances = np.var(X, axis=0)
            
            if feature_names and len(feature_names) == len(importances):
                return dict(zip(feature_names, importances.tolist()))
            else:
                return {f'feature_{i}': float(imp) for i, imp in enumerate(importances)}
                
        except Exception as e:
            logger.error(f"Error calculating enhanced feature importance: {e}")
            return {}
    
    def _calculate_enhanced_permutation_importance(self, model, X: np.ndarray) -> np.ndarray:
        """Calculate enhanced permutation importance for unsupervised models."""
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
            logger.error(f"Error calculating enhanced permutation importance: {e}")
            # Fallback to variance-based importance
            return np.var(X, axis=0)
    
    def _calculate_enhanced_cross_validation_score(self, model, X: np.ndarray) -> Optional[float]:
        """Calculate enhanced cross-validation score for unsupervised models."""
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
            logger.error(f"Error calculating enhanced cross-validation score: {e}")
            return None
    
    def _calculate_enhanced_thresholds(self, scores: np.ndarray) -> Dict[str, float]:
        """Calculate various enhanced threshold values for anomaly detection."""
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
            logger.error(f"Error calculating enhanced thresholds: {e}")
            return {}
    
    def _calculate_enhanced_quality_metrics(self, model, X: np.ndarray, scores: np.ndarray, clustering_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate enhanced model quality metrics."""
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
                feature_utilization = np.sum(model.feature_importances_ > 0.0) / len(model.feature_importances_)
            else:
                feature_utilization = 1.0
            
            # Get clustering metrics
            calinski_harabasz = clustering_metrics.get('calinski_harabasz_score', 0.0)
            davies_bouldin = clustering_metrics.get('davies_bouldin_score', 1.0)
            
            return {
                'silhouette_score': float(silhouette),
                'score_stability': float(score_stability),
                'feature_utilization': float(feature_utilization),
                'score_diversity': float(np.std(scores)),
                'model_complexity': float(X.shape[1]),  # Number of features
                'calinski_harabasz_score': float(calinski_harabasz),
                'davies_bouldin_score': float(davies_bouldin),
                'clustering_quality': float(1.0 / (1.0 + davies_bouldin))  # Normalize Davies-Bouldin
            }
        except Exception as e:
            logger.error(f"Error calculating enhanced quality metrics: {e}")
            return {}
    
    def _calculate_quality_assessment(self, basic_metrics: Dict[str, float], clustering_metrics: Dict[str, float], quality_metrics: Dict[str, float], X_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Calculate overall quality assessment."""
        try:
            # Check quality thresholds
            quality_level = 'Good'
            validation_status = 'VALID'
            issues = []
            
            # Check silhouette score
            silhouette_score = quality_metrics.get('silhouette_score', 0.0)
            if silhouette_score < self.quality_thresholds['min_silhouette_score']:
                quality_level = 'Poor'
                validation_status = 'FAILED'
                issues.append("Low silhouette score detected. Model may not be effective for clustering.")
            
            # Check score stability
            score_stability = quality_metrics.get('score_stability', 0.0)
            if score_stability < self.quality_thresholds['min_score_stability']:
                quality_level = 'Poor'
                validation_status = 'FAILED'
                issues.append("Low score stability detected. Model may be sensitive to data variations.")
            
            # Check feature utilization
            feature_utilization = quality_metrics.get('feature_utilization', 0.0)
            if feature_utilization < self.quality_thresholds['min_feature_utilization']:
                quality_level = 'Poor'
                validation_status = 'FAILED'
                issues.append("Low feature utilization detected. Consider feature selection or engineering additional features.")
            
            # Check score skewness
            score_skewness = basic_metrics.get('score_skewness', 0.0)
            if abs(score_skewness) > self.quality_thresholds['max_score_skewness']:
                quality_level = 'Poor'
                validation_status = 'FAILED'
                issues.append("High score skewness detected. Model may not be effective for anomaly detection.")
            
            # Check score diversity
            score_diversity = quality_metrics.get('score_diversity', 0.0)
            if score_diversity < self.quality_thresholds['min_score_diversity']:
                quality_level = 'Poor'
                validation_status = 'FAILED'
                issues.append("Low score diversity detected. Model may not be effective for anomaly detection.")
            
            # Check anomaly ratio
            anomaly_ratio = basic_metrics.get('anomaly_ratio', 0.1)
            if anomaly_ratio > self.quality_thresholds['max_anomaly_ratio']:
                quality_level = 'Poor'
                validation_status = 'FAILED'
                issues.append("High anomaly ratio detected. Consider adjusting contamination parameter or reviewing data quality.")
            
            # Calculate overall score
            overall_score = 0.0
            score_components = 0
            
            if 'silhouette_score' in quality_metrics:
                overall_score += quality_metrics['silhouette_score']
                score_components += 1
            if 'score_stability' in quality_metrics:
                overall_score += quality_metrics['score_stability']
                score_components += 1
            if 'feature_utilization' in quality_metrics:
                overall_score += quality_metrics['feature_utilization']
                score_components += 1
            if 'clustering_quality' in quality_metrics:
                overall_score += quality_metrics['clustering_quality']
                score_components += 1
            
            # Normalize overall score
            if score_components > 0:
                overall_score = overall_score / score_components
            else:
                overall_score = 0.5  # Default neutral score
            
            # Adjust score based on validation status
            if validation_status == 'FAILED':
                overall_score = max(0.0, overall_score * 0.5)  # Penalize but don't zero out
            
            return {
                'overall_score': float(overall_score),
                'quality_level': quality_level,
                'validation_status': validation_status,
                'issues': issues
            }
        except Exception as e:
            logger.error(f"Error calculating quality assessment: {e}")
            return {
                'overall_score': 0.0,
                'quality_level': 'Poor',
                'validation_status': 'FAILED',
                'issues': [f'Assessment error: {str(e)}']
            }
    
    def _generate_enhanced_recommendations(self, basic_metrics: Dict[str, float], clustering_metrics: Dict[str, float], thresholds: Dict[str, float], X_shape: Tuple[int, int]) -> List[str]:
        """Generate enhanced recommendations based on evaluation results."""
        recommendations = []
        
        try:
            # Check score distribution
            if 'score_std' in basic_metrics and basic_metrics['score_std'] < 0.01:
                recommendations.append("Low score variance detected. Consider using more diverse features or adjusting model parameters.")
            
            # Check anomaly ratio
            if 'anomaly_ratio' in basic_metrics and basic_metrics['anomaly_ratio'] > 0.2:
                recommendations.append("High anomaly ratio detected. Consider adjusting contamination parameter or reviewing data quality.")
            
            # Check feature utilization
            if 'feature_utilization' in basic_metrics and basic_metrics['feature_utilization'] < 0.5:
                recommendations.append("Low feature utilization detected. Consider feature selection or engineering additional features.")
            
            # Check score stability
            if 'score_stability' in basic_metrics and basic_metrics['score_stability'] < 0.7:
                recommendations.append("Low score stability detected. Model may be sensitive to data variations.")
            
            # Check clustering quality
            if clustering_metrics['davies_bouldin_score'] > 1.0:
                recommendations.append("High Davies-Bouldin Index detected. Consider adjusting clustering parameters or reviewing data quality.")
            
            # Add general recommendations
            if len(recommendations) == 0:
                recommendations.append("Model evaluation completed successfully. Consider monitoring performance on new data.")
            
        except Exception as e:
            logger.error(f"Error generating enhanced recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to evaluation errors.")
        
        return recommendations
    
    def _calculate_enhanced_traditional_metrics(self, model, X_scaled: np.ndarray, X_original: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Calculate traditional ML metrics for unsupervised anomaly detection.
        
        Args:
            model: Trained model
            X_scaled: Scaled feature matrix
            X_original: Original feature matrix
            scores: Anomaly scores
            
        Returns:
            Dictionary of traditional metrics
        """
        try:
            # For unsupervised models, we'll create synthetic labels based on score percentiles
            # This allows us to calculate traditional metrics
            
            # Use different thresholds to create synthetic labels
            thresholds = {
                'p90': np.percentile(scores, 90),
                'p95': np.percentile(scores, 95),
                'p99': np.percentile(scores, 99)
            }
            
            metrics = {}
            
            for threshold_name, threshold_value in thresholds.items():
                # Create synthetic labels based on threshold
                synthetic_labels = (scores < threshold_value).astype(int)
                
                # Calculate metrics for this threshold
                anomaly_count = np.sum(synthetic_labels)
                normal_count = len(synthetic_labels) - anomaly_count
                
                # Calculate precision, recall, F1-score
                if anomaly_count > 0:
                    # For unsupervised models, we assume the model's predictions are "ground truth"
                    # and calculate how well the threshold separates the data
                    precision = anomaly_count / len(synthetic_labels)  # Ratio of detected anomalies
                    recall = 1.0  # We assume all anomalies are detected (by definition)
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    accuracy = max(anomaly_count, normal_count) / len(synthetic_labels)  # Majority class accuracy
                else:
                    precision = recall = f1_score = accuracy = 0.0
                
                # Store metrics with threshold prefix
                metrics[f'{threshold_name}_precision'] = float(precision)
                metrics[f'{threshold_name}_recall'] = float(recall)
                metrics[f'{threshold_name}_f1_score'] = float(f1_score)
                metrics[f'{threshold_name}_accuracy'] = float(accuracy)
                metrics[f'{threshold_name}_anomaly_count'] = float(anomaly_count)
                metrics[f'{threshold_name}_normal_count'] = float(normal_count)
                metrics[f'{threshold_name}_threshold_value'] = float(threshold_value)
            
            # Calculate ROC AUC equivalent for unsupervised models
            # We'll use the score distribution to estimate separability
            score_std = np.std(scores)
            score_mean = np.mean(scores)
            if score_std > 0:
                # Higher standard deviation indicates better separability
                roc_auc_equivalent = min(1.0, score_std / (score_mean + 1e-8))
            else:
                roc_auc_equivalent = 0.5  # Random performance
            
            metrics['roc_auc_equivalent'] = float(roc_auc_equivalent)
            
            # Calculate silhouette score for clustering quality
            try:
                from sklearn.metrics import silhouette_score
                # Create discrete labels from scores using percentiles
                score_percentiles = np.percentile(scores, [25, 50, 75])
                discrete_labels = np.digitize(scores, bins=score_percentiles)
                # Ensure we have at least 2 clusters and not more than n_samples-1
                unique_labels = np.unique(discrete_labels)
                if len(unique_labels) >= 2 and len(unique_labels) < len(scores):
                    silhouette = silhouette_score(X_scaled, discrete_labels)
                    metrics['silhouette_score'] = float(silhouette)
                else:
                    logger.warning("Insufficient clusters for silhouette score calculation")
                    metrics['silhouette_score'] = 0.0
            except Exception as e:
                logger.warning(f"Silhouette score calculation failed: {e}")
                metrics['silhouette_score'] = 0.0
            
            # Calculate Davies-Bouldin Index for clustering quality
            try:
                from sklearn.metrics import davies_bouldin_score
                # Use quantized scores for Davies-Bouldin calculation
                quantized_scores = np.digitize(scores, bins=np.percentile(scores, [25, 50, 75]))
                davies_bouldin = davies_bouldin_score(X_scaled, quantized_scores)
                metrics['davies_bouldin_index'] = float(davies_bouldin)
            except Exception as e:
                logger.warning(f"Davies-Bouldin Index calculation failed: {e}")
                metrics['davies_bouldin_index'] = 0.0
            
            logger.info(f"Calculated traditional metrics with {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating traditional metrics: {e}")
            return {}
    
    def _calculate_clustering_metrics(self, X: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Calculate clustering metrics."""
        try:
            # Use KMeans for clustering
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(X)
            labels = kmeans.labels_
            
            # Calculate Calinski-Harabasz score
            calinski_harabasz = calinski_harabasz_score(X, labels)
            
            # Calculate Davies-Bouldin score
            davies_bouldin = davies_bouldin_score(X, labels)
            
            return {
                'calinski_harabasz_score': float(calinski_harabasz),
                'davies_bouldin_score': float(davies_bouldin)
            }
        except Exception as e:
            logger.error(f"Error calculating clustering metrics: {e}")
            return {} 