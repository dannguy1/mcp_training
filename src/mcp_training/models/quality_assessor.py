"""
Advanced quality assessment for WiFi anomaly detection models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality levels for model assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class QualityMetrics:
    """Quality metrics for model assessment."""
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    score_stability: float
    feature_utilization: float
    model_complexity: float
    data_quality: float
    overall_score: float


@dataclass
class QualityAssessment:
    """Complete quality assessment result."""
    quality_level: QualityLevel
    validation_status: str
    quality_metrics: QualityMetrics
    issues: List[str]
    recommendations: List[str]
    confidence_score: float
    assessment_summary: Dict[str, Any]


class AdvancedQualityAssessor:
    """Advanced quality assessment for anomaly detection models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the quality assessor."""
        self.config = config or self._get_default_config()
        self.quality_thresholds = self.config['quality_thresholds']
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'quality_thresholds': {
                'excellent': {
                    'min_silhouette_score': 0.7,
                    'min_calinski_harabasz': 500,
                    'max_davies_bouldin': 0.5,
                    'min_score_stability': 0.8,
                    'min_feature_utilization': 0.7,
                    'min_overall_score': 0.8
                },
                'good': {
                    'min_silhouette_score': 0.5,
                    'min_calinski_harabasz': 200,
                    'max_davies_bouldin': 1.0,
                    'min_score_stability': 0.6,
                    'min_feature_utilization': 0.5,
                    'min_overall_score': 0.6
                },
                'fair': {
                    'min_silhouette_score': 0.3,
                    'min_calinski_harabasz': 100,
                    'max_davies_bouldin': 1.5,
                    'min_score_stability': 0.4,
                    'min_feature_utilization': 0.3,
                    'min_overall_score': 0.4
                },
                'poor': {
                    'min_silhouette_score': 0.1,
                    'min_calinski_harabasz': 50,
                    'max_davies_bouldin': 2.0,
                    'min_score_stability': 0.2,
                    'min_feature_utilization': 0.2,
                    'min_overall_score': 0.2
                }
            },
            'assessment_weights': {
                'silhouette_score': 0.25,
                'calinski_harabasz_score': 0.20,
                'davies_bouldin_score': 0.15,
                'score_stability': 0.15,
                'feature_utilization': 0.15,
                'model_complexity': 0.10
            }
        }
    
    def assess_model_quality(
        self, 
        model: Any, 
        X: np.ndarray, 
        training_metrics: Optional[Dict[str, Any]] = None
    ) -> QualityAssessment:
        """Perform comprehensive quality assessment."""
        logger.info("Starting comprehensive quality assessment")
        
        try:
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(model, X, training_metrics)
            
            # Determine quality level
            quality_level = self._determine_quality_level(quality_metrics)
            
            # Generate issues and recommendations
            issues = self._identify_issues(quality_metrics, quality_level)
            recommendations = self._generate_recommendations(quality_metrics, issues, training_metrics)
            
            # Determine validation status
            validation_status = self._determine_validation_status(quality_level, issues)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(quality_metrics, training_metrics)
            
            # Create assessment summary
            assessment_summary = self._create_assessment_summary(
                quality_metrics, quality_level, issues, recommendations
            )
            
            assessment = QualityAssessment(
                quality_level=quality_level,
                validation_status=validation_status,
                quality_metrics=quality_metrics,
                issues=issues,
                recommendations=recommendations,
                confidence_score=confidence_score,
                assessment_summary=assessment_summary
            )
            
            logger.info(f"Quality assessment completed: {quality_level.value} ({validation_status})")
            return assessment
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            # Return failed assessment
            return self._create_failed_assessment(str(e))
    
    def _calculate_quality_metrics(
        self, 
        model: Any, 
        X: np.ndarray, 
        training_metrics: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        metrics = {}
        
        # Clustering quality metrics
        try:
            predictions = self._get_model_predictions(model, X)
            metrics['silhouette_score'] = self._calculate_silhouette_score(X, predictions)
            metrics['calinski_harabasz_score'] = self._calculate_calinski_harabasz_score(X, predictions)
            metrics['davies_bouldin_score'] = self._calculate_davies_bouldin_score(X, predictions)
        except Exception as e:
            logger.warning(f"Error calculating clustering metrics: {e}")
            metrics['silhouette_score'] = 0.0
            metrics['calinski_harabasz_score'] = 0.0
            metrics['davies_bouldin_score'] = 2.0
        
        # Score stability
        metrics['score_stability'] = self._calculate_score_stability(model, X)
        
        # Feature utilization
        metrics['feature_utilization'] = self._calculate_feature_utilization(model, X)
        
        # Model complexity
        metrics['model_complexity'] = self._calculate_model_complexity(model, X)
        
        # Data quality (from training metrics)
        metrics['data_quality'] = self._calculate_data_quality(training_metrics)
        
        # Overall score
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        
        return QualityMetrics(**metrics)
    
    def _get_model_predictions(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get predictions from the model."""
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
            # Handle anomaly detection models
            if -1 in predictions:
                # Convert anomaly scores to cluster labels
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(X)
                elif hasattr(model, 'score_samples'):
                    scores = model.score_samples(X)
                else:
                    scores = predictions
                
                # Create clusters based on score percentiles
                labels = pd.cut(scores, bins=3, labels=[0, 1, 2])
                return labels.astype(int)
            else:
                return predictions
        else:
            # Fallback: use K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            return kmeans.fit_predict(X)
    
    def _calculate_silhouette_score(self, X: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate silhouette score."""
        try:
            if len(np.unique(predictions)) > 1:
                return silhouette_score(X, predictions)
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_calinski_harabasz_score(self, X: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score."""
        try:
            if len(np.unique(predictions)) > 1:
                return calinski_harabasz_score(X, predictions)
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_davies_bouldin_score(self, X: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate Davies-Bouldin score."""
        try:
            if len(np.unique(predictions)) > 1:
                return davies_bouldin_score(X, predictions)
            else:
                return 2.0
        except:
            return 2.0
    
    def _calculate_score_stability(self, model: Any, X: np.ndarray) -> float:
        """Calculate score stability."""
        try:
            if hasattr(model, 'score_samples'):
                scores = model.score_samples(X)
                # Calculate stability as inverse of coefficient of variation
                cv = np.std(scores) / (np.mean(scores) + 1e-8)
                stability = 1.0 / (1.0 + cv)
                return min(stability, 1.0)
            else:
                return 0.5  # Default stability
        except:
            return 0.5
    
    def _calculate_feature_utilization(self, model: Any, X: np.ndarray) -> float:
        """Calculate feature utilization."""
        try:
            # Simplified feature utilization based on feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                utilization = np.mean(importances > 0.01)  # Features with >1% importance
                return utilization
            else:
                # Estimate based on feature variance
                feature_variance = np.var(X, axis=0)
                utilization = np.mean(feature_variance > np.percentile(feature_variance, 25))
                return utilization
        except:
            return 0.5
    
    def _calculate_model_complexity(self, model: Any, X: np.ndarray) -> float:
        """Calculate model complexity score."""
        try:
            # Simplified complexity based on model type and parameters
            if hasattr(model, 'n_estimators'):
                complexity = min(model.n_estimators / 100, 1.0)
            elif hasattr(model, 'n_clusters'):
                complexity = min(model.n_clusters / 10, 1.0)
            else:
                complexity = 0.5
            
            return complexity
        except:
            return 0.5
    
    def _calculate_data_quality(self, training_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate data quality score."""
        if not training_metrics:
            return 0.5
        
        try:
            # Extract quality indicators from training metrics
            quality_indicators = []
            
            # Check for preprocessing metrics
            if 'preprocessing' in training_metrics:
                prep = training_metrics['preprocessing']
                if 'valid_records' in prep and 'total_records' in prep:
                    quality_indicators.append(prep['valid_records'] / prep['total_records'])
            
            # Check for algorithm selection
            if 'algorithm_selection' in training_metrics:
                quality_indicators.append(0.8)  # Good if algorithm was selected
            
            # Check for hyperparameter optimization
            if 'hyperparameter_optimization' in training_metrics:
                opt = training_metrics['hyperparameter_optimization']
                if 'optimization_result' in opt:
                    best_score = opt['optimization_result'].get('best_score', 0)
                    quality_indicators.append(min(best_score, 1.0))
            
            if quality_indicators:
                return np.mean(quality_indicators)
            else:
                return 0.5
        except:
            return 0.5
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score."""
        weights = self.config['assessment_weights']
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            return score / total_weight
        else:
            return 0.0
    
    def _determine_quality_level(self, metrics: QualityMetrics) -> QualityLevel:
        """Determine quality level based on metrics."""
        thresholds = self.quality_thresholds
        
        # Check each level from excellent to poor
        for level_name, level_thresholds in thresholds.items():
            level = QualityLevel(level_name)
            if self._meets_thresholds(metrics, level_thresholds):
                return level
        
        return QualityLevel.FAILED
    
    def _meets_thresholds(self, metrics: QualityMetrics, thresholds: Dict[str, float]) -> bool:
        """Check if metrics meet the specified thresholds."""
        try:
            if metrics.silhouette_score < thresholds['min_silhouette_score']:
                return False
            if metrics.calinski_harabasz_score < thresholds['min_calinski_harabasz']:
                return False
            if metrics.davies_bouldin_score > thresholds['max_davies_bouldin']:
                return False
            if metrics.score_stability < thresholds['min_score_stability']:
                return False
            if metrics.feature_utilization < thresholds['min_feature_utilization']:
                return False
            if metrics.overall_score < thresholds['min_overall_score']:
                return False
            return True
        except:
            return False
    
    def _identify_issues(self, metrics: QualityMetrics, quality_level: QualityLevel) -> List[str]:
        """Identify specific issues with the model."""
        issues = []
        
        # Check individual metrics
        if metrics.silhouette_score < 0.3:
            issues.append("Low silhouette score indicates poor cluster separation")
        
        if metrics.calinski_harabasz_score < 100:
            issues.append("Low Calinski-Harabasz score suggests weak cluster structure")
        
        if metrics.davies_bouldin_score > 1.5:
            issues.append("High Davies-Bouldin score indicates poor cluster compactness")
        
        if metrics.score_stability < 0.5:
            issues.append("Low score stability suggests model inconsistency")
        
        if metrics.feature_utilization < 0.4:
            issues.append("Low feature utilization indicates poor feature engineering")
        
        if metrics.overall_score < 0.4:
            issues.append("Overall model quality is below acceptable threshold")
        
        # Add quality level specific issues
        if quality_level == QualityLevel.FAILED:
            issues.append("Model failed quality validation and should not be deployed")
        elif quality_level == QualityLevel.POOR:
            issues.append("Model quality is poor and requires significant improvements")
        
        return issues
    
    def _generate_recommendations(
        self, 
        metrics: QualityMetrics, 
        issues: List[str], 
        training_metrics: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate specific recommendations for improvement."""
        recommendations = []
        
        # Feature engineering recommendations
        if metrics.feature_utilization < 0.5:
            recommendations.append("Improve feature engineering: add domain-specific features")
            recommendations.append("Consider feature selection to remove irrelevant features")
        
        # Model selection recommendations
        if metrics.overall_score < 0.5:
            recommendations.append("Try different algorithms: consider ensemble methods")
            recommendations.append("Increase training data size if possible")
        
        # Hyperparameter optimization recommendations
        if training_metrics and 'hyperparameter_optimization' not in training_metrics:
            recommendations.append("Enable hyperparameter optimization for better performance")
        
        # Data quality recommendations
        if metrics.data_quality < 0.6:
            recommendations.append("Improve data quality: clean and preprocess data")
            recommendations.append("Check for data consistency and completeness")
        
        # Specific metric-based recommendations
        if metrics.silhouette_score < 0.3:
            recommendations.append("Improve cluster separation: try different distance metrics")
        
        if metrics.score_stability < 0.5:
            recommendations.append("Increase model robustness: use ensemble methods")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Model quality is acceptable for deployment")
        
        return recommendations
    
    def _determine_validation_status(self, quality_level: QualityLevel, issues: List[str]) -> str:
        """Determine validation status."""
        if quality_level == QualityLevel.FAILED:
            return "FAILED"
        elif quality_level == QualityLevel.POOR:
            return "REJECTED"
        elif quality_level == QualityLevel.FAIR:
            return "CONDITIONAL"
        elif quality_level == QualityLevel.GOOD:
            return "VALID"
        elif quality_level == QualityLevel.EXCELLENT:
            return "VALID"
        else:
            return "UNKNOWN"
    
    def _calculate_confidence_score(
        self, 
        metrics: QualityMetrics, 
        training_metrics: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate confidence score for the assessment."""
        confidence_factors = []
        
        # Metric consistency
        metric_scores = [
            metrics.silhouette_score,
            metrics.calinski_harabasz_score,
            1.0 - (metrics.davies_bouldin_score / 2.0),  # Normalize
            metrics.score_stability,
            metrics.feature_utilization
        ]
        confidence_factors.append(np.std(metric_scores))  # Lower std = higher confidence
        
        # Data quality
        confidence_factors.append(metrics.data_quality)
        
        # Training process quality
        if training_metrics:
            if 'hyperparameter_optimization' in training_metrics:
                confidence_factors.append(0.9)
            if 'algorithm_selection' in training_metrics:
                confidence_factors.append(0.8)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _create_assessment_summary(
        self, 
        metrics: QualityMetrics, 
        quality_level: QualityLevel, 
        issues: List[str], 
        recommendations: List[str]
    ) -> Dict[str, Any]:
        """Create comprehensive assessment summary."""
        return {
            'quality_level': quality_level.value,
            'overall_score': metrics.overall_score,
            'key_metrics': {
                'silhouette_score': metrics.silhouette_score,
                'calinski_harabasz_score': metrics.calinski_harabasz_score,
                'davies_bouldin_score': metrics.davies_bouldin_score,
                'score_stability': metrics.score_stability,
                'feature_utilization': metrics.feature_utilization
            },
            'issues_count': len(issues),
            'recommendations_count': len(recommendations),
            'assessment_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _create_failed_assessment(self, error_message: str) -> QualityAssessment:
        """Create a failed assessment when errors occur."""
        failed_metrics = QualityMetrics(
            silhouette_score=0.0,
            calinski_harabasz_score=0.0,
            davies_bouldin_score=2.0,
            score_stability=0.0,
            feature_utilization=0.0,
            model_complexity=0.0,
            data_quality=0.0,
            overall_score=0.0
        )
        
        return QualityAssessment(
            quality_level=QualityLevel.FAILED,
            validation_status="FAILED",
            quality_metrics=failed_metrics,
            issues=[f"Assessment failed: {error_message}"],
            recommendations=["Fix assessment errors and retry"],
            confidence_score=0.0,
            assessment_summary={
                'quality_level': 'failed',
                'error': error_message,
                'assessment_timestamp': pd.Timestamp.now().isoformat()
            }
        ) 