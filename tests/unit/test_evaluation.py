"""
Tests for the comprehensive model evaluation system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from sklearn.ensemble import IsolationForest

from mcp_training.models.evaluation import ModelEvaluator
from mcp_training.models.config import ModelConfig


class TestModelEvaluator:
    """Test the ModelEvaluator class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ModelConfig()
    
    @pytest.fixture
    def evaluator(self, config):
        """Create a test evaluator."""
        return ModelEvaluator(config)
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample trained model."""
        model = IsolationForest(n_estimators=10, random_state=42)
        # Create some sample data to fit the model
        X = np.random.randn(100, 5)
        model.fit(X)
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        # Create some pseudo-labels for testing
        y = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        return X, y
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.config is not None
        assert evaluator.metrics == {}
    
    def test_calculate_basic_metrics(self, evaluator, sample_data):
        """Test basic metrics calculation."""
        X, y = sample_data
        scores = np.random.randn(100)
        
        metrics = evaluator._calculate_basic_metrics(y, scores)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        assert 'average_precision' in metrics
        
        # Check that all metrics are floats
        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, float)
            assert 0 <= metric_value <= 1
    
    def test_calculate_advanced_metrics(self, evaluator, sample_data):
        """Test advanced metrics calculation."""
        X, y = sample_data
        scores = np.random.randn(100)
        
        advanced_metrics = evaluator._calculate_advanced_metrics(y, scores)
        
        assert 'confusion_matrix' in advanced_metrics
        assert 'classification_report' in advanced_metrics
        assert 'threshold_analysis' in advanced_metrics
        assert 'score_distribution' in advanced_metrics
        
        # Check confusion matrix
        cm = advanced_metrics['confusion_matrix']
        assert isinstance(cm, list)
        assert len(cm) > 0
        
        # Check score distribution
        score_dist = advanced_metrics['score_distribution']
        assert 'mean' in score_dist
        assert 'std' in score_dist
        assert 'min' in score_dist
        assert 'max' in score_dist
        assert 'percentiles' in score_dist
    
    def test_analyze_thresholds(self, evaluator, sample_data):
        """Test threshold analysis."""
        X, y = sample_data
        scores = np.random.randn(100)
        
        threshold_analysis = evaluator._analyze_thresholds(y, scores)
        
        assert isinstance(threshold_analysis, dict)
        assert len(threshold_analysis) > 0
        
        # Check that each threshold has the required metrics
        for threshold_name, threshold_data in threshold_analysis.items():
            assert 'threshold' in threshold_data
            assert 'precision' in threshold_data
            assert 'recall' in threshold_data
            assert 'f1_score' in threshold_data
            assert 'anomaly_ratio' in threshold_data
    
    def test_calculate_cross_validation(self, evaluator, sample_model, sample_data):
        """Test cross-validation calculation."""
        X, y = sample_data
        
        cv_results = evaluator._calculate_cross_validation(sample_model, X, y)
        
        assert 'mean_cv_score' in cv_results
        assert 'std_cv_score' in cv_results
        assert 'cv_scores' in cv_results
        
        # Check that cv_scores is a list
        assert isinstance(cv_results['cv_scores'], list)
    
    def test_calculate_feature_importance(self, evaluator, sample_model, sample_data):
        """Test feature importance calculation."""
        X, y = sample_data
        
        # Mock the model to have feature importances
        sample_model.feature_importances_ = np.random.rand(X.shape[1])
        
        feature_importance = evaluator._calculate_feature_importance(sample_model, X)
        
        assert isinstance(feature_importance, dict)
        assert len(feature_importance) == X.shape[1]
    
    def test_analyze_performance(self, evaluator, sample_data):
        """Test performance analysis."""
        X, y = sample_data
        scores = np.random.randn(100)
        
        performance_analysis = evaluator._analyze_performance(y, scores)
        
        assert 'score_analysis' in performance_analysis
        assert 'anomaly_detection_analysis' in performance_analysis
        
        score_analysis = performance_analysis['score_analysis']
        assert 'score_correlation_with_labels' in score_analysis
        assert 'score_separation' in score_analysis
        assert 'score_variance' in score_analysis
        assert 'score_skewness' in score_analysis
    
    def test_check_thresholds(self, evaluator):
        """Test threshold checking."""
        # Mock evaluation results
        results = {
            'basic_metrics': {
                'accuracy': 0.85,
                'precision': 0.75,
                'recall': 0.80,
                'f1_score': 0.77,
                'roc_auc': 0.82
            }
        }
        
        threshold_checks = evaluator._check_thresholds(results)
        
        assert isinstance(threshold_checks, dict)
        assert all(isinstance(check, bool) for check in threshold_checks.values())
    
    def test_get_evaluation_summary(self, evaluator):
        """Test evaluation summary generation."""
        # Mock metrics
        evaluator.metrics = {
            'basic_metrics': {
                'accuracy': 0.85,
                'precision': 0.75,
                'recall': 0.80,
                'f1_score': 0.77
            },
            'threshold_checks': {
                'accuracy': True,
                'precision': True,
                'recall': True,
                'f1_score': True
            }
        }
        
        summary = evaluator.get_evaluation_summary()
        
        assert 'overall_performance' in summary
        assert 'best_metrics' in summary
        assert 'recommendations' in summary
        
        overall_performance = summary['overall_performance']
        assert 'passed_thresholds' in overall_performance
        assert 'total_thresholds' in overall_performance
        assert 'threshold_pass_rate' in overall_performance
    
    def test_generate_recommendations(self, evaluator):
        """Test recommendation generation."""
        # Test with poor performance
        evaluator.metrics = {
            'basic_metrics': {
                'precision': 0.5,  # Below threshold
                'recall': 0.6,     # Below threshold
                'f1_score': 0.55,  # Below threshold
                'roc_auc': 0.6     # Below threshold
            },
            'threshold_checks': {
                'precision': False,
                'recall': False,
                'f1_score': False,
                'roc_auc': False
            },
            'advanced_metrics': {
                'score_distribution': {
                    'std': 0.05  # Low variance
                }
            }
        }
        
        recommendations = evaluator._generate_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check that recommendations contain expected text
        recommendation_text = ' '.join(recommendations).lower()
        assert 'contamination' in recommendation_text or 'parameter' in recommendation_text
    
    def test_comprehensive_evaluation(self, evaluator, sample_model, sample_data):
        """Test the complete evaluation process."""
        X, y = sample_data
        
        evaluation_results = evaluator.evaluate_model(sample_model, X, y)
        
        # Check that all expected sections are present
        assert 'basic_metrics' in evaluation_results
        assert 'advanced_metrics' in evaluation_results
        assert 'cross_validation' in evaluation_results
        assert 'feature_importance' in evaluation_results
        assert 'performance_analysis' in evaluation_results
        assert 'threshold_checks' in evaluation_results
        
        # Check that metrics are stored
        assert evaluator.metrics == evaluation_results
    
    def test_evaluation_with_none_labels(self, evaluator, sample_model, sample_data):
        """Test evaluation when labels are None (unsupervised)."""
        X, y = sample_data
        
        # Test with None labels
        evaluation_results = evaluator.evaluate_model(sample_model, X, None)
        
        assert 'basic_metrics' in evaluation_results
        assert 'threshold_checks' in evaluation_results
        
        # Should still generate pseudo-labels and calculate metrics
        basic_metrics = evaluation_results['basic_metrics']
        assert 'accuracy' in basic_metrics
        assert 'precision' in basic_metrics 