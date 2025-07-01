"""
Hyperparameter optimization for WiFi anomaly detection models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_method: str
    search_space: Dict[str, List[Any]]
    cv_results: Dict[str, Any]
    optimization_time: float


class HyperparameterOptimizer:
    """Optimize hyperparameters for anomaly detection models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the hyperparameter optimizer."""
        self.config = config or self._get_default_config()
        self.search_spaces = self._define_search_spaces()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default optimization configuration."""
        return {
            'optimization_method': 'randomized',  # 'grid' or 'randomized'
            'cv_folds': 3,
            'n_iter': 20,  # for randomized search
            'scoring_metric': 'silhouette_score',
            'enable_early_stopping': True,
            'max_time_minutes': 30,
            'parallel_jobs': -1
        }
    
    def _define_search_spaces(self) -> Dict[str, Dict[str, List[Any]]]:
        """Define search spaces for different algorithms."""
        return {
            'isolation_forest': {
                'n_estimators': [50, 100, 200, 300],
                'contamination': [0.05, 0.1, 0.15, 0.2],
                'max_samples': ['auto', 100, 200, 500],
                'max_features': [1.0, 0.8, 0.6, 0.4]
            },
            'local_outlier_factor': {
                'n_neighbors': [5, 10, 15, 20, 25],
                'contamination': [0.05, 0.1, 0.15, 0.2],
                'metric': ['euclidean', 'manhattan', 'cosine'],
                'leaf_size': [10, 20, 30, 50]
            },
            'dbscan': {
                'eps': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5],
                'min_samples': [2, 3, 5, 10, 15],
                'metric': ['euclidean', 'manhattan', 'cosine']
            },
            'kmeans': {
                'n_clusters': [2, 3, 4, 5, 6, 8, 10],
                'init': ['k-means++', 'random'],
                'n_init': [5, 10, 15],
                'max_iter': [100, 200, 300]
            }
        }
    
    def optimize_hyperparameters(
        self, 
        X: np.ndarray, 
        algorithm: str, 
        optimization_method: Optional[str] = None
    ) -> OptimizationResult:
        """Optimize hyperparameters for the specified algorithm."""
        import time
        
        start_time = time.time()
        method = optimization_method or self.config['optimization_method']
        
        logger.info(f"Starting hyperparameter optimization for {algorithm} using {method} search")
        
        if algorithm not in self.search_spaces:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        search_space = self.search_spaces[algorithm]
        
        # Create model instance
        model = self._create_model_instance(algorithm)
        
        # Create search object
        if method == 'grid':
            search = GridSearchCV(
                model,
                search_space,
                cv=self.config['cv_folds'],
                scoring=self._get_scoring_function(),
                n_jobs=self.config['parallel_jobs'],
                verbose=1
            )
        else:  # randomized
            search = RandomizedSearchCV(
                model,
                search_space,
                n_iter=self.config['n_iter'],
                cv=self.config['cv_folds'],
                scoring=self._get_scoring_function(),
                n_jobs=self.config['parallel_jobs'],
                verbose=1,
                random_state=42
            )
        
        # Perform optimization
        search.fit(X)
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            best_params=search.best_params_,
            best_score=search.best_score_,
            optimization_method=method,
            search_space=search_space,
            cv_results=search.cv_results_,
            optimization_time=optimization_time
        )
        
        logger.info(f"Optimization completed in {optimization_time:.2f}s")
        logger.info(f"Best parameters: {result.best_params}")
        logger.info(f"Best score: {result.best_score:.4f}")
        
        return result
    
    def _create_model_instance(self, algorithm: str):
        """Create a model instance for the specified algorithm."""
        if algorithm == 'isolation_forest':
            return IsolationForest(random_state=42)
        elif algorithm == 'local_outlier_factor':
            return LocalOutlierFactor(novelty=False)
        elif algorithm == 'dbscan':
            return DBSCAN()
        elif algorithm == 'kmeans':
            return KMeans(random_state=42)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _get_scoring_function(self) -> Callable:
        """Get the scoring function for optimization."""
        def silhouette_scorer(estimator, X):
            try:
                if hasattr(estimator, 'predict'):
                    # For anomaly detection models
                    predictions = estimator.predict(X)
                    # Convert to cluster labels for silhouette score
                    if -1 in predictions:  # Anomaly detection
                        # Create artificial clusters based on anomaly scores
                        if hasattr(estimator, 'decision_function'):
                            scores = estimator.decision_function(X)
                        elif hasattr(estimator, 'score_samples'):
                            scores = estimator.score_samples(X)
                        else:
                            scores = predictions
                        
                        # Create clusters based on score percentiles
                        labels = pd.cut(scores, bins=3, labels=[0, 1, 2])
                        labels = labels.astype(int)
                    else:
                        labels = predictions
                    
                    if len(np.unique(labels)) > 1:
                        return silhouette_score(X, labels)
                    else:
                        return 0.0
                else:
                    return 0.0
            except:
                return 0.0
        
        return silhouette_scorer
    
    def get_optimized_model(
        self, 
        X: np.ndarray, 
        algorithm: str, 
        optimization_result: Optional[OptimizationResult] = None
    ) -> Tuple[Any, OptimizationResult]:
        """Get an optimized model instance."""
        if optimization_result is None:
            optimization_result = self.optimize_hyperparameters(X, algorithm)
        
        # Create model with optimized parameters
        if algorithm == 'isolation_forest':
            model = IsolationForest(**optimization_result.best_params, random_state=42)
        elif algorithm == 'local_outlier_factor':
            model = LocalOutlierFactor(**optimization_result.best_params, novelty=False)
        elif algorithm == 'dbscan':
            model = DBSCAN(**optimization_result.best_params)
        elif algorithm == 'kmeans':
            model = KMeans(**optimization_result.best_params, random_state=42)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return model, optimization_result
    
    def analyze_optimization_results(self, result: OptimizationResult) -> Dict[str, Any]:
        """Analyze optimization results and provide insights."""
        analysis = {
            'optimization_summary': {
                'method': result.optimization_method,
                'best_score': result.best_score,
                'optimization_time': result.optimization_time,
                'total_combinations': len(result.cv_results['params'])
            },
            'parameter_importance': self._analyze_parameter_importance(result),
            'recommendations': self._generate_recommendations(result)
        }
        
        return analysis
    
    def _analyze_parameter_importance(self, result: OptimizationResult) -> Dict[str, float]:
        """Analyze the importance of different parameters."""
        importance = {}
        
        # Calculate parameter importance based on score variation
        for param_name in result.search_space.keys():
            param_values = []
            scores = []
            
            for i, params in enumerate(result.cv_results['params']):
                if param_name in params:
                    param_values.append(params[param_name])
                    scores.append(result.cv_results['mean_test_score'][i])
            
            if len(set(param_values)) > 1:
                # Calculate coefficient of variation
                score_std = np.std(scores)
                score_mean = np.mean(scores)
                if score_mean != 0:
                    importance[param_name] = score_std / abs(score_mean)
                else:
                    importance[param_name] = 0.0
            else:
                importance[param_name] = 0.0
        
        return importance
    
    def _generate_recommendations(self, result: OptimizationResult) -> List[str]:
        """Generate recommendations based on optimization results."""
        recommendations = []
        
        # Analyze optimization time
        if result.optimization_time > 60:
            recommendations.append("Consider using fewer parameter combinations or randomized search for faster optimization")
        
        # Analyze score improvement
        if result.best_score < 0.3:
            recommendations.append("Model performance is low. Consider feature engineering or different algorithm")
        
        # Analyze parameter values
        if 'contamination' in result.best_params:
            contamination = result.best_params['contamination']
            if contamination > 0.2:
                recommendations.append("High contamination suggests many anomalies. Consider data quality or feature engineering")
        
        if 'n_clusters' in result.best_params:
            n_clusters = result.best_params['n_clusters']
            if n_clusters < 3:
                recommendations.append("Low number of clusters. Consider more clusters for better separation")
        
        return recommendations


class AdaptiveOptimizer:
    """Adaptive optimizer that adjusts search space based on data characteristics."""
    
    def __init__(self, base_optimizer: HyperparameterOptimizer):
        """Initialize adaptive optimizer."""
        self.base_optimizer = base_optimizer
        self.data_characteristics = {}
        
    def analyze_data_and_adapt(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze data characteristics and adapt search spaces."""
        characteristics = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'data_density': self._calculate_data_density(X),
            'feature_correlation': self._calculate_feature_correlation(X),
            'data_complexity': self._calculate_data_complexity(X)
        }
        
        self.data_characteristics = characteristics
        
        # Adapt search spaces based on characteristics
        adapted_spaces = self._adapt_search_spaces(characteristics)
        
        return {
            'characteristics': characteristics,
            'adapted_spaces': adapted_spaces
        }
    
    def _calculate_data_density(self, X: np.ndarray) -> float:
        """Calculate data density."""
        from sklearn.neighbors import NearestNeighbors
        
        if X.shape[0] < 10:
            return 0.5
        
        nbrs = NearestNeighbors(n_neighbors=min(5, X.shape[0]-1)).fit(X)
        distances, _ = nbrs.kneighbors(X)
        avg_distance = np.mean(distances[:, 1:])
        density = 1.0 / (1.0 + avg_distance)
        return min(density, 1.0)
    
    def _calculate_feature_correlation(self, X: np.ndarray) -> float:
        """Calculate average feature correlation."""
        if X.shape[1] < 2:
            return 0.0
        
        corr_matrix = np.corrcoef(X.T)
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        return np.mean(np.abs(upper_triangle))
    
    def _calculate_data_complexity(self, X: np.ndarray) -> float:
        """Calculate data complexity score."""
        # Simplified complexity based on variance and feature count
        feature_variance = np.var(X, axis=0)
        complexity = np.mean(feature_variance) * np.log(X.shape[1] + 1)
        return complexity
    
    def _adapt_search_spaces(self, characteristics: Dict[str, Any]) -> Dict[str, Dict[str, List[Any]]]:
        """Adapt search spaces based on data characteristics."""
        adapted_spaces = {}
        
        n_samples = characteristics['n_samples']
        n_features = characteristics['n_features']
        data_density = characteristics['data_density']
        
        # Adapt Isolation Forest
        if n_samples < 100:
            adapted_spaces['isolation_forest'] = {
                'n_estimators': [50, 100],
                'contamination': [0.1, 0.2],
                'max_samples': ['auto', 50],
                'max_features': [1.0, 0.8]
            }
        elif n_samples > 1000:
            adapted_spaces['isolation_forest'] = {
                'n_estimators': [100, 200, 300],
                'contamination': [0.05, 0.1, 0.15],
                'max_samples': ['auto', 200, 500],
                'max_features': [1.0, 0.8, 0.6]
            }
        else:
            adapted_spaces['isolation_forest'] = self.base_optimizer.search_spaces['isolation_forest']
        
        # Adapt LOF based on data density
        if data_density < 0.3:
            adapted_spaces['local_outlier_factor'] = {
                'n_neighbors': [5, 10, 15],
                'contamination': [0.1, 0.2],
                'metric': ['euclidean', 'manhattan'],
                'leaf_size': [10, 20]
            }
        else:
            adapted_spaces['local_outlier_factor'] = self.base_optimizer.search_spaces['local_outlier_factor']
        
        # Adapt DBSCAN based on feature count
        if n_features > 50:
            adapted_spaces['dbscan'] = {
                'eps': [0.5, 1.0, 1.5, 2.0],
                'min_samples': [3, 5, 10],
                'metric': ['euclidean', 'manhattan']
            }
        else:
            adapted_spaces['dbscan'] = self.base_optimizer.search_spaces['dbscan']
        
        # Adapt K-Means based on sample count
        max_clusters = min(10, n_samples // 10)
        adapted_spaces['kmeans'] = {
            'n_clusters': list(range(2, max(3, max_clusters + 1))),
            'init': ['k-means++', 'random'],
            'n_init': [5, 10],
            'max_iter': [100, 200]
        }
        
        return adapted_spaces
    
    def optimize_with_adaptation(
        self, 
        X: np.ndarray, 
        algorithm: str
    ) -> Tuple[OptimizationResult, Dict[str, Any]]:
        """Optimize with adapted search spaces."""
        # Analyze and adapt
        adaptation_result = self.analyze_data_and_adapt(X)
        
        # Temporarily replace search spaces
        original_spaces = self.base_optimizer.search_spaces.copy()
        self.base_optimizer.search_spaces = adaptation_result['adapted_spaces']
        
        try:
            # Perform optimization
            result = self.base_optimizer.optimize_hyperparameters(X, algorithm)
        finally:
            # Restore original spaces
            self.base_optimizer.search_spaces = original_spaces
        
        return result, adaptation_result 