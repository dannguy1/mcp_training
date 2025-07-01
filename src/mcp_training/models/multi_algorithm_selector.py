"""
Multi-algorithm model selector for WiFi anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy import stats
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """Supported algorithm types."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    DBSCAN = "dbscan"
    KMEANS = "kmeans"
    ONE_CLASS_SVM = "one_class_svm"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"


@dataclass
class AlgorithmConfig:
    """Configuration for an algorithm."""
    name: str
    algorithm_type: AlgorithmType
    parameters: Dict[str, Any]
    description: str
    data_requirements: Dict[str, Any]
    performance_characteristics: Dict[str, Any]


class MultiAlgorithmSelector:
    """Selects the best algorithm for anomaly detection based on data characteristics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the algorithm selector."""
        self.config = config or self._get_default_config()
        self.algorithms = self._initialize_algorithms()
        self.data_characteristics = {}
        self.selected_algorithm = None
        self.selection_reason = ""
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enable_auto_selection': True,
            'evaluation_metrics': ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score'],
            'performance_thresholds': {
                'min_silhouette_score': 0.2,
                'min_calinski_harabasz_score': 100,
                'max_davies_bouldin_score': 2.0
            },
            'data_analysis': {
                'enable_dimensionality_reduction': True,
                'max_features_for_clustering': 50,
                'min_samples_for_evaluation': 100
            }
        }
    
    def _initialize_algorithms(self) -> Dict[str, AlgorithmConfig]:
        """Initialize available algorithms with configurations."""
        return {
            'isolation_forest': AlgorithmConfig(
                name="Isolation Forest",
                algorithm_type=AlgorithmType.ISOLATION_FOREST,
                parameters={
                    'n_estimators': 100,
                    'contamination': 0.1,
                    'random_state': 42,
                    'max_samples': 'auto'
                },
                description="Tree-based anomaly detection, good for high-dimensional data",
                data_requirements={
                    'min_samples': 50,
                    'max_features': 1000,
                    'preferred_data_types': ['numerical', 'mixed']
                },
                performance_characteristics={
                    'scalability': 'high',
                    'interpretability': 'medium',
                    'robustness': 'high',
                    'sensitivity': 'medium'
                }
            ),
            'local_outlier_factor': AlgorithmConfig(
                name="Local Outlier Factor",
                algorithm_type=AlgorithmType.LOCAL_OUTLIER_FACTOR,
                parameters={
                    'n_neighbors': 20,
                    'contamination': 0.1,
                    'metric': 'euclidean',
                    'novelty': False
                },
                description="Density-based anomaly detection, good for local patterns",
                data_requirements={
                    'min_samples': 100,
                    'max_features': 100,
                    'preferred_data_types': ['numerical']
                },
                performance_characteristics={
                    'scalability': 'medium',
                    'interpretability': 'high',
                    'robustness': 'medium',
                    'sensitivity': 'high'
                }
            ),
            'dbscan': AlgorithmConfig(
                name="DBSCAN",
                algorithm_type=AlgorithmType.DBSCAN,
                parameters={
                    'eps': 0.5,
                    'min_samples': 5,
                    'metric': 'euclidean'
                },
                description="Density-based clustering, identifies clusters and noise",
                data_requirements={
                    'min_samples': 50,
                    'max_features': 50,
                    'preferred_data_types': ['numerical']
                },
                performance_characteristics={
                    'scalability': 'medium',
                    'interpretability': 'high',
                    'robustness': 'medium',
                    'sensitivity': 'high'
                }
            ),
            'kmeans': AlgorithmConfig(
                name="K-Means",
                algorithm_type=AlgorithmType.KMEANS,
                parameters={
                    'n_clusters': 3,
                    'random_state': 42,
                    'n_init': 10
                },
                description="Centroid-based clustering, good for well-separated clusters",
                data_requirements={
                    'min_samples': 50,
                    'max_features': 100,
                    'preferred_data_types': ['numerical']
                },
                performance_characteristics={
                    'scalability': 'high',
                    'interpretability': 'medium',
                    'robustness': 'low',
                    'sensitivity': 'medium'
                }
            )
        }
    
    def analyze_data_characteristics(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze data characteristics to inform algorithm selection."""
        logger.info(f"Analyzing data characteristics for {X.shape[0]} samples with {X.shape[1]} features")
        
        characteristics = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'data_density': self._calculate_data_density(X),
            'feature_correlation': self._calculate_feature_correlation(X),
            'data_distribution': self._analyze_data_distribution(X),
            'dimensionality_complexity': self._calculate_dimensionality_complexity(X),
            'cluster_tendency': self._calculate_cluster_tendency(X),
            'noise_level': self._estimate_noise_level(X)
        }
        
        self.data_characteristics = characteristics
        logger.info(f"Data characteristics: {characteristics}")
        return characteristics
    
    def select_best_algorithm(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[str, Dict[str, Any]]:
        """Select the best algorithm based on data characteristics and evaluation."""
        # Robust check for X
        if not hasattr(X, 'ndim') or X.ndim != 2:
            logger.error(f"Input X to select_best_algorithm is not 2D. Type: {type(X)}, Value: {X}")
            raise ValueError(f"Input X must be a 2D array. Got type: {type(X)}, value: {X}")
        logger.info("Selecting best algorithm for anomaly detection")
        
        # Analyze data characteristics
        self.analyze_data_characteristics(X)
        
        # Filter algorithms based on data requirements
        suitable_algorithms = self._filter_suitable_algorithms(X)
        
        if not suitable_algorithms:
            logger.warning("No suitable algorithms found, using default Isolation Forest")
            return 'isolation_forest', self.algorithms['isolation_forest'].parameters
        
        # Evaluate algorithms if we have enough data
        if X.shape[0] >= self.config['data_analysis']['min_samples_for_evaluation']:
            algorithm_scores = self._evaluate_algorithms(X, suitable_algorithms)
            best_algorithm = max(algorithm_scores.items(), key=lambda x: x[1]['overall_score'])
            self.selected_algorithm = best_algorithm[0]
            self.selection_reason = f"Selected {best_algorithm[0]} with score {best_algorithm[1]['overall_score']:.3f}"
        else:
            # Use heuristics for small datasets
            best_algorithm = self._select_by_heuristics(suitable_algorithms)
            self.selected_algorithm = best_algorithm
            self.selection_reason = f"Selected {best_algorithm} based on heuristics for small dataset"
        
        logger.info(f"Selected algorithm: {self.selected_algorithm}")
        logger.info(f"Selection reason: {self.selection_reason}")
        
        return self.selected_algorithm, self.algorithms[self.selected_algorithm].parameters
    
    def _filter_suitable_algorithms(self, X: np.ndarray) -> List[str]:
        """Filter algorithms based on data requirements."""
        suitable = []
        
        for name, config in self.algorithms.items():
            requirements = config.data_requirements
            
            # Check sample size requirements
            if X.shape[0] < requirements['min_samples']:
                continue
            
            # Check feature count requirements
            if X.shape[1] > requirements['max_features']:
                continue
            
            # Check data type preferences (simplified)
            if requirements['preferred_data_types'] == ['numerical'] and not self._is_numerical_data(X):
                continue
            
            suitable.append(name)
        
        logger.info(f"Suitable algorithms: {suitable}")
        return suitable
    
    def _evaluate_algorithms(self, X: np.ndarray, algorithms: List[str]) -> Dict[str, Dict[str, float]]:
        """Evaluate algorithms using multiple metrics."""
        logger.info(f"Evaluating {len(algorithms)} algorithms")
        
        # Reduce dimensionality if needed
        if X.shape[1] > self.config['data_analysis']['max_features_for_clustering']:
            X_reduced = self._reduce_dimensionality(X)
        else:
            X_reduced = X
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        
        results = {}
        
        for algorithm_name in algorithms:
            logger.info(f"Evaluating {algorithm_name}")
            scores = self._evaluate_single_algorithm(X_scaled, algorithm_name)
            results[algorithm_name] = scores
        
        return results
    
    def _evaluate_single_algorithm(self, X: np.ndarray, algorithm_name: str) -> Dict[str, float]:
        """Evaluate a single algorithm."""
        config = self.algorithms[algorithm_name]
        scores = {}
        
        try:
            if algorithm_name == 'isolation_forest':
                model = IsolationForest(**config.parameters)
                predictions = model.fit_predict(X)
                scores['anomaly_detection_score'] = self._calculate_anomaly_detection_score(X, predictions)
                
            elif algorithm_name == 'local_outlier_factor':
                model = LocalOutlierFactor(**config.parameters)
                predictions = model.fit_predict(X)
                scores['anomaly_detection_score'] = self._calculate_anomaly_detection_score(X, predictions)
                
            elif algorithm_name == 'dbscan':
                model = DBSCAN(**config.parameters)
                predictions = model.fit_predict(X)
                scores['clustering_score'] = self._calculate_clustering_score(X, predictions)
                
            elif algorithm_name == 'kmeans':
                model = KMeans(**config.parameters)
                predictions = model.fit_predict(X)
                scores['clustering_score'] = self._calculate_clustering_score(X, predictions)
            
            # Calculate additional metrics
            if 'clustering_score' in scores:
                scores['silhouette_score'] = self._calculate_silhouette_score(X, predictions)
                scores['calinski_harabasz_score'] = self._calculate_calinski_harabasz_score(X, predictions)
            
            # Calculate overall score
            scores['overall_score'] = self._calculate_overall_score(scores)
            
        except Exception as e:
            logger.warning(f"Error evaluating {algorithm_name}: {e}")
            scores = {'overall_score': 0.0}
        
        return scores
    
    def _select_by_heuristics(self, suitable_algorithms: List[str]) -> str:
        """Select algorithm using heuristics for small datasets."""
        characteristics = self.data_characteristics
        
        # Prefer Isolation Forest for high-dimensional data
        if characteristics['n_features'] > 50:
            if 'isolation_forest' in suitable_algorithms:
                return 'isolation_forest'
        
        # Prefer LOF for low-dimensional data with good cluster tendency
        if characteristics['n_features'] <= 20 and characteristics['cluster_tendency'] > 0.5:
            if 'local_outlier_factor' in suitable_algorithms:
                return 'local_outlier_factor'
        
        # Prefer DBSCAN for medium-dimensional data with good density
        if 10 <= characteristics['n_features'] <= 50 and characteristics['data_density'] > 0.3:
            if 'dbscan' in suitable_algorithms:
                return 'dbscan'
        
        # Default to first suitable algorithm
        return suitable_algorithms[0]
    
    def _calculate_data_density(self, X: np.ndarray) -> float:
        """Calculate data density."""
        # Simplified: use nearest neighbor distances
        if X.shape[0] < 10:
            return 0.5
        
        nbrs = NearestNeighbors(n_neighbors=min(5, X.shape[0]-1)).fit(X)
        distances, _ = nbrs.kneighbors(X)
        avg_distance = np.mean(distances[:, 1:])  # Exclude self
        density = 1.0 / (1.0 + avg_distance)
        return min(density, 1.0)
    
    def _calculate_feature_correlation(self, X: np.ndarray) -> float:
        """Calculate average feature correlation."""
        if X.shape[1] < 2:
            return 0.0
        
        corr_matrix = np.corrcoef(X.T)
        # Get upper triangle excluding diagonal
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        return np.mean(np.abs(upper_triangle))
    
    def _analyze_data_distribution(self, X: np.ndarray) -> Dict[str, float]:
        """Analyze data distribution characteristics."""
        return {
            'skewness': np.mean([np.abs(stats.skew(X[:, i])) for i in range(X.shape[1]) if np.std(X[:, i]) > 0]),
            'kurtosis': np.mean([stats.kurtosis(X[:, i]) for i in range(X.shape[1]) if np.std(X[:, i]) > 0]),
            'variance': np.mean(np.var(X, axis=0))
        }
    
    def _calculate_dimensionality_complexity(self, X: np.ndarray) -> float:
        """Calculate dimensionality complexity."""
        # Simplified: ratio of samples to features
        return X.shape[0] / (X.shape[1] + 1)
    
    def _calculate_cluster_tendency(self, X: np.ndarray) -> float:
        """Calculate cluster tendency using Hopkins statistic."""
        if X.shape[0] < 20:
            return 0.5
        
        # Simplified cluster tendency calculation
        try:
            # Use silhouette score with random labels as proxy
            random_labels = np.random.randint(0, min(3, X.shape[0]//10), X.shape[0])
            if len(np.unique(random_labels)) > 1:
                return silhouette_score(X, random_labels)
            else:
                return 0.5
        except:
            return 0.5
    
    def _estimate_noise_level(self, X: np.ndarray) -> float:
        """Estimate noise level in the data."""
        # Simplified: use variance of nearest neighbor distances
        if X.shape[0] < 10:
            return 0.5
        
        nbrs = NearestNeighbors(n_neighbors=min(3, X.shape[0]-1)).fit(X)
        distances, _ = nbrs.kneighbors(X)
        noise_level = np.std(distances[:, 1:]) / (np.mean(distances[:, 1:]) + 1e-8)
        return min(noise_level, 1.0)
    
    def _is_numerical_data(self, X: np.ndarray) -> bool:
        """Check if data is numerical."""
        return np.issubdtype(X.dtype, np.number)
    
    def _reduce_dimensionality(self, X: np.ndarray) -> np.ndarray:
        """Reduce dimensionality using PCA."""
        max_features = self.config['data_analysis']['max_features_for_clustering']
        pca = PCA(n_components=min(max_features, X.shape[1], X.shape[0]-1))
        return pca.fit_transform(X)
    
    def _calculate_anomaly_detection_score(self, X: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate anomaly detection performance score."""
        # For anomaly detection, we want a good balance of anomaly detection
        anomaly_ratio = np.mean(predictions == -1)
        
        # Prefer moderate anomaly ratios (not too few, not too many)
        if 0.05 <= anomaly_ratio <= 0.2:
            score = 1.0
        elif 0.02 <= anomaly_ratio <= 0.3:
            score = 0.8
        else:
            score = 0.3
        
        return score
    
    def _calculate_clustering_score(self, X: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate clustering performance score."""
        n_clusters = len(np.unique(predictions))
        
        # Prefer reasonable number of clusters
        if 2 <= n_clusters <= 10:
            score = 1.0
        elif n_clusters == 1:
            score = 0.3
        else:
            score = 0.6
        
        return score
    
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
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall algorithm score."""
        weights = {
            'anomaly_detection_score': 0.4,
            'clustering_score': 0.3,
            'silhouette_score': 0.2,
            'calinski_harabasz_score': 0.1
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in scores:
                overall_score += scores[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            return overall_score / total_weight
        else:
            return 0.0
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of algorithm selection."""
        return {
            'selected_algorithm': self.selected_algorithm,
            'selection_reason': self.selection_reason,
            'data_characteristics': self.data_characteristics,
            'available_algorithms': list(self.algorithms.keys())
        }
    
    def create_model(self, algorithm_name: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Create a model instance for the selected algorithm."""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        config = self.algorithms[algorithm_name]
        model_params = parameters or config.parameters
        
        if algorithm_name == 'isolation_forest':
            return IsolationForest(**model_params)
        elif algorithm_name == 'local_outlier_factor':
            return LocalOutlierFactor(**model_params)
        elif algorithm_name == 'dbscan':
            return DBSCAN(**model_params)
        elif algorithm_name == 'kmeans':
            return KMeans(**model_params)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}") 