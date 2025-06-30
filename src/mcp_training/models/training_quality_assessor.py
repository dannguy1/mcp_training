"""
Training Quality Assessor for MCP Training Service.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import logging
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)


class TrainingQualityAssessor:
    """Assess training quality and provide recommendations."""
    
    def __init__(self):
        """Initialize quality assessor."""
        self.quality_thresholds = {
            'min_samples': 1000,
            'min_features': 10,
            'max_training_time': 300,  # 5 minutes
            'min_score_variance': 0.01,
            'max_memory_usage': 1024,  # 1GB
            'min_quality_score': 0.6,
            'max_feature_count': 100,
            'min_data_quality_score': 0.7
        }
    
    def assess_training_quality(self, training_metrics: Dict[str, Any], 
                              evaluation_results: Dict[str, Any],
                              pipeline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall training quality.
        
        Args:
            training_metrics: Training process metrics
            evaluation_results: Model evaluation results
            pipeline_metrics: Pipeline performance metrics
            
        Returns:
            Quality assessment results
        """
        try:
            quality_report = {
                'overall_score': 0.0,
                'passed_checks': [],
                'failed_checks': [],
                'warnings': [],
                'recommendations': [],
                'assessment_timestamp': datetime.now().isoformat(),
                'detailed_scores': {}
            }
            
            # Assess data quality
            data_quality = self._assess_data_quality(training_metrics, pipeline_metrics)
            quality_report['detailed_scores']['data_quality'] = data_quality
            
            # Assess model performance
            model_performance = self._assess_model_performance(evaluation_results)
            quality_report['detailed_scores']['model_performance'] = model_performance
            
            # Assess resource usage
            resource_usage = self._assess_resource_usage(pipeline_metrics)
            quality_report['detailed_scores']['resource_usage'] = resource_usage
            
            # Assess training efficiency
            training_efficiency = self._assess_training_efficiency(pipeline_metrics)
            quality_report['detailed_scores']['training_efficiency'] = training_efficiency
            
            # Calculate overall score
            quality_report['overall_score'] = self._calculate_overall_score(quality_report['detailed_scores'])
            
            # Generate recommendations
            quality_report['recommendations'] = self._generate_recommendations(quality_report)
            
            # Update passed/failed checks
            quality_report['passed_checks'] = self._get_passed_checks(quality_report['detailed_scores'])
            quality_report['failed_checks'] = self._get_failed_checks(quality_report['detailed_scores'])
            quality_report['warnings'] = self._get_warnings(quality_report['detailed_scores'])
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error assessing training quality: {e}")
            return {
                'overall_score': 0.5,
                'passed_checks': [],
                'failed_checks': ['quality_assessment_failed'],
                'warnings': [],
                'recommendations': ['Unable to assess quality due to errors'],
                'assessment_timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _assess_data_quality(self, training_metrics: Dict[str, Any], 
                           pipeline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality metrics."""
        data_checks = {
            'score': 0.0,
            'passed_checks': [],
            'failed_checks': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Check sample count
            sample_count = pipeline_metrics.get('stages', {}).get('data_loading', {}).get('record_count', 0)
            data_checks['details']['sample_count'] = sample_count
            
            if sample_count >= self.quality_thresholds['min_samples']:
                data_checks['passed_checks'].append('sufficient_samples')
                data_checks['score'] += 0.3
            else:
                data_checks['failed_checks'].append('insufficient_samples')
            
            # Check feature count
            feature_count = pipeline_metrics.get('stages', {}).get('feature_extraction', {}).get('feature_count', 0)
            data_checks['details']['feature_count'] = feature_count
            
            if feature_count >= self.quality_thresholds['min_features']:
                data_checks['passed_checks'].append('sufficient_features')
                data_checks['score'] += 0.2
            else:
                data_checks['failed_checks'].append('insufficient_features')
            
            if feature_count > self.quality_thresholds['max_feature_count']:
                data_checks['warnings'].append('too_many_features')
            
            # Check data diversity
            preprocessing_metrics = training_metrics.get('preprocessing', {})
            valid_records = preprocessing_metrics.get('valid_records', 0)
            total_records = preprocessing_metrics.get('total_records', 0)
            
            if total_records > 0:
                data_quality_ratio = valid_records / total_records
                data_checks['details']['data_quality_ratio'] = data_quality_ratio
                
                if data_quality_ratio >= self.quality_thresholds['min_data_quality_score']:
                    data_checks['passed_checks'].append('good_data_quality')
                    data_checks['score'] += 0.3
                else:
                    data_checks['failed_checks'].append('poor_data_quality')
            
            # Check for missing values and duplicates
            missing_ratio = preprocessing_metrics.get('missing_timestamp', 0) / total_records if total_records > 0 else 0
            duplicate_ratio = preprocessing_metrics.get('duplicate_records', 0) / total_records if total_records > 0 else 0
            
            data_checks['details']['missing_ratio'] = missing_ratio
            data_checks['details']['duplicate_ratio'] = duplicate_ratio
            
            if missing_ratio < 0.1:
                data_checks['passed_checks'].append('low_missing_values')
                data_checks['score'] += 0.1
            else:
                data_checks['warnings'].append('high_missing_values')
            
            if duplicate_ratio < 0.05:
                data_checks['passed_checks'].append('low_duplicates')
                data_checks['score'] += 0.1
            else:
                data_checks['warnings'].append('high_duplicates')
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            data_checks['failed_checks'].append('data_quality_assessment_failed')
        
        return data_checks
    
    def _assess_model_performance(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model performance metrics."""
        performance_checks = {
            'score': 0.0,
            'passed_checks': [],
            'failed_checks': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Check basic metrics
            basic_metrics = evaluation_results.get('basic_metrics', {})
            quality_metrics = evaluation_results.get('quality_metrics', {})
            
            # Score variance
            score_variance = basic_metrics.get('score_variance', 0)
            performance_checks['details']['score_variance'] = score_variance
            
            if score_variance >= self.quality_thresholds['min_score_variance']:
                performance_checks['passed_checks'].append('good_score_variance')
                performance_checks['score'] += 0.2
            else:
                performance_checks['failed_checks'].append('low_score_variance')
            
            # Quality score from evaluation
            quality_score = evaluation_results.get('evaluation_summary', {}).get('model_quality_score', 0.5)
            performance_checks['details']['quality_score'] = quality_score
            
            if quality_score >= self.quality_thresholds['min_quality_score']:
                performance_checks['passed_checks'].append('good_model_quality')
                performance_checks['score'] += 0.3
            else:
                performance_checks['failed_checks'].append('poor_model_quality')
            
            # Silhouette score
            silhouette_score = quality_metrics.get('silhouette_score', 0)
            performance_checks['details']['silhouette_score'] = silhouette_score
            
            if silhouette_score > 0.3:
                performance_checks['passed_checks'].append('good_clustering')
                performance_checks['score'] += 0.2
            elif silhouette_score > 0.1:
                performance_checks['warnings'].append('moderate_clustering')
                performance_checks['score'] += 0.1
            else:
                performance_checks['failed_checks'].append('poor_clustering')
            
            # Feature utilization
            feature_utilization = quality_metrics.get('feature_utilization', 1.0)
            performance_checks['details']['feature_utilization'] = feature_utilization
            
            if feature_utilization >= 0.5:
                performance_checks['passed_checks'].append('good_feature_utilization')
                performance_checks['score'] += 0.2
            else:
                performance_checks['warnings'].append('low_feature_utilization')
            
            # Score stability
            score_stability = quality_metrics.get('score_stability', 0)
            performance_checks['details']['score_stability'] = score_stability
            
            if score_stability >= 0.7:
                performance_checks['passed_checks'].append('stable_scores')
                performance_checks['score'] += 0.1
            else:
                performance_checks['warnings'].append('unstable_scores')
            
        except Exception as e:
            logger.error(f"Error assessing model performance: {e}")
            performance_checks['failed_checks'].append('performance_assessment_failed')
        
        return performance_checks
    
    def _assess_resource_usage(self, pipeline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess resource usage during training."""
        resource_checks = {
            'score': 0.0,
            'passed_checks': [],
            'failed_checks': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Check memory usage
            max_memory_usage = 0
            for stage_name, stage_metrics in pipeline_metrics.get('stages', {}).items():
                memory_usage = stage_metrics.get('memory_usage_mb', 0)
                max_memory_usage = max(max_memory_usage, memory_usage)
            
            resource_checks['details']['max_memory_usage_mb'] = max_memory_usage
            
            if max_memory_usage <= self.quality_thresholds['max_memory_usage']:
                resource_checks['passed_checks'].append('reasonable_memory_usage')
                resource_checks['score'] += 0.5
            else:
                resource_checks['warnings'].append('high_memory_usage')
            
            # Check training duration
            total_duration = pipeline_metrics.get('total_duration', 0)
            resource_checks['details']['total_duration_seconds'] = total_duration
            
            if total_duration <= self.quality_thresholds['max_training_time']:
                resource_checks['passed_checks'].append('reasonable_training_time')
                resource_checks['score'] += 0.3
            else:
                resource_checks['warnings'].append('long_training_time')
            
            # Check individual stage performance
            stage_performance = {}
            for stage_name, stage_metrics in pipeline_metrics.get('stages', {}).items():
                stage_duration = stage_metrics.get('duration', 0)
                stage_performance[stage_name] = stage_duration
                
                # Flag extremely slow stages
                if stage_duration > 60:  # More than 1 minute
                    resource_checks['warnings'].append(f'slow_{stage_name}')
            
            resource_checks['details']['stage_performance'] = stage_performance
            
            # Check CPU usage (if available)
            current_cpu_percent = psutil.cpu_percent()
            resource_checks['details']['current_cpu_percent'] = current_cpu_percent
            
            if current_cpu_percent < 80:
                resource_checks['passed_checks'].append('reasonable_cpu_usage')
                resource_checks['score'] += 0.2
            else:
                resource_checks['warnings'].append('high_cpu_usage')
            
        except Exception as e:
            logger.error(f"Error assessing resource usage: {e}")
            resource_checks['failed_checks'].append('resource_assessment_failed')
        
        return resource_checks
    
    def _assess_training_efficiency(self, pipeline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess training efficiency."""
        efficiency_checks = {
            'score': 0.0,
            'passed_checks': [],
            'failed_checks': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Check overall success
            success = pipeline_metrics.get('success', False)
            if success:
                efficiency_checks['passed_checks'].append('training_completed')
                efficiency_checks['score'] += 0.4
            else:
                efficiency_checks['failed_checks'].append('training_failed')
            
            # Check stage completion
            stages = pipeline_metrics.get('stages', {})
            expected_stages = ['data_loading', 'feature_extraction', 'model_training', 'model_evaluation', 'model_saving']
            
            completed_stages = len(stages)
            efficiency_checks['details']['completed_stages'] = completed_stages
            efficiency_checks['details']['expected_stages'] = len(expected_stages)
            
            if completed_stages >= len(expected_stages):
                efficiency_checks['passed_checks'].append('all_stages_completed')
                efficiency_checks['score'] += 0.3
            else:
                efficiency_checks['failed_checks'].append('incomplete_pipeline')
            
            # Check for errors
            error = pipeline_metrics.get('error')
            if not error:
                efficiency_checks['passed_checks'].append('no_errors')
                efficiency_checks['score'] += 0.3
            else:
                efficiency_checks['failed_checks'].append('training_errors')
                efficiency_checks['details']['error'] = error
            
        except Exception as e:
            logger.error(f"Error assessing training efficiency: {e}")
            efficiency_checks['failed_checks'].append('efficiency_assessment_failed')
        
        return efficiency_checks
    
    def _calculate_overall_score(self, detailed_scores: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        try:
            weights = {
                'data_quality': 0.3,
                'model_performance': 0.4,
                'resource_usage': 0.2,
                'training_efficiency': 0.1
            }
            
            total_score = 0.0
            total_weight = 0.0
            
            for category, weight in weights.items():
                if category in detailed_scores:
                    score = detailed_scores[category].get('score', 0.0)
                    total_score += score * weight
                    total_weight += weight
            
            if total_weight > 0:
                return min(1.0, max(0.0, total_score / total_weight))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.5
    
    def _generate_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []
        
        try:
            overall_score = quality_report.get('overall_score', 0.0)
            detailed_scores = quality_report.get('detailed_scores', {})
            
            # Overall quality recommendations
            if overall_score < 0.5:
                recommendations.append("Training quality is poor. Consider reviewing data quality and model parameters.")
            elif overall_score < 0.7:
                recommendations.append("Training quality is moderate. Consider improving data quality or feature engineering.")
            elif overall_score >= 0.9:
                recommendations.append("Training quality is excellent!")
            
            # Data quality recommendations
            data_quality = detailed_scores.get('data_quality', {})
            if 'insufficient_samples' in data_quality.get('failed_checks', []):
                recommendations.append("Increase training data size for better model performance.")
            
            if 'insufficient_features' in data_quality.get('failed_checks', []):
                recommendations.append("Add more features or improve feature engineering.")
            
            if 'high_missing_values' in data_quality.get('warnings', []):
                recommendations.append("Address missing values in the training data.")
            
            # Model performance recommendations
            model_performance = detailed_scores.get('model_performance', {})
            if 'low_score_variance' in model_performance.get('failed_checks', []):
                recommendations.append("Consider using more diverse features or different model parameters.")
            
            if 'poor_model_quality' in model_performance.get('failed_checks', []):
                recommendations.append("Review model parameters and consider different algorithms.")
            
            if 'low_feature_utilization' in model_performance.get('warnings', []):
                recommendations.append("Consider feature selection to improve model efficiency.")
            
            # Resource usage recommendations
            resource_usage = detailed_scores.get('resource_usage', {})
            if 'high_memory_usage' in resource_usage.get('warnings', []):
                recommendations.append("Consider reducing data size or using more memory-efficient processing.")
            
            if 'long_training_time' in resource_usage.get('warnings', []):
                recommendations.append("Consider optimizing feature extraction or using faster algorithms.")
            
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append("Training completed successfully. Monitor model performance on new data.")
            
            recommendations.append("Consider retraining periodically with new data to maintain model performance.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to assessment errors.")
        
        return recommendations
    
    def _get_passed_checks(self, detailed_scores: Dict[str, Any]) -> List[str]:
        """Get all passed checks from detailed scores."""
        passed_checks = []
        for category, checks in detailed_scores.items():
            passed_checks.extend(checks.get('passed_checks', []))
        return passed_checks
    
    def _get_failed_checks(self, detailed_scores: Dict[str, Any]) -> List[str]:
        """Get all failed checks from detailed scores."""
        failed_checks = []
        for category, checks in detailed_scores.items():
            failed_checks.extend(checks.get('failed_checks', []))
        return failed_checks
    
    def _get_warnings(self, detailed_scores: Dict[str, Any]) -> List[str]:
        """Get all warnings from detailed scores."""
        warnings = []
        for category, checks in detailed_scores.items():
            warnings.extend(checks.get('warnings', []))
        return warnings 