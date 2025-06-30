"""
Enhanced training monitoring system for MCP Training Service.
"""

import logging
import time
import psutil
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    training_id: str
    timestamp: str
    step: str
    progress: float
    duration_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    disk_usage_percent: float
    record_count: Optional[int] = None
    feature_count: Optional[int] = None
    model_type: Optional[str] = None
    error: Optional[str] = None
    warning: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for training steps."""
    step_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    record_count: Optional[int] = None
    feature_count: Optional[int] = None
    memory_peak_mb: Optional[float] = None
    cpu_peak_percent: Optional[float] = None
    throughput_records_per_second: Optional[float] = None
    throughput_features_per_second: Optional[float] = None


class TrainingMonitor:
    """Enhanced training monitoring system."""
    
    def __init__(self, training_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize training monitor.
        
        Args:
            training_id: Unique training job ID
            config: Monitoring configuration
        """
        self.training_id = training_id
        self.config = config or {}
        self.start_time = datetime.now()
        self.metrics_history: List[TrainingMetrics] = []
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.current_step = "initializing"
        self.step_start_time = self.start_time
        self.resource_monitor_active = False
        self.quality_thresholds = {
            'min_samples': 1000,
            'min_features': 10,
            'max_training_time': 3600,  # 1 hour
            'min_score_variance': 0.01,
            'max_memory_usage': 2048,  # 2GB
            'max_cpu_percent': 90
        }
        
        # Override with config if provided
        if 'quality_thresholds' in self.config:
            self.quality_thresholds.update(self.config['quality_thresholds'])
    
    async def start_step(self, step_name: str, **kwargs):
        """Start monitoring a training step.
        
        Args:
            step_name: Name of the training step
            **kwargs: Additional step parameters
        """
        self.current_step = step_name
        self.step_start_time = datetime.now()
        
        # Create performance metrics for this step
        self.performance_metrics[step_name] = PerformanceMetrics(
            step_name=step_name,
            start_time=self.step_start_time,
            record_count=kwargs.get('record_count'),
            feature_count=kwargs.get('feature_count'),
            model_type=kwargs.get('model_type')
        )
        
        logger.info(f"Training step started: {step_name} (ID: {self.training_id})")
        
        # Start resource monitoring if not already active
        if not self.resource_monitor_active:
            asyncio.create_task(self._monitor_resources())
    
    async def end_step(self, step_name: str, **kwargs):
        """End monitoring a training step.
        
        Args:
            step_name: Name of the training step
            **kwargs: Additional step results
        """
        if step_name in self.performance_metrics:
            end_time = datetime.now()
            duration = (end_time - self.step_start_time).total_seconds()
            
            # Update performance metrics
            self.performance_metrics[step_name].end_time = end_time
            self.performance_metrics[step_name].duration_seconds = duration
            
            # Calculate throughput if record count is available
            if kwargs.get('record_count') and duration > 0:
                self.performance_metrics[step_name].throughput_records_per_second = (
                    kwargs['record_count'] / duration
                )
            
            if kwargs.get('feature_count') and duration > 0:
                self.performance_metrics[step_name].throughput_features_per_second = (
                    kwargs['feature_count'] / duration
                )
            
            # Update with final metrics
            self.performance_metrics[step_name].record_count = kwargs.get('record_count')
            self.performance_metrics[step_name].feature_count = kwargs.get('feature_count')
            
            logger.info(f"Training step completed: {step_name} in {duration:.2f}s (ID: {self.training_id})")
    
    async def track_progress(self, progress: float, step: str, message: str = "", **kwargs):
        """Track training progress with detailed metrics.
        
        Args:
            progress: Progress percentage (0-100)
            step: Current training step
            message: Additional message
            **kwargs: Additional metrics
        """
        current_time = datetime.now()
        duration = (current_time - self.start_time).total_seconds()
        
        # Get current resource usage
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
        cpu_percent = psutil.cpu_percent()
        disk_usage = psutil.disk_usage('/').percent
        
        # Create metrics record
        metrics = TrainingMetrics(
            training_id=self.training_id,
            timestamp=current_time.isoformat(),
            step=step,
            progress=progress,
            duration_seconds=duration,
            memory_usage_mb=memory_usage,
            cpu_percent=cpu_percent,
            disk_usage_percent=disk_usage,
            record_count=kwargs.get('record_count'),
            feature_count=kwargs.get('feature_count'),
            model_type=kwargs.get('model_type'),
            error=kwargs.get('error'),
            warning=kwargs.get('warning')
        )
        
        self.metrics_history.append(metrics)
        
        # Check for warnings
        warnings = self._check_warnings(metrics)
        if warnings:
            metrics.warning = '; '.join(warnings)
            logger.warning(f"Training warnings: {warnings} (ID: {self.training_id})")
        
        # Log progress
        logger.info(f"Training progress: {progress}% - {step} - {message} (ID: {self.training_id})")
    
    async def _monitor_resources(self):
        """Monitor system resources during training."""
        self.resource_monitor_active = True
        
        try:
            while self.resource_monitor_active:
                # Get current resource usage
                memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
                cpu_percent = psutil.cpu_percent()
                
                # Update peak values for current step
                if self.current_step in self.performance_metrics:
                    current_metrics = self.performance_metrics[self.current_step]
                    if current_metrics.memory_peak_mb is None or memory_usage > current_metrics.memory_peak_mb:
                        current_metrics.memory_peak_mb = memory_usage
                    if current_metrics.cpu_peak_percent is None or cpu_percent > current_metrics.cpu_peak_percent:
                        current_metrics.cpu_peak_percent = cpu_percent
                
                # Check for resource warnings
                if memory_usage > self.quality_thresholds['max_memory_usage']:
                    logger.warning(f"High memory usage: {memory_usage:.1f}MB (ID: {self.training_id})")
                
                if cpu_percent > self.quality_thresholds['max_cpu_percent']:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}% (ID: {self.training_id})")
                
                await asyncio.sleep(1)  # Monitor every second
                
        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")
        finally:
            self.resource_monitor_active = False
    
    def _check_warnings(self, metrics: TrainingMetrics) -> List[str]:
        """Check for training warnings based on metrics."""
        warnings = []
        
        # Check memory usage
        if metrics.memory_usage_mb > self.quality_thresholds['max_memory_usage']:
            warnings.append(f"High memory usage: {metrics.memory_usage_mb:.1f}MB")
        
        # Check CPU usage
        if metrics.cpu_percent > self.quality_thresholds['max_cpu_percent']:
            warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Check training duration
        if metrics.duration_seconds > self.quality_thresholds['max_training_time']:
            warnings.append(f"Long training duration: {metrics.duration_seconds:.1f}s")
        
        # Check sample count
        if metrics.record_count and metrics.record_count < self.quality_thresholds['min_samples']:
            warnings.append(f"Low sample count: {metrics.record_count}")
        
        # Check feature count
        if metrics.feature_count and metrics.feature_count < self.quality_thresholds['min_features']:
            warnings.append(f"Low feature count: {metrics.feature_count}")
        
        return warnings
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.metrics_history:
            return {}
        
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate summary statistics
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        cpu_usage = [m.cpu_percent for m in self.metrics_history]
        
        summary = {
            'training_id': self.training_id,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': total_duration,
            'current_step': self.current_step,
            'total_metrics_points': len(self.metrics_history),
            'performance_metrics': {
                step: asdict(metrics) for step, metrics in self.performance_metrics.items()
            },
            'resource_usage': {
                'memory_mb': {
                    'min': min(memory_usage) if memory_usage else 0,
                    'max': max(memory_usage) if memory_usage else 0,
                    'mean': np.mean(memory_usage) if memory_usage else 0,
                    'std': np.std(memory_usage) if memory_usage else 0
                },
                'cpu_percent': {
                    'min': min(cpu_usage) if cpu_usage else 0,
                    'max': max(cpu_usage) if cpu_usage else 0,
                    'mean': np.mean(cpu_usage) if cpu_usage else 0,
                    'std': np.std(cpu_usage) if cpu_usage else 0
                }
            },
            'quality_assessment': self._assess_training_quality(),
            'warnings': self._get_all_warnings(),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _assess_training_quality(self) -> Dict[str, Any]:
        """Assess overall training quality."""
        if not self.metrics_history:
            return {'overall_score': 0.0, 'passed_checks': [], 'failed_checks': []}
        
        passed_checks = []
        failed_checks = []
        
        # Check data quality
        record_counts = [m.record_count for m in self.metrics_history if m.record_count]
        if record_counts:
            max_records = max(record_counts)
            if max_records >= self.quality_thresholds['min_samples']:
                passed_checks.append('sufficient_samples')
            else:
                failed_checks.append('insufficient_samples')
        
        feature_counts = [m.feature_count for m in self.metrics_history if m.feature_count]
        if feature_counts:
            max_features = max(feature_counts)
            if max_features >= self.quality_thresholds['min_features']:
                passed_checks.append('sufficient_features')
            else:
                failed_checks.append('insufficient_features')
        
        # Check resource usage
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        if memory_usage:
            max_memory = max(memory_usage)
            if max_memory <= self.quality_thresholds['max_memory_usage']:
                passed_checks.append('acceptable_memory_usage')
            else:
                failed_checks.append('excessive_memory_usage')
        
        # Check training duration
        total_duration = (datetime.now() - self.start_time).total_seconds()
        if total_duration <= self.quality_thresholds['max_training_time']:
            passed_checks.append('reasonable_training_time')
        else:
            failed_checks.append('excessive_training_time')
        
        # Calculate overall score
        total_checks = len(passed_checks) + len(failed_checks)
        overall_score = len(passed_checks) / total_checks if total_checks > 0 else 0.0
        
        return {
            'overall_score': overall_score,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'total_checks': total_checks
        }
    
    def _get_all_warnings(self) -> List[str]:
        """Get all warnings from training."""
        warnings = []
        for metrics in self.metrics_history:
            if metrics.warning:
                warnings.append(f"{metrics.timestamp} - {metrics.step}: {metrics.warning}")
        return warnings
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on training metrics."""
        recommendations = []
        
        # Check performance metrics
        for step_name, metrics in self.performance_metrics.items():
            if metrics.duration_seconds and metrics.duration_seconds > 60:
                recommendations.append(f"Consider optimizing {step_name} step (took {metrics.duration_seconds:.1f}s)")
            
            if metrics.throughput_records_per_second and metrics.throughput_records_per_second < 1000:
                recommendations.append(f"Low throughput in {step_name}: {metrics.throughput_records_per_second:.1f} records/s")
        
        # Check resource usage
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        if memory_usage and max(memory_usage) > 1024:  # 1GB
            recommendations.append("Consider reducing memory usage or increasing available memory")
        
        # Check data quality
        record_counts = [m.record_count for m in self.metrics_history if m.record_count]
        if record_counts and max(record_counts) < 5000:
            recommendations.append("Consider collecting more training data for better model performance")
        
        return recommendations
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self.resource_monitor_active = False
    
    def save_metrics(self, file_path: str):
        """Save training metrics to file."""
        try:
            metrics_data = {
                'training_id': self.training_id,
                'config': self.config,
                'quality_thresholds': self.quality_thresholds,
                'metrics_history': [asdict(m) for m in self.metrics_history],
                'performance_metrics': {
                    step: asdict(metrics) for step, metrics in self.performance_metrics.items()
                },
                'summary': self.get_training_summary()
            }
            
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            logger.info(f"Training metrics saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving training metrics: {e}")


class TrainingPerformanceDashboard:
    """Real-time training performance monitoring dashboard."""
    
    def __init__(self):
        """Initialize performance dashboard."""
        self.active_trainings: Dict[str, TrainingMonitor] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.websocket_clients: List[Callable] = []
    
    def register_training(self, training_id: str, monitor: TrainingMonitor):
        """Register a training job for monitoring."""
        self.active_trainings[training_id] = monitor
        logger.info(f"Training registered for monitoring: {training_id}")
    
    def unregister_training(self, training_id: str):
        """Unregister a training job from monitoring."""
        if training_id in self.active_trainings:
            del self.active_trainings[training_id]
            logger.info(f"Training unregistered from monitoring: {training_id}")
    
    async def update_training_metrics(self, training_id: str, metrics: Dict[str, Any]):
        """Update training metrics in real-time."""
        if training_id in self.active_trainings:
            # Add to performance history
            self.performance_history.append({
                'training_id': training_id,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
            
            # Keep only last 1000 entries
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # Broadcast to WebSocket clients
            await self._broadcast_metrics_update(training_id, metrics)
    
    async def _broadcast_metrics_update(self, training_id: str, metrics: Dict[str, Any]):
        """Broadcast metrics update to WebSocket clients."""
        message = {
            'type': 'training_metrics_update',
            'training_id': training_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        # Send to all registered WebSocket clients
        for client in self.websocket_clients:
            try:
                await client(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket client: {e}")
    
    def get_active_trainings(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active trainings."""
        return {
            training_id: monitor.get_training_summary()
            for training_id, monitor in self.active_trainings.items()
        }
    
    def get_performance_history(self, training_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get performance history for specific training or all trainings."""
        if training_id:
            return [entry for entry in self.performance_history if entry['training_id'] == training_id]
        return self.performance_history
    
    def register_websocket_client(self, client: Callable):
        """Register a WebSocket client for real-time updates."""
        self.websocket_clients.append(client)
    
    def unregister_websocket_client(self, client: Callable):
        """Unregister a WebSocket client."""
        if client in self.websocket_clients:
            self.websocket_clients.remove(client)


# Global dashboard instance
training_dashboard = TrainingPerformanceDashboard() 