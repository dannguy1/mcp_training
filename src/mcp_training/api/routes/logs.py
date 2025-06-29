"""
Training Logs API routes for MCP Training Service.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()


class TrainingLogEntry(BaseModel):
    """Training log entry model."""
    timestamp: datetime = Field(..., description="Log timestamp")
    level: str = Field(..., description="Log level")
    message: str = Field(..., description="Log message")
    training_id: Optional[str] = Field(None, description="Training job ID")
    step: Optional[str] = Field(None, description="Training step")
    progress: Optional[float] = Field(None, description="Training progress percentage")
    module: Optional[str] = Field(None, description="Module name")
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional log data")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingLogList(BaseModel):
    """Training log list response model."""
    logs: List[TrainingLogEntry] = Field(..., description="List of training log entries")
    total: int = Field(..., description="Total number of logs")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")


async def broadcast_log_entry(log_entry: Dict[str, Any]):
    """Broadcast log entry via WebSocket."""
    try:
        from .websocket import broadcast_log_entry as ws_broadcast_log
        await ws_broadcast_log(log_entry)
    except ImportError:
        # WebSocket not available, skip broadcasting
        pass
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to broadcast log entry: {e}")


def _read_training_logs(log_file_path: str, max_lines: int = 1000) -> List[Dict[str, Any]]:
    """Read training-specific log entries from a log file."""
    logs = []
    
    if not os.path.exists(log_file_path):
        return logs
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Read last max_lines lines
        lines = lines[-max_lines:] if len(lines) > max_lines else lines
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse JSON log entry
                log_entry = json.loads(line)
                
                # Only include training-related logs
                message = log_entry.get('message', '').lower()
                logger_name = log_entry.get('logger', '').lower()
                
                # Filter for training-related content
                training_keywords = [
                    'training', 'model', 'feature', 'export', 'validation',
                    'train', 'fit', 'evaluate', 'progress', 'step',
                    'training_id', 'model_path', 'export_file'
                ]
                
                is_training_log = any(keyword in message for keyword in training_keywords) or \
                                any(keyword in logger_name for keyword in training_keywords)
                
                if not is_training_log:
                    continue
                
                # Convert timestamp string to datetime
                if 'timestamp' in log_entry:
                    try:
                        log_entry['timestamp'] = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))
                    except:
                        log_entry['timestamp'] = datetime.now()
                
                # Extract training-specific fields
                extra_fields = {}
                for key, value in log_entry.items():
                    if key not in ['timestamp', 'level', 'message', 'module', 'function', 'line', 'logger']:
                        # Ensure value is serializable
                        try:
                            # Test if value can be JSON serialized
                            json.dumps(value)
                            extra_fields[key] = value
                        except (TypeError, ValueError):
                            # Convert non-serializable values to strings
                            extra_fields[key] = str(value)
                
                log_entry['extra'] = extra_fields
                
                # Try to extract training_id from message or extra fields
                training_id = None
                if 'training_id' in extra_fields:
                    training_id = extra_fields['training_id']
                elif 'training_id' in message:
                    # Extract from message like "Training completed successfully: training_123"
                    import re
                    match = re.search(r'training[_-]?id[:\s]+([a-zA-Z0-9_-]+)', message, re.IGNORECASE)
                    if match:
                        training_id = match.group(1)
                
                log_entry['training_id'] = training_id
                
                # Extract progress from message or extra fields
                progress = None
                if 'progress' in extra_fields:
                    progress = extra_fields['progress']
                elif 'progress' in message:
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)\s*%', message)
                    if match:
                        progress = float(match.group(1))
                
                log_entry['progress'] = progress
                
                logs.append(log_entry)
                
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        print(f"Error reading training log file {log_file_path}: {e}")
    
    return logs


@router.get("/", response_model=TrainingLogList)
async def get_training_logs(
    training_id: Optional[str] = Query(None, description="Filter by training job ID"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(100, description="Maximum number of logs to return"),
    offset: int = Query(0, description="Number of logs to skip")
) -> TrainingLogList:
    """Get training logs.
    
    Args:
        training_id: Filter by training job ID
        level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        start_time: Start time filter
        end_time: End time filter
        limit: Maximum number of logs to return
        offset: Number of logs to skip
        
    Returns:
        List of training log entries
    """
    try:
        # Read from training log files
        log_files = [
            "logs/training.log"
        ]
        
        all_logs = []
        for log_file in log_files:
            if os.path.exists(log_file):
                logs = _read_training_logs(log_file, max_lines=1000)
                all_logs.extend(logs)
        
        # Sort by timestamp (newest first)
        all_logs.sort(key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
        
        # Apply filters
        if training_id:
            all_logs = [log for log in all_logs if log.get('training_id') == training_id]
        
        if level:
            all_logs = [log for log in all_logs if log.get('level', '').upper() == level.upper()]
        
        if start_time:
            all_logs = [log for log in all_logs if log.get('timestamp', datetime.now()) >= start_time]
            
        if end_time:
            all_logs = [log for log in all_logs if log.get('timestamp', datetime.now()) <= end_time]
        
        # Apply pagination
        total = len(all_logs)
        logs = all_logs[offset:offset + limit]
        
        # Convert to TrainingLogEntry objects
        log_entries = []
        for log in logs:
            try:
                # Ensure extra field is a dict and serializable
                extra_data = log.get('extra', {})
                if not isinstance(extra_data, dict):
                    extra_data = {}
                
                # Clean extra data to ensure it's serializable
                clean_extra = {}
                for key, value in extra_data.items():
                    try:
                        json.dumps(value)
                        clean_extra[key] = value
                    except (TypeError, ValueError):
                        clean_extra[key] = str(value)
                
                log_entries.append(TrainingLogEntry(
                    timestamp=log.get('timestamp', datetime.now()),
                    level=log.get('level', 'INFO'),
                    message=log.get('message', ''),
                    training_id=log.get('training_id'),
                    step=log.get('extra', {}).get('step'),
                    progress=log.get('progress'),
                    module=log.get('module'),
                    extra=clean_extra
                ))
            except Exception as e:
                # Skip problematic log entries
                print(f"Error creating TrainingLogEntry: {e}")
                continue
        
        return TrainingLogList(
            logs=log_entries,
            total=total,
            page=(offset // limit) + 1,
            page_size=limit
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training logs: {str(e)}")


@router.get("/stats")
async def get_training_log_stats() -> Dict[str, Any]:
    """Get training log statistics.
    
    Returns:
        Training log statistics
    """
    try:
        # Read from training log files
        log_files = [
            "logs/training.log"
        ]
        
        all_logs = []
        for log_file in log_files:
            if os.path.exists(log_file):
                logs = _read_training_logs(log_file, max_lines=5000)
                all_logs.extend(logs)
        
        # Calculate statistics
        total_logs = len(all_logs)
        
        # Level distribution
        level_distribution = {}
        for log in all_logs:
            level = log.get('level', 'INFO')
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        # Training job distribution
        training_jobs = {}
        for log in all_logs:
            training_id = log.get('training_id', 'unknown')
            training_jobs[training_id] = training_jobs.get(training_id, 0) + 1
        
        # Recent activity (last 24 hours)
        now = datetime.now()
        recent_logs = [log for log in all_logs if log.get('timestamp', now) >= now - timedelta(hours=24)]
        
        # Training progress logs
        progress_logs = [log for log in all_logs if log.get('progress') is not None]
        
        return {
            "total_training_logs": total_logs,
            "logs_today": len(recent_logs),
            "active_training_jobs": len([tid for tid, count in training_jobs.items() if tid != 'unknown' and count > 5]),
            "level_distribution": level_distribution,
            "training_jobs": len([tid for tid in training_jobs.keys() if tid != 'unknown']),
            "progress_logs": len(progress_logs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training log stats: {str(e)}")


@router.delete("/")
async def clear_training_logs() -> Dict[str, str]:
    """Clear training logs.
    
    Returns:
        Success message
    """
    try:
        # Clear training log files
        log_files = [
            "logs/training.log"
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                # Truncate file instead of deleting
                with open(log_file, 'w') as f:
                    f.write('')
        
        return {"message": "Training logs cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear training logs: {str(e)}") 