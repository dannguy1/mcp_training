"""
Validation utilities for MCP Training Service.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def validate_file_path(file_path: Union[str, Path]) -> bool:
    """Validate that a file path exists and is accessible.
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception as e:
        logger.error(f"Error validating file path {file_path}: {e}")
        return False


def validate_directory_path(directory_path: Union[str, Path]) -> bool:
    """Validate that a directory path exists and is accessible.
    
    Args:
        directory_path: Path to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(directory_path)
        return path.exists() and path.is_dir()
    except Exception as e:
        logger.error(f"Error validating directory path {directory_path}: {e}")
        return False


def validate_json_data(data: Any) -> bool:
    """Validate that data is valid JSON-serializable.
    
    Args:
        data: Data to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError) as e:
        logger.error(f"Error validating JSON data: {e}")
        return False


def validate_export_structure(data: Dict[str, Any]) -> List[str]:
    """Validate export data structure.
    
    Args:
        data: Export data to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required top-level keys
    required_keys = ['data']
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    # Check data array
    if 'data' in data and not isinstance(data['data'], list):
        errors.append("'data' must be an array")
    
    # Check export_metadata if present
    if 'export_metadata' in data:
        metadata = data['export_metadata']
        if not isinstance(metadata, dict):
            errors.append("'export_metadata' must be an object")
        else:
            # Validate metadata fields
            if 'created_at' in metadata:
                if not _is_valid_iso_timestamp(metadata['created_at']):
                    errors.append("'export_metadata.created_at' must be a valid ISO timestamp")
    
    return errors


def validate_log_entry(entry: Dict[str, Any]) -> List[str]:
    """Validate a single log entry.
    
    Args:
        entry: Log entry to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required fields
    required_fields = ['timestamp', 'message', 'process_name']
    for field in required_fields:
        if field not in entry:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(entry[field], str):
            errors.append(f"Field '{field}' must be a string")
    
    # Validate timestamp
    if 'timestamp' in entry and isinstance(entry['timestamp'], str):
        if not _is_valid_iso_timestamp(entry['timestamp']):
            errors.append("'timestamp' must be a valid ISO timestamp")
    
    # Validate log_level if present
    if 'log_level' in entry:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if entry['log_level'] not in valid_levels:
            errors.append(f"'log_level' must be one of: {valid_levels}")
    
    # Validate structured_data if present
    if 'structured_data' in entry:
        if not isinstance(entry['structured_data'], dict):
            errors.append("'structured_data' must be an object")
    
    return errors


def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """Validate model configuration.
    
    Args:
        config: Model configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required sections
    required_sections = ['model', 'features', 'training', 'storage']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate model parameters
    if 'model' in config:
        model_config = config['model']
        if not isinstance(model_config, dict):
            errors.append("'model' section must be an object")
        else:
            # Check model type
            if 'type' in model_config:
                valid_types = ['isolation_forest', 'local_outlier_factor']
                if model_config['type'] not in valid_types:
                    errors.append(f"Model type must be one of: {valid_types}")
    
    # Validate features
    if 'features' in config:
        features_config = config['features']
        if not isinstance(features_config, dict):
            errors.append("'features' section must be an object")
        else:
            if 'numeric' in features_config:
                if not isinstance(features_config['numeric'], list):
                    errors.append("'features.numeric' must be an array")
    
    # Validate training parameters
    if 'training' in config:
        training_config = config['training']
        if not isinstance(training_config, dict):
            errors.append("'training' section must be an object")
        else:
            # Validate numeric parameters
            numeric_params = ['test_size', 'random_state', 'cross_validation_folds']
            for param in numeric_params:
                if param in training_config:
                    if not isinstance(training_config[param], (int, float)):
                        errors.append(f"'{param}' must be a number")
                    elif param == 'test_size' and not (0 < training_config[param] < 1):
                        errors.append("'test_size' must be between 0 and 1")
    
    return errors


def validate_training_config(config: Dict[str, Any]) -> List[str]:
    """Validate training service configuration.
    
    Args:
        config: Training configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required sections
    required_sections = ['service', 'api', 'storage']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate service settings
    if 'service' in config:
        service_config = config['service']
        if not isinstance(service_config, dict):
            errors.append("'service' section must be an object")
        else:
            if 'name' in service_config and not isinstance(service_config['name'], str):
                errors.append("'service.name' must be a string")
    
    # Validate API settings
    if 'api' in config:
        api_config = config['api']
        if not isinstance(api_config, dict):
            errors.append("'api' section must be an object")
        else:
            # Validate port
            if 'port' in api_config:
                if not isinstance(api_config['port'], int):
                    errors.append("'api.port' must be an integer")
                elif not (1024 <= api_config['port'] <= 65535):
                    errors.append("'api.port' must be between 1024 and 65535")
    
    # Validate storage settings
    if 'storage' in config:
        storage_config = config['storage']
        if not isinstance(storage_config, dict):
            errors.append("'storage' section must be an object")
        else:
            # Validate directory paths
            dir_fields = ['models_dir', 'exports_dir', 'logs_dir']
            for field in dir_fields:
                if field in storage_config:
                    if not isinstance(storage_config[field], str):
                        errors.append(f"'{field}' must be a string")
    
    return errors


def validate_email(email: str) -> bool:
    """Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))


def validate_mac_address(mac: str) -> bool:
    """Validate MAC address format.
    
    Args:
        mac: MAC address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
    return bool(re.match(pattern, mac))


def validate_ip_address(ip: str) -> bool:
    """Validate IP address format.
    
    Args:
        ip: IP address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    return bool(re.match(pattern, ip))


def _is_valid_iso_timestamp(timestamp: str) -> bool:
    """Check if a string is a valid ISO timestamp.
    
    Args:
        timestamp: Timestamp string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False


def validate_file_extension(file_path: Union[str, Path], allowed_extensions: List[str]) -> bool:
    """Validate file extension.
    
    Args:
        file_path: Path to file
        allowed_extensions: List of allowed extensions (with or without dot)
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Normalize extensions (add dot if missing)
        normalized_extensions = [
            ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
            for ext in allowed_extensions
        ]
        
        return extension in normalized_extensions
    except Exception as e:
        logger.error(f"Error validating file extension for {file_path}: {e}")
        return False


def validate_file_size(file_path: Union[str, Path], max_size_mb: float) -> bool:
    """Validate file size.
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum size in MB
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return False
        
        file_size_mb = path.stat().st_size / (1024 * 1024)
        return file_size_mb <= max_size_mb
    except Exception as e:
        logger.error(f"Error validating file size for {file_path}: {e}")
        return False 