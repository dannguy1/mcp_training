"""
Export data validation for MCP Training Service.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from .config import config

logger = logging.getLogger(__name__)


class ExportValidator:
    """Validate export data format and content."""
    
    def __init__(self):
        """Initialize the export validator."""
        self.required_fields = ['export_metadata', 'data']
        self.metadata_fields = ['created_at', 'total_records', 'format']
        self.data_fields = ['timestamp', 'message', 'process_name', 'log_level']
        
    def validate_export_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """Validate an export file."""
        errors = []
        
        # Check file exists
        if not Path(file_path).exists():
            errors.append(f"Export file does not exist: {file_path}")
            return False, errors
        
        # Check file is readable
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Failed to read file: {e}")
            return False, errors
        
        # Validate structure
        structure_valid, structure_errors = self._validate_structure(data)
        errors.extend(structure_errors)
        
        if not structure_valid:
            return False, errors
        
        # Validate metadata
        metadata_valid, metadata_errors = self._validate_metadata(data.get('export_metadata', {}))
        errors.extend(metadata_errors)
        
        # Validate data
        data_valid, data_errors = self._validate_data(data.get('data', []))
        errors.extend(data_errors)
        
        return len(errors) == 0, errors
    
    def _validate_structure(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate the basic structure of the export data."""
        errors = []
        
        # Check required top-level fields
        for field in self.required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check data types
        if 'export_metadata' in data and not isinstance(data['export_metadata'], dict):
            errors.append("export_metadata must be a dictionary")
        
        if 'data' in data and not isinstance(data['data'], list):
            errors.append("data must be a list")
        
        return len(errors) == 0, errors
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate export metadata."""
        errors = []
        
        # Check required metadata fields
        for field in self.metadata_fields:
            if field not in metadata:
                errors.append(f"Missing metadata field: {field}")
        
        # Validate created_at format
        if 'created_at' in metadata:
            try:
                datetime.fromisoformat(metadata['created_at'].replace('Z', '+00:00'))
            except ValueError:
                errors.append("Invalid created_at format. Expected ISO 8601 format")
        
        # Validate total_records
        if 'total_records' in metadata:
            if not isinstance(metadata['total_records'], int) or metadata['total_records'] < 0:
                errors.append("total_records must be a non-negative integer")
        
        # Validate format
        if 'format' in metadata and metadata['format'] != 'json':
            errors.append("Unsupported format. Only 'json' is supported")
        
        return len(errors) == 0, errors
    
    def _validate_data(self, data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate data records."""
        errors = []
        
        if not data:
            errors.append("Data list is empty")
            return False, errors
        
        # Check each record
        for i, record in enumerate(data):
            record_valid, record_errors = self._validate_record(record, i)
            if not record_valid:
                errors.extend(record_errors)
        
        # Check data consistency
        consistency_errors = self._check_data_consistency(data)
        errors.extend(consistency_errors)
        
        return len(errors) == 0, errors
    
    def _validate_record(self, record: Dict[str, Any], index: int) -> Tuple[bool, List[str]]:
        """Validate a single data record."""
        errors = []
        
        # Check required fields
        for field in self.data_fields:
            if field not in record:
                errors.append(f"Record {index}: Missing required field '{field}'")
        
        # Validate timestamp
        if 'timestamp' in record:
            try:
                datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                errors.append(f"Record {index}: Invalid timestamp format")
        
        # Validate message
        if 'message' in record and not isinstance(record['message'], str):
            errors.append(f"Record {index}: message must be a string")
        
        # Validate process_name
        if 'process_name' in record and not isinstance(record['process_name'], str):
            errors.append(f"Record {index}: process_name must be a string")
        
        # Validate log_level
        if 'log_level' in record:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if record['log_level'] not in valid_levels:
                errors.append(f"Record {index}: Invalid log_level. Must be one of {valid_levels}")
        
        return len(errors) == 0, errors
    
    def _check_data_consistency(self, data: List[Dict[str, Any]]) -> List[str]:
        """Check consistency across all data records."""
        errors = []
        
        # Check timestamp ordering
        timestamps = []
        for i, record in enumerate(data):
            if 'timestamp' in record:
                try:
                    ts = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                    timestamps.append((i, ts))
                except ValueError:
                    continue
        
        if timestamps:
            sorted_timestamps = sorted(timestamps, key=lambda x: x[1])
            if timestamps != sorted_timestamps:
                errors.append("Data records are not in chronological order")
        
        # Check for duplicate records
        seen_records = set()
        for i, record in enumerate(data):
            record_key = (
                record.get('timestamp'),
                record.get('message'),
                record.get('process_name')
            )
            if record_key in seen_records:
                errors.append(f"Duplicate record found at index {i}")
            seen_records.add(record_key)
        
        return errors
    
    def get_export_summary(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the export file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read export file: {e}")
            return None
        
        metadata = data.get('export_metadata', {})
        records = data.get('data', [])
        
        # Analyze data
        process_counts = {}
        log_level_counts = {}
        message_lengths = []
        timestamps = []
        
        for record in records:
            # Process counts
            process = record.get('process_name', 'unknown')
            process_counts[process] = process_counts.get(process, 0) + 1
            
            # Log level counts
            level = record.get('log_level', 'unknown')
            log_level_counts[level] = log_level_counts.get(level, 0) + 1
            
            # Message lengths
            message = record.get('message', '')
            message_lengths.append(len(message))
            
            # Timestamps
            if 'timestamp' in record:
                try:
                    ts = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                    timestamps.append(ts)
                except ValueError:
                    pass
        
        # Calculate time range
        time_range = None
        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)
            time_range = {
                'start': min_time.isoformat(),
                'end': max_time.isoformat(),
                'duration_hours': (max_time - min_time).total_seconds() / 3600
            }
        
        return {
            'file_path': file_path,
            'file_size_mb': Path(file_path).stat().st_size / (1024 * 1024),
            'metadata': metadata,
            'record_count': len(records),
            'process_distribution': dict(sorted(process_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'log_level_distribution': log_level_counts,
            'message_stats': {
                'avg_length': sum(message_lengths) / len(message_lengths) if message_lengths else 0,
                'min_length': min(message_lengths) if message_lengths else 0,
                'max_length': max(message_lengths) if message_lengths else 0
            },
            'time_range': time_range,
            'validation': {
                'is_valid': self.validate_export_file(file_path)[0]
            }
        }
    
    def validate_multiple_exports(self, export_dir: str) -> Dict[str, Any]:
        """Validate all export files in a directory."""
        export_path = Path(export_dir)
        if not export_path.exists():
            return {'error': f'Export directory does not exist: {export_dir}'}
        
        results = {
            'directory': export_dir,
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'files': []
        }
        
        for file_path in export_path.glob('*.json'):
            results['total_files'] += 1
            
            is_valid, errors = self.validate_export_file(str(file_path))
            summary = self.get_export_summary(str(file_path))
            
            file_result = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'is_valid': is_valid,
                'errors': errors if not is_valid else [],
                'summary': summary
            }
            
            results['files'].append(file_result)
            
            if is_valid:
                results['valid_files'] += 1
            else:
                results['invalid_files'] += 1
        
        return results 