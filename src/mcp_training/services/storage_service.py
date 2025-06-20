"""
Storage service for MCP Training Service.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import shutil
import json
from datetime import datetime, timedelta

from ..utils.logger import get_logger
from ..utils.file_utils import (
    ensure_directory, copy_file, move_file, delete_file,
    cleanup_old_files, get_file_size, get_file_hash,
    list_files, get_directory_size
)
from ..utils.validation import validate_file_path, validate_directory_path

logger = get_logger(__name__)


class StorageService:
    """Service for managing file storage operations."""
    
    def __init__(
        self,
        models_dir: str = "models",
        exports_dir: str = "exports",
        logs_dir: str = "logs",
        temp_dir: str = "/tmp/mcp_training"
    ):
        """Initialize storage service.
        
        Args:
            models_dir: Directory for storing models
            exports_dir: Directory for storing exports
            logs_dir: Directory for storing logs
            temp_dir: Directory for temporary files
        """
        self.models_dir = Path(models_dir)
        self.exports_dir = Path(exports_dir)
        self.logs_dir = Path(logs_dir)
        self.temp_dir = Path(temp_dir)
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        ensure_directory(self.models_dir)
        ensure_directory(self.exports_dir)
        ensure_directory(self.logs_dir)
        ensure_directory(self.temp_dir)
    
    def store_export(self, export_file: Union[str, Path], filename: Optional[str] = None) -> Optional[Path]:
        """Store an export file in the exports directory.
        
        Args:
            export_file: Path to export file or file object
            filename: Optional filename to use (uses original name if not provided)
            
        Returns:
            Path to stored file or None if error
        """
        try:
            # Handle file objects (from FastAPI UploadFile)
            if hasattr(export_file, 'read'):
                return self.store_uploaded_export(export_file, filename)
            
            # Handle file paths
            source_path = Path(export_file)
            if not validate_file_path(source_path):
                logger.error(f"Invalid export file: {export_file}")
                return None
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"export_{timestamp}.json"
            
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            dest_path = self.exports_dir / filename
            
            # Copy file
            success = copy_file(source_path, dest_path, overwrite=True)
            if success:
                logger.info(f"Export stored: {dest_path}")
                return dest_path
            else:
                logger.error(f"Failed to store export: {export_file}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing export {export_file}: {e}")
            return None
    
    def store_uploaded_export(self, file_obj, filename: Optional[str] = None) -> Optional[Path]:
        """Store an uploaded export file in the exports directory.
        
        Args:
            file_obj: File object from FastAPI UploadFile
            filename: Optional filename to use (uses original name if not provided)
            
        Returns:
            Path to stored file or None if error
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"export_{timestamp}.json"
            
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            dest_path = self.exports_dir / filename
            
            # Write uploaded file content
            with open(dest_path, 'wb') as f:
                # Reset file pointer to beginning
                file_obj.seek(0)
                # Copy file content
                shutil.copyfileobj(file_obj, f)
            
            logger.info(f"Uploaded export stored: {dest_path}")
            return dest_path
                
        except Exception as e:
            logger.error(f"Error storing uploaded export: {e}")
            return None
    
    def list_exports(self, pattern: str = "*.json") -> List[Dict[str, Any]]:
        """List all export files.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of export file information
        """
        try:
            files = list_files(self.exports_dir, pattern=pattern)
            exports = []
            
            for file_path in files:
                file_info = {
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size': get_file_size(file_path),
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'hash': get_file_hash(file_path)
                }
                exports.append(file_info)
            
            # Sort by modification time (newest first)
            exports.sort(key=lambda x: x['modified'], reverse=True)
            return exports
            
        except Exception as e:
            logger.error(f"Error listing exports: {e}")
            return []
    
    def get_export_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific export file.
        
        Args:
            filename: Export filename
            
        Returns:
            Export file information or None if not found
        """
        try:
            file_path = self.exports_dir / filename
            if not validate_file_path(file_path):
                return None
            
            # Load and validate export data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            file_info = {
                'filename': filename,
                'path': str(file_path),
                'size': get_file_size(file_path),
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'hash': get_file_hash(file_path),
                'record_count': len(data.get('data', [])),
                'has_metadata': 'export_metadata' in data
            }
            
            # Add metadata info if available
            if 'export_metadata' in data:
                metadata = data['export_metadata']
                file_info.update({
                    'created_at': metadata.get('created_at'),
                    'total_records': metadata.get('total_records'),
                    'format': metadata.get('format'),
                    'export_id': metadata.get('export_id')
                })
            
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting export info for {filename}: {e}")
            return None
    
    def delete_export(self, filename: str) -> bool:
        """Delete an export file.
        
        Args:
            filename: Export filename to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.exports_dir / filename
            success = delete_file(file_path)
            if success:
                logger.info(f"Export deleted: {filename}")
            return success
        except Exception as e:
            logger.error(f"Error deleting export {filename}: {e}")
            return False
    
    def cleanup_old_exports(self, days_old: int = 7) -> List[str]:
        """Clean up old export files.
        
        Args:
            days_old: Delete exports older than this many days
            
        Returns:
            List of deleted filenames
        """
        try:
            deleted_files = cleanup_old_files(
                self.exports_dir,
                pattern="*.json",
                days_old=days_old
            )
            
            deleted_filenames = [f.name for f in deleted_files]
            logger.info(f"Cleanup completed: {len(deleted_filenames)} exports deleted")
            return deleted_filenames
            
        except Exception as e:
            logger.error(f"Error during export cleanup: {e}")
            return []
    
    def store_model_file(
        self,
        source_file: Union[str, Path],
        model_version: str,
        file_type: str = "model"
    ) -> Optional[Path]:
        """Store a model file in the models directory.
        
        Args:
            source_file: Path to source file
            model_version: Model version
            file_type: Type of file (model, scaler, metadata)
            
        Returns:
            Path to stored file or None if error
        """
        try:
            source_path = Path(source_file)
            if not validate_file_path(source_path):
                logger.error(f"Invalid source file: {source_file}")
                return None
            
            # Create model directory
            model_dir = self.models_dir / model_version
            ensure_directory(model_dir)
            
            # Determine filename based on file type
            if file_type == "model":
                filename = "model.joblib"
            elif file_type == "scaler":
                filename = "scaler.joblib"
            elif file_type == "metadata":
                filename = "metadata.json"
            else:
                filename = f"{file_type}.joblib"
            
            dest_path = model_dir / filename
            
            # Copy file
            success = copy_file(source_path, dest_path, overwrite=True)
            if success:
                logger.info(f"Model file stored: {dest_path}")
                return dest_path
            else:
                logger.error(f"Failed to store model file: {source_file}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing model file {source_file}: {e}")
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Storage statistics dictionary
        """
        try:
            stats = {
                'models': {
                    'directory': str(self.models_dir),
                    'size': get_directory_size(self.models_dir),
                    'file_count': len(list_files(self.models_dir, recursive=True))
                },
                'exports': {
                    'directory': str(self.exports_dir),
                    'size': get_directory_size(self.exports_dir),
                    'file_count': len(list_files(self.exports_dir, pattern="*.json"))
                },
                'logs': {
                    'directory': str(self.logs_dir),
                    'size': get_directory_size(self.logs_dir),
                    'file_count': len(list_files(self.logs_dir, pattern="*.log"))
                },
                'temp': {
                    'directory': str(self.temp_dir),
                    'size': get_directory_size(self.temp_dir),
                    'file_count': len(list_files(self.temp_dir))
                }
            }
            
            # Calculate totals
            total_size = sum(section['size'] for section in stats.values())
            total_files = sum(section['file_count'] for section in stats.values())
            
            stats['total'] = {
                'size': total_size,
                'file_count': total_files
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    def backup_model(self, model_version: str, backup_dir: Union[str, Path]) -> bool:
        """Create a backup of a model.
        
        Args:
            model_version: Model version to backup
            backup_dir: Backup directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_dir = self.models_dir / model_version
            if not validate_directory_path(model_dir):
                logger.error(f"Model directory not found: {model_dir}")
                return False
            
            backup_path = Path(backup_dir)
            ensure_directory(backup_path)
            
            # Create timestamped backup directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_model_dir = backup_path / f"{model_version}_{timestamp}"
            ensure_directory(backup_model_dir)
            
            # Copy all files in model directory
            success = True
            for file_path in model_dir.iterdir():
                if file_path.is_file():
                    dest_path = backup_model_dir / file_path.name
                    if not copy_file(file_path, dest_path, overwrite=True):
                        success = False
            
            if success:
                logger.info(f"Model backup created: {backup_model_dir}")
            return success
            
        except Exception as e:
            logger.error(f"Error backing up model {model_version}: {e}")
            return False
    
    def restore_model(self, backup_dir: Union[str, Path], model_version: str) -> bool:
        """Restore a model from backup.
        
        Args:
            backup_dir: Backup directory
            model_version: Model version to restore
            
        Returns:
            True if successful, False otherwise
        """
        try:
            backup_path = Path(backup_dir)
            if not validate_directory_path(backup_path):
                logger.error(f"Backup directory not found: {backup_path}")
                return False
            
            # Find backup directory for this model version
            backup_dirs = [d for d in backup_path.iterdir() 
                          if d.is_dir() and d.name.startswith(model_version)]
            
            if not backup_dirs:
                logger.error(f"No backup found for model version: {model_version}")
                return False
            
            # Use the most recent backup
            backup_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            backup_model_dir = backup_dirs[0]
            
            # Create model directory
            model_dir = self.models_dir / model_version
            ensure_directory(model_dir)
            
            # Copy all files from backup
            success = True
            for file_path in backup_model_dir.iterdir():
                if file_path.is_file():
                    dest_path = model_dir / file_path.name
                    if not copy_file(file_path, dest_path, overwrite=True):
                        success = False
            
            if success:
                logger.info(f"Model restored from backup: {backup_model_dir}")
            return success
            
        except Exception as e:
            logger.error(f"Error restoring model {model_version}: {e}")
            return False
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> List[str]:
        """Clean up temporary files.
        
        Args:
            max_age_hours: Delete temp files older than this many hours
            
        Returns:
            List of deleted filenames
        """
        try:
            deleted_files = cleanup_old_files(
                self.temp_dir,
                pattern="*",
                days_old=max_age_hours / 24
            )
            
            deleted_filenames = [f.name for f in deleted_files]
            logger.info(f"Temp cleanup completed: {len(deleted_filenames)} files deleted")
            return deleted_filenames
            
        except Exception as e:
            logger.error(f"Error during temp cleanup: {e}")
            return []
    
    def validate_storage_integrity(self) -> Dict[str, Any]:
        """Validate storage integrity by checking file hashes and permissions.
        
        Returns:
            Integrity check results
        """
        try:
            results = {
                'models': {'valid': True, 'errors': []},
                'exports': {'valid': True, 'errors': []},
                'logs': {'valid': True, 'errors': []}
            }
            
            # Check models directory
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    model_files = list(model_dir.glob("*"))
                    if not model_files:
                        results['models']['errors'].append(f"Empty model directory: {model_dir.name}")
                        results['models']['valid'] = False
                    
                    # Check for required files
                    required_files = ['model.joblib', 'metadata.json']
                    for req_file in required_files:
                        if not (model_dir / req_file).exists():
                            results['models']['errors'].append(
                                f"Missing {req_file} in {model_dir.name}"
                            )
                            results['models']['valid'] = False
            
            # Check exports directory
            for export_file in self.exports_dir.glob("*.json"):
                try:
                    with open(export_file, 'r') as f:
                        json.load(f)  # Validate JSON
                except Exception as e:
                    results['exports']['errors'].append(
                        f"Invalid JSON in {export_file.name}: {e}"
                    )
                    results['exports']['valid'] = False
            
            # Check logs directory
            for log_file in self.logs_dir.glob("*.log"):
                if not log_file.stat().st_size > 0:
                    results['logs']['errors'].append(f"Empty log file: {log_file.name}")
                    results['logs']['valid'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Error during storage integrity check: {e}")
            return {'error': str(e)} 