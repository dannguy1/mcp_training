"""
File utilities for MCP Training Service.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Tuple
import hashlib
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def ensure_directory(directory: Union[str, Path], create_parents: bool = True) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        create_parents: Whether to create parent directories
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    if create_parents:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(exist_ok=True)
    return path


def copy_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False,
    preserve_metadata: bool = True
) -> bool:
    """Copy a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file
        preserve_metadata: Whether to preserve file metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            logger.error(f"Source file does not exist: {source_path}")
            return False
        
        if dest_path.exists() and not overwrite:
            logger.warning(f"Destination file exists and overwrite=False: {dest_path}")
            return False
        
        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        if preserve_metadata:
            shutil.copy2(source_path, dest_path)
        else:
            shutil.copy(source_path, dest_path)
        
        logger.info(f"File copied: {source_path} -> {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error copying file {source} to {destination}: {e}")
        return False


def move_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """Move a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            logger.error(f"Source file does not exist: {source_path}")
            return False
        
        if dest_path.exists() and not overwrite:
            logger.warning(f"Destination file exists and overwrite=False: {dest_path}")
            return False
        
        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(str(source_path), str(dest_path))
        
        logger.info(f"File moved: {source_path} -> {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error moving file {source} to {destination}: {e}")
        return False


def delete_file(file_path: Union[str, Path]) -> bool:
    """Delete a file.
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.info(f"File deleted: {path}")
            return True
        else:
            logger.warning(f"File does not exist: {path}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False


def cleanup_old_files(
    directory: Union[str, Path],
    pattern: str = "*",
    days_old: int = 30,
    recursive: bool = False
) -> List[Path]:
    """Clean up old files in a directory.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match (glob pattern)
        days_old: Delete files older than this many days
        recursive: Whether to search recursively
        
    Returns:
        List of deleted file paths
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_files = []
        
        # Find files matching pattern
        if recursive:
            files = dir_path.rglob(pattern)
        else:
            files = dir_path.glob(pattern)
        
        for file_path in files:
            if file_path.is_file():
                # Check file modification time
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff_date:
                    try:
                        file_path.unlink()
                        deleted_files.append(file_path)
                        logger.info(f"Deleted old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {e}")
        
        logger.info(f"Cleanup completed: {len(deleted_files)} files deleted")
        return deleted_files
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return []


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    try:
        return Path(file_path).stat().st_size
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return 0


def get_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> Optional[str]:
    """Calculate file hash.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        File hash as hex string, or None if error
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        
        hash_func = hashlib.new(algorithm)
        
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
        
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None


def create_temp_file(
    suffix: str = "",
    prefix: str = "mcp_training_",
    directory: Optional[Union[str, Path]] = None
) -> Tuple[Path, int]:
    """Create a temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        directory: Directory to create file in
        
    Returns:
        Tuple of (file_path, file_descriptor)
    """
    try:
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=directory)
        return Path(path), fd
    except Exception as e:
        logger.error(f"Error creating temp file: {e}")
        raise


def create_temp_directory(
    suffix: str = "",
    prefix: str = "mcp_training_",
    directory: Optional[Union[str, Path]] = None
) -> Path:
    """Create a temporary directory.
    
    Args:
        suffix: Directory suffix
        prefix: Directory prefix
        directory: Parent directory
        
    Returns:
        Path to temporary directory
    """
    try:
        return Path(tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=directory))
    except Exception as e:
        logger.error(f"Error creating temp directory: {e}")
        raise


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False,
    include_dirs: bool = False
) -> List[Path]:
    """List files in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search recursively
        include_dirs: Whether to include directories
        
    Returns:
        List of file paths
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        
        if recursive:
            items = dir_path.rglob(pattern)
        else:
            items = dir_path.glob(pattern)
        
        if include_dirs:
            return [item for item in items if item.exists()]
        else:
            return [item for item in items if item.is_file()]
            
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {e}")
        return []


def get_directory_size(directory: Union[str, Path]) -> int:
    """Calculate total size of a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    try:
        total_size = 0
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return 0
        
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
        
    except Exception as e:
        logger.error(f"Error calculating directory size for {directory}: {e}")
        return 0 