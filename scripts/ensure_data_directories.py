#!/usr/bin/env python3
"""
Ensure MoRAG data directories exist with proper permissions.

This script creates the necessary directory structure for MoRAG,
including remote jobs directories, with proper permissions.
"""

import os
import sys
import stat
from pathlib import Path
import structlog

# Set up basic logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def ensure_directory(path: Path, mode: int = 0o755) -> bool:
    """Ensure a directory exists with proper permissions."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        
        # Set permissions
        path.chmod(mode)
        
        logger.info("Directory ensured", path=str(path), mode=oct(mode))
        return True
        
    except PermissionError as e:
        logger.error("Permission denied creating directory", path=str(path), error=str(e))
        return False
    except Exception as e:
        logger.error("Failed to create directory", path=str(path), error=str(e))
        return False


def check_directory_writable(path: Path) -> bool:
    """Check if a directory is writable."""
    try:
        # Try to create a test file
        test_file = path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        
        logger.debug("Directory is writable", path=str(path))
        return True
        
    except Exception as e:
        logger.error("Directory is not writable", path=str(path), error=str(e))
        return False


def main():
    """Main function to ensure all data directories exist."""
    logger.info("Starting data directory initialization")
    
    # Get base data directory from environment or use default
    base_data_dir = Path(os.getenv('MORAG_DATA_DIR', '/app/data'))
    
    # Define all required directories
    directories = [
        base_data_dir,
        base_data_dir / 'remote_jobs',
        base_data_dir / 'remote_jobs' / 'pending',
        base_data_dir / 'remote_jobs' / 'processing',
        base_data_dir / 'remote_jobs' / 'completed',
        base_data_dir / 'remote_jobs' / 'failed',
        base_data_dir / 'remote_jobs' / 'timeout',
        base_data_dir / 'remote_jobs' / 'cancelled',
        base_data_dir / 'uploads',
        base_data_dir / 'temp',
        base_data_dir / 'cache',
    ]
    
    # Additional directories for logs and temp
    log_dir = Path(os.getenv('MORAG_LOG_DIR', '/app/logs'))
    temp_dir = Path(os.getenv('MORAG_TEMP_DIR', '/app/temp'))
    
    directories.extend([log_dir, temp_dir])
    
    success = True
    
    # Create all directories
    for directory in directories:
        if not ensure_directory(directory):
            success = False
    
    # Check if directories are writable
    critical_dirs = [
        base_data_dir,
        base_data_dir / 'remote_jobs',
        log_dir,
        temp_dir
    ]
    
    for directory in critical_dirs:
        if directory.exists() and not check_directory_writable(directory):
            logger.error("Critical directory is not writable", path=str(directory))
            success = False
    
    # Print summary
    if success:
        logger.info("All data directories initialized successfully")
        print("✅ Data directories initialized successfully")
        
        # Print directory structure
        print("\n📁 Directory structure:")
        for directory in sorted(directories):
            if directory.exists():
                perms = oct(directory.stat().st_mode)[-3:]
                print(f"  {directory} (permissions: {perms})")
        
        return 0
    else:
        logger.error("Failed to initialize some data directories")
        print("❌ Failed to initialize some data directories")
        print("\n🔧 Troubleshooting:")
        print("1. Check if the process has write permissions to the parent directory")
        print("2. Ensure the Docker volume mounts are configured correctly")
        print("3. Check if SELinux or AppArmor is blocking directory creation")
        print("4. Verify the user running the process has appropriate permissions")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
