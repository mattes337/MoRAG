#!/usr/bin/env python3
"""Initialize SQL database with required tables and schema."""

import sys
import time
from pathlib import Path
import structlog

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-core" / "src"))

from morag_core.database.initialization import DatabaseInitializer
from morag_core.config import get_settings

logger = structlog.get_logger()

def wait_for_database(max_retries: int = 30, retry_delay: int = 2) -> bool:
    """Wait for database to be available."""
    logger.info("Waiting for database to be available")
    
    for attempt in range(max_retries):
        try:
            settings = get_settings()
            initializer = DatabaseInitializer()
            
            # Test connection
            if initializer.db_manager.test_connection():
                logger.info("Database connection successful")
                return True
                
        except Exception as e:
            logger.warning(
                "Database connection failed, retrying", 
                attempt=attempt + 1, 
                max_retries=max_retries,
                error=str(e)
            )
            
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    logger.error("Failed to connect to database after all retries")
    return False

def main():
    """Initialize the SQL database."""
    try:
        logger.info("Starting SQL database initialization")
        
        # Wait for database to be available
        if not wait_for_database():
            logger.error("Database is not available")
            sys.exit(1)
        
        # Initialize database
        initializer = DatabaseInitializer()
        
        # Check if database is already initialized
        if initializer.verify_schema():
            logger.info("Database schema already exists and is valid")
            return
        
        # Initialize database with tables
        logger.info("Creating database tables")
        success = initializer.initialize_database(drop_existing=False)
        
        if success:
            logger.info("SQL database initialized successfully")
        else:
            logger.error("Failed to initialize SQL database")
            sys.exit(1)
            
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
