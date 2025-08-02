"""Connection and database management operations for Neo4j storage."""

import logging
from typing import Optional
from neo4j import AsyncGraphDatabase, AsyncDriver

from .base_operations import BaseOperations

logger = logging.getLogger(__name__)


class ConnectionOperations(BaseOperations):
    """Handles Neo4j connection and database management operations."""
    
    def __init__(self, config):
        """Initialize connection operations.
        
        Args:
            config: Neo4j configuration object
        """
        self.config = config
        self.driver: Optional[AsyncDriver] = None
        # Don't call super().__init__ yet since driver is None
    
    async def connect(self) -> None:
        """Connect to Neo4J database."""
        try:
            # Configure basic driver settings
            driver_kwargs = {
                "auth": (self.config.username, self.config.password),
                "max_connection_lifetime": self.config.max_connection_lifetime,
                "max_connection_pool_size": self.config.max_connection_pool_size,
                "connection_acquisition_timeout": self.config.connection_acquisition_timeout,
            }
            
            # Handle SSL configuration for newer Neo4j driver versions
            if not self.config.verify_ssl:
                # For newer Neo4j drivers, use trust constants
                from neo4j import Config, TRUST_SYSTEM_CA_SIGNED_CERTIFICATES
                driver_kwargs["config"] = Config(
                    trust=TRUST_SYSTEM_CA_SIGNED_CERTIFICATES
                )
            elif self.config.trust_all_certificates:
                # Trust all certificates (for self-signed)
                from neo4j import Config, TRUST_ALL_CERTIFICATES
                driver_kwargs["config"] = Config(
                    trust=TRUST_ALL_CERTIFICATES
                )
            
            self.driver = AsyncGraphDatabase.driver(self.config.uri, **driver_kwargs)
            
            # Test the connection
            await self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4J at {self.config.uri}")
            
            # Ensure database exists
            await self._ensure_database_exists()
            
            # Now initialize base class with driver
            super().__init__(self.driver, self.config.database)

        except Exception as e:
            logger.error(f"Failed to connect to Neo4J: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Neo4J database."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4J database")

    async def _ensure_database_exists(self) -> None:
        """Ensure the specified database exists, create it if it doesn't."""
        try:
            # First try to connect to the system database to check/create the target database
            # This works in Neo4j Enterprise Edition
            async with self.driver.session(database="system") as session:
                # Check if database exists
                result = await session.run(
                    "SHOW DATABASES YIELD name WHERE name = $db_name",
                    {"db_name": self.config.database}
                )
                
                databases = [record async for record in result]
                
                if not databases:
                    # Database doesn't exist, try to create it
                    logger.info(f"Database '{self.config.database}' not found, attempting to create it")
                    await session.run(f"CREATE DATABASE `{self.config.database}`")
                    logger.info(f"Created database '{self.config.database}'")
                else:
                    logger.info(f"Database '{self.config.database}' already exists")
                    
        except Exception as e:
            # If we can't access system database or create databases, 
            # try to connect directly to the target database
            try:
                async with self.driver.session(database=self.config.database) as session:
                    # Simple query to test database access
                    await session.run("RETURN 1")
                    logger.info(f"Successfully connected to database '{self.config.database}'")
            except Exception as direct_error:
                if "database does not exist" in str(direct_error).lower():
                    logger.error(f"Database '{self.config.database}' does not exist and cannot be created automatically. "
                               f"Please either: 1) Create the database manually, 2) Use the default 'neo4j' database, "
                               f"or 3) Use Neo4j Enterprise Edition for automatic database creation.")
                raise direct_error

    async def create_database_if_not_exists(self, database_name: str) -> bool:
        """
        Manually create a database if it doesn't exist.

        Args:
            database_name: Name of the database to create

        Returns:
            True if database was created or already exists, False otherwise
        """
        try:
            async with self.driver.session(database="system") as session:
                # Check if database exists
                result = await session.run(
                    "SHOW DATABASES YIELD name WHERE name = $db_name",
                    {"db_name": database_name}
                )
                
                databases = [record async for record in result]
                
                if not databases:
                    # Create the database
                    await session.run(f"CREATE DATABASE `{database_name}`")
                    logger.info(f"Created database '{database_name}'")
                    return True
                else:
                    logger.info(f"Database '{database_name}' already exists")
                    return True

        except Exception as e:
            logger.error(f"Failed to create database {database_name}: {e}")
            return False
