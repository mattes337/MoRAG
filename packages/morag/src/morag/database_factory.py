"""Database connection factory for dynamic database server configurations."""

import os
from typing import Optional, List, Dict, Any, Union
import structlog

from morag_graph import (
    Neo4jStorage, QdrantStorage, Neo4jConfig, QdrantConfig,
    DatabaseServerConfig, DatabaseType
)

logger = structlog.get_logger(__name__)


class DatabaseConnectionFactory:
    """Factory for creating database connections from server configurations."""
    
    @staticmethod
    def create_neo4j_storage(server_config: DatabaseServerConfig) -> Neo4jStorage:
        """Create Neo4j storage from server configuration."""
        if server_config.type != DatabaseType.NEO4J:
            raise ValueError(f"Expected Neo4j server config, got {server_config.type}")
        
        # Use provided config or fall back to environment defaults
        uri = server_config.hostname or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        username = server_config.username or os.getenv("NEO4J_USERNAME", "neo4j")
        password = server_config.password or os.getenv("NEO4J_PASSWORD", "password")
        database = server_config.database_name or os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Handle port if specified
        if server_config.port and "://" in uri:
            # Replace port in URI if specified
            protocol, rest = uri.split("://", 1)
            host_part = rest.split("/")[0].split(":")[0]  # Remove existing port
            uri = f"{protocol}://{host_part}:{server_config.port}"
        
        # Get SSL configuration from config options or environment
        config_options = server_config.config_options or {}
        verify_ssl = config_options.get("verify_ssl", os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true")
        trust_all_certificates = config_options.get("trust_all_certificates", 
                                                   os.getenv("NEO4J_TRUST_ALL_CERTIFICATES", "false").lower() == "true")
        
        neo4j_config = Neo4jConfig(
            uri=uri,
            username=username,
            password=password,
            database=database,
            verify_ssl=verify_ssl,
            trust_all_certificates=trust_all_certificates
        )
        
        logger.info("Creating Neo4j storage", 
                   uri=uri, username=username, database=database)
        return Neo4jStorage(neo4j_config)
    
    @staticmethod
    def create_qdrant_storage(server_config: DatabaseServerConfig) -> QdrantStorage:
        """Create Qdrant storage from server configuration."""
        if server_config.type != DatabaseType.QDRANT:
            raise ValueError(f"Expected Qdrant server config, got {server_config.type}")
        
        # Use provided config or fall back to environment defaults
        host = server_config.hostname or os.getenv("QDRANT_HOST", "localhost")
        port = server_config.port or int(os.getenv("QDRANT_PORT", "6333"))
        api_key = server_config.password or os.getenv("QDRANT_API_KEY")  # Use password field for API key
        collection_name = server_config.database_name or os.getenv("QDRANT_COLLECTION", "morag_vectors")
        
        # Get additional configuration from config options or environment
        config_options = server_config.config_options or {}
        https = config_options.get("https", os.getenv("QDRANT_HTTPS", "false").lower() == "true")
        verify_ssl = config_options.get("verify_ssl", os.getenv("QDRANT_VERIFY_SSL", "true").lower() == "true")
        
        qdrant_config = QdrantConfig(
            host=host,
            port=port,
            https=https,
            api_key=api_key,
            collection_name=collection_name,
            verify_ssl=verify_ssl
        )
        
        logger.info("Creating Qdrant storage", 
                   host=host, port=port, collection=collection_name)
        return QdrantStorage(qdrant_config)
    
    @staticmethod
    def create_storage_from_config(server_config: DatabaseServerConfig) -> Union[Neo4jStorage, QdrantStorage]:
        """Create appropriate storage instance from server configuration."""
        if server_config.type == DatabaseType.NEO4J:
            return DatabaseConnectionFactory.create_neo4j_storage(server_config)
        elif server_config.type == DatabaseType.QDRANT:
            return DatabaseConnectionFactory.create_qdrant_storage(server_config)
        else:
            raise ValueError(f"Unsupported database type: {server_config.type}")


def parse_database_servers(database_servers: Optional[List[Dict[str, Any]]]) -> List[DatabaseServerConfig]:
    """Parse database server configurations from request data."""
    if not database_servers:
        return []
    
    configs = []
    for server_data in database_servers:
        try:
            config = DatabaseServerConfig(**server_data)
            configs.append(config)
        except Exception as e:
            logger.warning("Invalid database server configuration", 
                          server_data=server_data, error=str(e))
            # Skip invalid configurations rather than failing the entire request
            continue
    
    return configs


def get_neo4j_storages(database_servers: Optional[List[Dict[str, Any]]]) -> List[Neo4jStorage]:
    """Get Neo4j storage instances from database server configurations."""
    configs = parse_database_servers(database_servers)
    neo4j_configs = [config for config in configs if config.type == DatabaseType.NEO4J]
    
    storages = []
    for config in neo4j_configs:
        try:
            storage = DatabaseConnectionFactory.create_neo4j_storage(config)
            storages.append(storage)
        except Exception as e:
            logger.error("Failed to create Neo4j storage", 
                        config=config.dict(), error=str(e))
            # Continue with other storages
            continue
    
    return storages


def get_qdrant_storages(database_servers: Optional[List[Dict[str, Any]]]) -> List[QdrantStorage]:
    """Get Qdrant storage instances from database server configurations."""
    configs = parse_database_servers(database_servers)
    qdrant_configs = [config for config in configs if config.type == DatabaseType.QDRANT]
    
    storages = []
    for config in qdrant_configs:
        try:
            storage = DatabaseConnectionFactory.create_qdrant_storage(config)
            storages.append(storage)
        except Exception as e:
            logger.error("Failed to create Qdrant storage", 
                        config=config.dict(), error=str(e))
            # Continue with other storages
            continue
    
    return storages


def get_default_neo4j_storage() -> Optional[Neo4jStorage]:
    """Get default Neo4j storage from environment configuration."""
    try:
        from morag.dependencies import get_neo4j_storage
        return get_neo4j_storage()
    except Exception as e:
        logger.error("Failed to get default Neo4j storage", error=str(e))
        return None


def get_default_qdrant_storage() -> Optional[QdrantStorage]:
    """Get default Qdrant storage from environment configuration."""
    try:
        from morag.dependencies import get_qdrant_storage
        return get_qdrant_storage()
    except Exception as e:
        logger.error("Failed to get default Qdrant storage", error=str(e))
        return None
