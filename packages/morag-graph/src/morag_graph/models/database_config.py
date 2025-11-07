"""Database configuration models for multi-database support."""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class DatabaseType(str, Enum):
    """Supported database types."""
    NEO4J = "neo4j"
    QDRANT = "qdrant"


class DatabaseServerConfig(BaseModel):
    """Configuration for a single database server connection."""

    type: DatabaseType = Field(..., description="Database type")
    hostname: Optional[str] = Field(None, description="Database hostname/URI")
    port: Optional[int] = Field(None, description="Database port")
    username: Optional[str] = Field(None, description="Database username")
    password: Optional[str] = Field(None, description="Database password")
    database_name: Optional[str] = Field(None, description="Database name/collection name")

    # Additional configuration options
    config_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional database-specific options")

    def get_connection_key(self) -> str:
        """Generate a unique key for this database connection.

        Used to prevent duplicate connections to the same database.
        Excludes password for security.
        """
        key_parts = [
            self.type.value,
            self.hostname or "default",
            str(self.port) if self.port else "default",
            self.username or "default",
            self.database_name or "default"
        ]
        return ":".join(key_parts)

    def is_default_config(self) -> bool:
        """Check if this uses default configuration (no custom connection details)."""
        return all([
            self.hostname is None,
            self.port is None,
            self.username is None,
            self.password is None,
            self.database_name is None
        ])


# Legacy alias for backward compatibility
DatabaseConfig = DatabaseServerConfig


class DatabaseServerArray(BaseModel):
    """Array of database server configurations for multi-database operations."""

    servers: List[DatabaseServerConfig] = Field(default_factory=list, description="List of database server configurations")

    def get_servers_by_type(self, db_type: DatabaseType) -> List[DatabaseServerConfig]:
        """Get all servers of a specific type."""
        return [server for server in self.servers if server.type == db_type]

    def has_servers_of_type(self, db_type: DatabaseType) -> bool:
        """Check if array contains servers of a specific type."""
        return any(server.type == db_type for server in self.servers)

    def is_empty(self) -> bool:
        """Check if the server array is empty."""
        return len(self.servers) == 0


class DatabaseResult(BaseModel):
    """Result of database ingestion operation."""

    database_type: DatabaseType
    connection_key: str
    success: bool
    entities_count: int = 0
    relations_count: int = 0
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
