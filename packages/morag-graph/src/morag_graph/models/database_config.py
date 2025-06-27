"""Database configuration models for multi-database support."""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class DatabaseType(str, Enum):
    """Supported database types."""
    NEO4J = "neo4j"
    QDRANT = "qdrant"


class DatabaseConfig(BaseModel):
    """Configuration for a database connection."""
    
    type: DatabaseType = Field(..., description="Database type")
    hostname: Optional[str] = Field(None, description="Database hostname/URI")
    port: Optional[int] = Field(None, description="Database port")
    username: Optional[str] = Field(None, description="Database username")
    password: Optional[str] = Field(None, description="Database password")
    database_name: Optional[str] = Field(None, description="Database name")
    
    # Additional configuration options
    config_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional database-specific options")
    
    def get_connection_key(self) -> str:
        """Generate a unique key for this database connection.
        
        Used to prevent duplicate ingestion to the same database.
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