# Task 06: Database Server Management

## 📋 Task Overview

**Objective**: Implement database server configuration management allowing users to configure and manage multiple vector database connections (Qdrant, Pinecone, Weaviate, etc.) with proper connection testing and failover support.

**Priority**: High - Required for multi-database support
**Estimated Time**: 1-2 weeks
**Dependencies**: Task 05 (Job Tracking Integration)

## 🎯 Goals

1. Implement database server configuration management
2. Add support for multiple vector database types
3. Create connection testing and validation
4. Implement database server health monitoring
5. Add failover and load balancing capabilities
6. Create database management API endpoints
7. Integrate with existing vector storage system

## 📊 Current State Analysis

### Existing Database Server Model
- **Fields**: ID, name, type, host, port, username, password, api_key, database, collection, is_active, user_id
- **Types**: QDRANT, NEO4J, PINECONE, WEAVIATE, CHROMA
- **Relationships**: User (owner), Databases (logical databases)

### Current MoRAG Vector Storage
- **System**: Single Qdrant instance configuration
- **Configuration**: Environment variables only
- **Connections**: No connection pooling or management
- **Failover**: No failover support

## 🔧 Implementation Plan

### Step 1: Create Database Server Service Layer

**Files to Create**:
```
packages/morag-core/src/morag_core/
├── database_servers/
│   ├── __init__.py
│   ├── models.py          # Pydantic models for database servers
│   ├── service.py         # Database server service logic
│   ├── connectors/        # Database-specific connectors
│   │   ├── __init__.py
│   │   ├── base.py        # Base connector interface
│   │   ├── qdrant.py      # Qdrant connector
│   │   ├── pinecone.py    # Pinecone connector
│   │   └── weaviate.py    # Weaviate connector
│   ├── health.py          # Health monitoring
│   └── manager.py         # Connection management
```

**Implementation Details**:

1. **Database Server Models**:
```python
# packages/morag-core/src/morag_core/database_servers/models.py
"""Database server management models."""

from pydantic import BaseModel, Field, validator, SecretStr
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class DatabaseType(str, Enum):
    QDRANT = "QDRANT"
    PINECONE = "PINECONE"
    WEAVIATE = "WEAVIATE"
    CHROMA = "CHROMA"
    NEO4J = "NEO4J"

class DatabaseServerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    type: DatabaseType
    host: str = Field(..., min_length=1, max_length=255)
    port: int = Field(..., gt=0, le=65535)
    username: Optional[str] = Field(None, max_length=255)
    password: Optional[SecretStr] = None
    api_key: Optional[SecretStr] = None
    database: Optional[str] = Field(None, max_length=255)
    collection: Optional[str] = Field(None, max_length=255)
    config: Optional[Dict[str, Any]] = None  # Additional configuration

class DatabaseServerUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    host: Optional[str] = Field(None, min_length=1, max_length=255)
    port: Optional[int] = Field(None, gt=0, le=65535)
    username: Optional[str] = Field(None, max_length=255)
    password: Optional[SecretStr] = None
    api_key: Optional[SecretStr] = None
    database: Optional[str] = Field(None, max_length=255)
    collection: Optional[str] = Field(None, max_length=255)
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None

class DatabaseServerResponse(BaseModel):
    id: str
    name: str
    type: DatabaseType
    host: str
    port: int
    username: Optional[str]
    database: Optional[str]
    collection: Optional[str]
    is_active: bool
    created_at: datetime
    last_connected: Optional[datetime]
    updated_at: datetime
    user_id: str
    config: Optional[Dict[str, Any]]
    # Note: Sensitive fields (password, api_key) are excluded

class ConnectionTestRequest(BaseModel):
    server_id: Optional[str] = None  # Test existing server
    # Or provide connection details directly
    type: Optional[DatabaseType] = None
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[SecretStr] = None
    api_key: Optional[SecretStr] = None
    database: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class ConnectionTestResponse(BaseModel):
    success: bool
    message: str
    response_time_ms: float
    server_info: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None

class DatabaseServerHealth(BaseModel):
    server_id: str
    name: str
    type: DatabaseType
    status: str  # "healthy", "degraded", "unhealthy"
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    collections_count: Optional[int] = None
    storage_size_mb: Optional[float] = None
```

2. **Base Connector Interface**:
```python
# packages/morag-core/src/morag_core/database_servers/connectors/base.py
"""Base connector interface for vector databases."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)

class VectorDatabaseConnector(ABC):
    """Abstract base class for vector database connectors."""
    
    def __init__(self, host: str, port: int, **kwargs):
        self.host = host
        self.port = port
        self.config = kwargs
        self.client = None
        self.connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to the database."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> Tuple[bool, str, float]:
        """Test connection and return (success, message, response_time_ms)."""
        pass
    
    @abstractmethod
    async def get_server_info(self) -> Dict[str, Any]:
        """Get server information and statistics."""
        pass
    
    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all collections in the database."""
        pass
    
    @abstractmethod
    async def create_collection(self, name: str, config: Dict[str, Any]) -> bool:
        """Create a new collection."""
        pass
    
    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        pass
    
    @abstractmethod
    async def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get collection information and statistics."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status information."""
        try:
            success, message, response_time = await self.test_connection()
            
            health_data = {
                "status": "healthy" if success else "unhealthy",
                "message": message,
                "response_time_ms": response_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if success:
                try:
                    server_info = await self.get_server_info()
                    collections = await self.list_collections()
                    
                    health_data.update({
                        "server_info": server_info,
                        "collections_count": len(collections),
                        "collections": collections[:10]  # Limit to first 10
                    })
                except Exception as e:
                    health_data["status"] = "degraded"
                    health_data["warning"] = f"Connection OK but info retrieval failed: {str(e)}"
            
            return health_data
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "response_time_ms": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
```

3. **Qdrant Connector Implementation**:
```python
# packages/morag-core/src/morag_core/database_servers/connectors/qdrant.py
"""Qdrant vector database connector."""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
import structlog

from .base import VectorDatabaseConnector

logger = structlog.get_logger(__name__)

class QdrantConnector(VectorDatabaseConnector):
    """Qdrant database connector."""
    
    def __init__(self, host: str, port: int, api_key: Optional[str] = None, **kwargs):
        super().__init__(host, port, **kwargs)
        self.api_key = api_key
        self.timeout = kwargs.get('timeout', 30)
    
    async def connect(self) -> bool:
        """Establish connection to Qdrant."""
        try:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                timeout=self.timeout
            )
            
            # Test connection
            await asyncio.to_thread(self.client.get_collections)
            self.connected = True
            
            logger.info("Connected to Qdrant", host=self.host, port=self.port)
            return True
            
        except Exception as e:
            logger.error("Failed to connect to Qdrant", 
                        host=self.host, port=self.port, error=str(e))
            self.connected = False
            return False
    
    async def disconnect(self):
        """Close connection to Qdrant."""
        if self.client:
            try:
                self.client.close()
                self.connected = False
                logger.info("Disconnected from Qdrant", host=self.host, port=self.port)
            except Exception as e:
                logger.warning("Error during Qdrant disconnect", error=str(e))
    
    async def test_connection(self) -> Tuple[bool, str, float]:
        """Test connection to Qdrant."""
        start_time = time.time()
        
        try:
            if not self.client:
                await self.connect()
            
            # Simple test operation
            collections = await asyncio.to_thread(self.client.get_collections)
            
            response_time = (time.time() - start_time) * 1000
            return True, f"Connected successfully. Found {len(collections.collections)} collections.", response_time
            
        except ResponseHandlingException as e:
            response_time = (time.time() - start_time) * 1000
            return False, f"Qdrant API error: {str(e)}", response_time
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return False, f"Connection failed: {str(e)}", response_time
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get Qdrant server information."""
        if not self.client:
            await self.connect()
        
        try:
            # Get cluster info if available
            cluster_info = await asyncio.to_thread(self.client.cluster_info)
            collections = await asyncio.to_thread(self.client.get_collections)
            
            return {
                "cluster_info": cluster_info,
                "collections_count": len(collections.collections),
                "version": getattr(cluster_info, 'version', 'unknown')
            }
            
        except Exception as e:
            logger.warning("Failed to get Qdrant server info", error=str(e))
            return {"error": str(e)}
    
    async def list_collections(self) -> List[str]:
        """List all collections in Qdrant."""
        if not self.client:
            await self.connect()
        
        try:
            collections = await asyncio.to_thread(self.client.get_collections)
            return [col.name for col in collections.collections]
            
        except Exception as e:
            logger.error("Failed to list Qdrant collections", error=str(e))
            return []
    
    async def create_collection(self, name: str, config: Dict[str, Any]) -> bool:
        """Create a new collection in Qdrant."""
        if not self.client:
            await self.connect()
        
        try:
            from qdrant_client.models import Distance, VectorParams
            
            # Default configuration
            vector_size = config.get('vector_size', 384)
            distance = Distance(config.get('distance', 'COSINE'))
            
            await asyncio.to_thread(
                self.client.create_collection,
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            
            logger.info("Created Qdrant collection", collection=name)
            return True
            
        except Exception as e:
            logger.error("Failed to create Qdrant collection", 
                        collection=name, error=str(e))
            return False
    
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection from Qdrant."""
        if not self.client:
            await self.connect()
        
        try:
            await asyncio.to_thread(self.client.delete_collection, collection_name=name)
            logger.info("Deleted Qdrant collection", collection=name)
            return True
            
        except Exception as e:
            logger.error("Failed to delete Qdrant collection", 
                        collection=name, error=str(e))
            return False
    
    async def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get collection information from Qdrant."""
        if not self.client:
            await self.connect()
        
        try:
            info = await asyncio.to_thread(self.client.get_collection, collection_name=name)
            return {
                "name": name,
                "status": info.status,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "params": info.config.params.__dict__ if info.config.params else None,
                    "hnsw_config": info.config.hnsw_config.__dict__ if info.config.hnsw_config else None,
                    "optimizer_config": info.config.optimizer_config.__dict__ if info.config.optimizer_config else None,
                }
            }
            
        except Exception as e:
            logger.error("Failed to get Qdrant collection info", 
                        collection=name, error=str(e))
            return None
```

4. **Database Server Service**:
```python
# packages/morag-core/src/morag_core/database_servers/service.py
"""Database server management service."""

from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import structlog
from datetime import datetime

from morag_core.database import (
    DatabaseServer, get_database_manager, DatabaseType as DBDatabaseType
)
from .models import (
    DatabaseServerCreate, DatabaseServerUpdate, DatabaseServerResponse,
    ConnectionTestRequest, ConnectionTestResponse, DatabaseServerHealth,
    DatabaseType
)
from .connectors import get_connector
from morag_core.exceptions import NotFoundError, ValidationError, ConflictError

logger = structlog.get_logger(__name__)

class DatabaseServerService:
    """Database server management service."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
    
    def create_server(self, user_id: str, server_data: DatabaseServerCreate) -> DatabaseServerResponse:
        """Create a new database server configuration."""
        with self.db_manager.get_session() as session:
            # Check for duplicate names
            existing = session.query(DatabaseServer).filter(
                and_(
                    DatabaseServer.user_id == user_id,
                    DatabaseServer.name == server_data.name
                )
            ).first()
            
            if existing:
                raise ConflictError(f"Database server '{server_data.name}' already exists")
            
            # Create server
            server = DatabaseServer(
                name=server_data.name,
                type=DBDatabaseType(server_data.type.value),
                host=server_data.host,
                port=server_data.port,
                username=server_data.username,
                password=server_data.password.get_secret_value() if server_data.password else None,
                api_key=server_data.api_key.get_secret_value() if server_data.api_key else None,
                database=server_data.database,
                collection=server_data.collection,
                user_id=user_id,
                is_active=False  # Start inactive until tested
            )
            
            session.add(server)
            session.flush()
            
            logger.info("Database server created", 
                       server_id=server.id, 
                       name=server.name,
                       type=server_data.type.value,
                       user_id=user_id)
            
            return self._server_to_response(server)
    
    def get_server(self, server_id: str, user_id: str) -> Optional[DatabaseServerResponse]:
        """Get database server by ID with user ownership check."""
        with self.db_manager.get_session() as session:
            server = session.query(DatabaseServer).filter(
                and_(
                    DatabaseServer.id == server_id,
                    DatabaseServer.user_id == user_id
                )
            ).first()
            
            if server:
                return self._server_to_response(server)
            return None
    
    def list_servers(self, user_id: str) -> List[DatabaseServerResponse]:
        """List all database servers for user."""
        with self.db_manager.get_session() as session:
            servers = session.query(DatabaseServer).filter(
                DatabaseServer.user_id == user_id
            ).order_by(desc(DatabaseServer.created_at)).all()
            
            return [self._server_to_response(server) for server in servers]
    
    async def test_connection(self, request: ConnectionTestRequest, user_id: str) -> ConnectionTestResponse:
        """Test database connection."""
        try:
            if request.server_id:
                # Test existing server
                server = self.get_server(request.server_id, user_id)
                if not server:
                    return ConnectionTestResponse(
                        success=False,
                        message="Server not found",
                        response_time_ms=0.0
                    )
                
                # Get full server details including sensitive data
                with self.db_manager.get_session() as session:
                    db_server = session.query(DatabaseServer).filter_by(id=request.server_id).first()
                    
                    connector = get_connector(
                        server.type,
                        server.host,
                        server.port,
                        api_key=db_server.api_key,
                        username=db_server.username,
                        password=db_server.password,
                        database=db_server.database
                    )
            else:
                # Test provided connection details
                connector = get_connector(
                    request.type,
                    request.host,
                    request.port,
                    api_key=request.api_key.get_secret_value() if request.api_key else None,
                    username=request.username,
                    password=request.password.get_secret_value() if request.password else None,
                    database=request.database,
                    **(request.config or {})
                )
            
            # Perform connection test
            success, message, response_time = await connector.test_connection()
            
            server_info = None
            if success:
                try:
                    server_info = await connector.get_server_info()
                except Exception as e:
                    logger.warning("Failed to get server info during test", error=str(e))
            
            await connector.disconnect()
            
            # Update last_connected if testing existing server
            if request.server_id and success:
                with self.db_manager.get_session() as session:
                    server = session.query(DatabaseServer).filter_by(id=request.server_id).first()
                    if server:
                        server.last_connected = datetime.utcnow()
            
            return ConnectionTestResponse(
                success=success,
                message=message,
                response_time_ms=response_time,
                server_info=server_info
            )
            
        except Exception as e:
            logger.error("Connection test failed", error=str(e))
            return ConnectionTestResponse(
                success=False,
                message=f"Connection test failed: {str(e)}",
                response_time_ms=0.0,
                error_details=str(e)
            )
    
    def _server_to_response(self, server: DatabaseServer) -> DatabaseServerResponse:
        """Convert DatabaseServer model to DatabaseServerResponse."""
        return DatabaseServerResponse(
            id=server.id,
            name=server.name,
            type=DatabaseType(server.type.value),
            host=server.host,
            port=server.port,
            username=server.username,
            database=server.database,
            collection=server.collection,
            is_active=server.is_active,
            created_at=server.created_at,
            last_connected=server.last_connected,
            updated_at=server.updated_at,
            user_id=server.user_id,
            config={}  # TODO: Add config field to DatabaseServer model
        )
```

## 🧪 Testing Requirements

### Unit Tests
```python
# tests/test_database_server_management.py
import pytest
from morag_core.database_servers import DatabaseServerService
from morag_core.database_servers.models import DatabaseServerCreate, DatabaseType

@pytest.mark.asyncio
async def test_database_server_creation():
    """Test database server creation."""
    service = DatabaseServerService()
    server_data = DatabaseServerCreate(
        name="Test Qdrant",
        type=DatabaseType.QDRANT,
        host="localhost",
        port=6333
    )
    
    server = service.create_server("user123", server_data)
    assert server.name == "Test Qdrant"
    assert server.type == DatabaseType.QDRANT
    assert server.host == "localhost"
    assert server.port == 6333

@pytest.mark.asyncio
async def test_connection_testing():
    """Test database connection testing."""
    service = DatabaseServerService()
    
    # Test with mock Qdrant server
    test_request = ConnectionTestRequest(
        type=DatabaseType.QDRANT,
        host="localhost",
        port=6333
    )
    
    result = await service.test_connection(test_request, "user123")
    # Result depends on whether Qdrant is running
    assert isinstance(result.success, bool)
    assert isinstance(result.response_time_ms, float)
```

## 📋 Acceptance Criteria

- [ ] Database server service with CRUD operations implemented
- [ ] Support for multiple vector database types
- [ ] Connection testing and validation working
- [ ] Database server health monitoring functional
- [ ] Connector architecture for different databases
- [ ] API endpoints for database server management
- [ ] Integration with existing vector storage system
- [ ] Comprehensive unit tests passing
- [ ] Error handling for connection failures
- [ ] Security for sensitive connection data

## 🔄 Next Steps

After completing this task:
1. Proceed to [Task 07: API Key Management](./task-07-api-key-management.md)
2. Integrate database server management with document processing
3. Add database server monitoring dashboard
4. Test with multiple database types

## 📝 Notes

- Implement proper encryption for sensitive connection data
- Add connection pooling for better performance
- Consider implementing database server load balancing
- Add comprehensive logging for connection events
- Implement database server failover mechanisms
