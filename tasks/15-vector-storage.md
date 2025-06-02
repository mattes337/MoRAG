# Task 15: Vector Storage Implementation

## Objective
Complete the Qdrant vector storage implementation with missing methods and enhanced functionality for storing embeddings, chunks, and metadata.

## Current Status
- ✅ Basic QdrantService class exists
- ✅ Connection and collection management implemented
- ✅ Basic store_chunks and search_similar methods exist
- ❌ Missing store_embedding method (used by image and YouTube tasks)
- ❌ Missing store_chunk method (used by audio tasks)
- ❌ Missing collection-specific storage methods
- ❌ Missing advanced search and filtering capabilities
- ❌ Missing batch operations optimization
- ❌ Missing metadata indexing

## Implementation Steps

### 1. Add Missing Storage Methods
Extend `src/morag/services/storage.py` with missing methods:

```python
async def store_embedding(
    self,
    embedding: List[float],
    text: str,
    metadata: Dict[str, Any],
    collection_name: Optional[str] = None,
    point_id: Optional[str] = None
) -> str:
    """Store a single embedding with text and metadata."""

async def store_chunk(
    self,
    chunk_id: str,
    text: str,
    summary: str,
    embedding: List[float],
    metadata: Dict[str, Any],
    collection_name: Optional[str] = None
) -> str:
    """Store a text chunk with embedding and metadata."""
```

### 2. Collection Management
Add support for multiple collections:

```python
async def create_collection_if_not_exists(
    self,
    collection_name: str,
    vector_size: int = 768
) -> None:
    """Create collection if it doesn't exist."""

async def list_collections(self) -> List[Dict[str, Any]]:
    """List all collections with their info."""
```

### 3. Advanced Search Features
Enhance search capabilities:

```python
async def search_by_metadata(
    self,
    filters: Dict[str, Any],
    limit: int = 10,
    collection_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search by metadata filters only."""

async def hybrid_search(
    self,
    query_embedding: List[float],
    text_query: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    score_threshold: float = 0.7,
    collection_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Perform hybrid vector and text search."""
```

### 4. Batch Operations
Optimize for large-scale operations:

```python
async def batch_store_embeddings(
    self,
    embeddings_data: List[Dict[str, Any]],
    collection_name: Optional[str] = None,
    batch_size: int = 100
) -> List[str]:
    """Store multiple embeddings in batches."""

async def batch_delete_by_ids(
    self,
    point_ids: List[str],
    collection_name: Optional[str] = None
) -> int:
    """Delete multiple points by IDs."""
```

### 5. Metadata Indexing
Add support for indexed metadata fields:

```python
async def create_payload_index(
    self,
    field_name: str,
    field_type: str = "keyword",
    collection_name: Optional[str] = None
) -> None:
    """Create an index on a payload field for faster filtering."""
```

## Testing Requirements

### Unit Tests
Create `tests/test_15_vector_storage.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from morag.services.storage import QdrantService

class TestVectorStorage:
    @pytest.mark.asyncio
    async def test_store_embedding(self):
        """Test storing single embedding."""
        
    @pytest.mark.asyncio
    async def test_store_chunk(self):
        """Test storing text chunk with embedding."""
        
    @pytest.mark.asyncio
    async def test_collection_management(self):
        """Test collection creation and listing."""
        
    @pytest.mark.asyncio
    async def test_advanced_search(self):
        """Test metadata and hybrid search."""
        
    @pytest.mark.asyncio
    async def test_batch_operations(self):
        """Test batch storage and deletion."""
```

### Integration Tests
Create `tests/test_15_integration.py`:

```python
@pytest.mark.asyncio
async def test_end_to_end_storage():
    """Test complete storage workflow."""
    
@pytest.mark.asyncio
async def test_multiple_collections():
    """Test working with multiple collections."""
    
@pytest.mark.asyncio
async def test_large_batch_operations():
    """Test performance with large batches."""
```

## Success Criteria
- [ ] All missing storage methods implemented
- [ ] Support for multiple collections
- [ ] Advanced search and filtering capabilities
- [ ] Optimized batch operations
- [ ] Metadata indexing support
- [ ] All unit tests pass (>95% coverage)
- [ ] All integration tests pass (>90% coverage)
- [ ] Existing task files work without modification
- [ ] Performance benchmarks meet requirements

## Dependencies
- qdrant-client
- structlog
- asyncio
- uuid
- datetime

## Notes
- Maintain backward compatibility with existing methods
- Optimize for performance with large datasets
- Add proper error handling and logging
- Support both single and batch operations
- Implement proper connection pooling if needed
