# Task 5: Implement Document Replacement Functionality

## Problem Analysis

### Current Issues
1. **No document replacement capability**: Re-processing the same document creates duplicate entries in vector storage.

2. **No document identification**: System lacks a way to identify and track documents for replacement.

3. **Vector storage accumulation**: Repeated ingestion of the same document creates multiple vector entries.

4. **No cleanup mechanism**: Old versions of documents remain in storage when new versions are processed.

### Current Implementation Analysis
From `packages/morag/src/morag/ingest_tasks.py` and `packages/morag-services/src/morag_services/storage.py`:
- Vector points are created with random UUIDs
- No mechanism to identify existing documents
- No deletion of old vectors before adding new ones
- No document versioning or replacement logic

### Use Cases for Document Replacement
1. **Document updates**: When a document is modified and re-processed
2. **Re-processing with different settings**: Same document with different chunk sizes or strategies
3. **Error correction**: Replacing incorrectly processed documents
4. **Version management**: Updating to newer versions of documents

## Solution Approach

### 1. Document Identification System
- Add document identifier support in API endpoints
- Allow callers to specify custom document IDs
- Generate consistent IDs based on source/content when not provided
- Support both user-defined and auto-generated identifiers

### 2. Replacement Logic Implementation
- Detect existing documents by identifier or source
- Delete old vector points before adding new ones
- Implement atomic replacement operations
- Add versioning and metadata tracking

### 3. API Enhancement
- Add document_id parameter to ingestion endpoints
- Support replacement mode vs append mode
- Add endpoints for document management (list, delete, replace)
- Provide clear documentation on replacement behavior

### 4. Storage Optimization
- Implement efficient document lookup by identifier
- Add batch deletion capabilities
- Optimize vector storage for replacement operations
- Add transaction-like behavior for atomic replacements

## Implementation Plan

### Phase 1: Document Identification Infrastructure
1. **Add document ID support**
   - Add document_id parameter to ingestion APIs
   - Implement ID generation strategies
   - Add ID validation and normalization

2. **Update storage schema**
   - Add document_id field to vector metadata
   - Implement document lookup capabilities
   - Add indexing for efficient document queries

### Phase 2: Replacement Logic
1. **Implement document detection**
   - Add methods to find existing documents
   - Support lookup by ID, source, or content hash
   - Implement efficient document querying

2. **Add deletion capabilities**
   - Implement document deletion by ID
   - Add batch deletion for document chunks
   - Ensure complete cleanup of old vectors

### Phase 3: API Integration
1. **Update ingestion endpoints**
   - Add document_id parameter
   - Add replace_existing parameter
   - Implement replacement workflow

2. **Add document management endpoints**
   - Add document listing endpoint
   - Add document deletion endpoint
   - Add document replacement endpoint

### Phase 4: Testing and Documentation
1. **Comprehensive testing**
   - Test document replacement scenarios
   - Validate no data loss or corruption
   - Test concurrent replacement operations

2. **Documentation and examples**
   - Document replacement API usage
   - Provide examples for common scenarios
   - Add troubleshooting guide

## Technical Implementation

### Document ID Generation
```python
def generate_document_id(source: str, content: Optional[str] = None) -> str:
    """Generate consistent document ID from source and optionally content."""
    
    # For URLs, use normalized URL as base
    if source.startswith(('http://', 'https://')):
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(source)
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    # For files, use filename and optionally content hash
    elif os.path.isfile(source):
        filename = os.path.basename(source)
        if content:
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
            return f"{filename}_{content_hash}"
        return filename
    
    # For other sources, use direct hash
    else:
        return hashlib.sha256(source.encode()).hexdigest()[:16]
```

### Document Replacement Logic
```python
async def replace_document(
    self,
    document_id: str,
    new_vectors: List[List[float]],
    new_metadata: List[Dict[str, Any]],
    collection_name: Optional[str] = None
) -> List[str]:
    """Replace existing document with new vectors."""
    
    target_collection = collection_name or self.collection_name
    
    try:
        # Find existing document chunks
        existing_points = await self.find_document_points(document_id, target_collection)
        
        # Delete existing points
        if existing_points:
            await self.delete_document_points(document_id, target_collection)
            logger.info(
                "Deleted existing document points",
                document_id=document_id,
                points_deleted=len(existing_points)
            )
        
        # Store new vectors with document ID
        for metadata in new_metadata:
            metadata['document_id'] = document_id
            metadata['replaced_at'] = datetime.now(timezone.utc).isoformat()
        
        new_point_ids = await self.store_vectors(
            new_vectors,
            new_metadata,
            target_collection
        )
        
        logger.info(
            "Document replaced successfully",
            document_id=document_id,
            old_points=len(existing_points) if existing_points else 0,
            new_points=len(new_point_ids)
        )
        
        return new_point_ids
        
    except Exception as e:
        logger.error(
            "Failed to replace document",
            document_id=document_id,
            error=str(e)
        )
        raise StorageError(f"Failed to replace document: {str(e)}")
```

### Document Lookup Methods
```python
async def find_document_points(
    self,
    document_id: str,
    collection_name: Optional[str] = None
) -> List[str]:
    """Find all vector points for a document."""
    
    target_collection = collection_name or self.collection_name
    
    try:
        # Search for points with matching document_id
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
        
        # Scroll through all matching points
        points = []
        offset = None
        
        while True:
            result = await asyncio.to_thread(
                self.client.scroll,
                collection_name=target_collection,
                scroll_filter=search_filter,
                limit=100,
                offset=offset,
                with_payload=False  # We only need IDs
            )
            
            batch_points = result[0]
            if not batch_points:
                break
                
            points.extend([point.id for point in batch_points])
            offset = result[1]  # Next offset
            
            if offset is None:
                break
        
        return points
        
    except Exception as e:
        logger.error(
            "Failed to find document points",
            document_id=document_id,
            error=str(e)
        )
        return []
```

### Updated Ingestion API
```python
# packages/morag/src/morag/server.py
@app.post("/api/v1/ingest/file", tags=["Ingestion"])
async def ingest_file(
    file: UploadFile = File(...),
    source_type: Optional[str] = Form(None),
    document_id: Optional[str] = Form(None),
    replace_existing: bool = Form(False),
    webhook_url: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    chunking_strategy: Optional[str] = Form(None)
):
    """Ingest a file with document replacement support."""
    
    try:
        # Generate document ID if not provided
        if not document_id:
            document_id = generate_document_id(file.filename)
        
        # Validate document ID
        if not re.match(r'^[a-zA-Z0-9_-]+$', document_id):
            raise HTTPException(
                status_code=400,
                detail="Document ID must contain only alphanumeric characters, hyphens, and underscores"
            )
        
        # Check if document exists when replace_existing is False
        if not replace_existing:
            existing_points = await vector_storage.find_document_points(document_id)
            if existing_points:
                raise HTTPException(
                    status_code=409,
                    detail=f"Document '{document_id}' already exists. Use replace_existing=true to replace it."
                )
        
        # Process file with replacement support
        task_options = {
            'document_id': document_id,
            'replace_existing': replace_existing,
            'webhook_url': webhook_url or "",
            'metadata': parsed_metadata or {},
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'chunking_strategy': chunking_strategy
        }
        
        # Start ingestion task
        task = ingest_file_task.delay(temp_file_path, task_options)
        
        return {
            "task_id": task.id,
            "document_id": document_id,
            "status": "processing",
            "replace_existing": replace_existing,
            "message": f"File ingestion started for document '{document_id}'"
        }
        
    except Exception as e:
        logger.error("File ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

### Document Management Endpoints
```python
@app.get("/api/v1/documents", tags=["Document Management"])
async def list_documents(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all documents in the vector storage."""
    
    try:
        documents = await get_morag_api().list_documents(limit, offset)
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/documents/{document_id}", tags=["Document Management"])
async def delete_document(document_id: str):
    """Delete a document and all its chunks from vector storage."""
    
    try:
        deleted_count = await get_morag_api().delete_document(document_id)
        
        if deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found"
            )
        
        return {
            "document_id": document_id,
            "deleted_chunks": deleted_count,
            "message": f"Document '{document_id}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

## Files to Modify

### Core Storage
1. `packages/morag-services/src/morag_services/storage.py`
   - Add document lookup methods
   - Implement document replacement logic
   - Add document deletion capabilities
   - Add document listing functionality

### Ingestion Tasks
1. `packages/morag/src/morag/ingest_tasks.py`
   - Add document ID support
   - Implement replacement logic in ingestion
   - Add document ID to vector metadata
   - Handle replacement vs append modes

### API Endpoints
1. `packages/morag/src/morag/server.py`
   - Add document_id parameter to ingestion endpoints
   - Add replace_existing parameter
   - Add document management endpoints
   - Add validation and error handling

### Core API
1. `packages/morag/src/morag/api.py`
   - Add document management methods
   - Implement document listing and deletion
   - Add document replacement support

### Models and Schemas
1. `packages/morag-core/src/morag_core/models/api.py`
   - Add document management request/response models
   - Add document ID validation
   - Add replacement parameters

## Testing Strategy

### Document Replacement Tests
1. **Basic replacement tests**
   - Test replacing existing documents
   - Validate old vectors are deleted
   - Test with different document types

2. **ID generation tests**
   - Test auto-generated document IDs
   - Test custom document IDs
   - Validate ID uniqueness and consistency

### API Integration Tests
1. **Ingestion endpoint tests**
   - Test with document_id parameter
   - Test replace_existing functionality
   - Test error handling for duplicate documents

2. **Document management tests**
   - Test document listing
   - Test document deletion
   - Test document replacement via API

### Concurrency Tests
1. **Concurrent replacement tests**
   - Test multiple replacements of same document
   - Test concurrent ingestion and replacement
   - Validate data consistency

2. **Race condition tests**
   - Test replacement during search operations
   - Test deletion during ingestion
   - Validate atomic operations

## Success Criteria

### Functional Requirements
1. ✅ Documents can be identified by custom or auto-generated IDs
2. ✅ Existing documents can be replaced without duplicates
3. ✅ Old vector points are properly deleted during replacement
4. ✅ Document management endpoints work correctly

### Quality Requirements
1. ✅ No data loss during replacement operations
2. ✅ Atomic replacement behavior (all or nothing)
3. ✅ Consistent document identification across operations
4. ✅ Proper error handling and validation

### Performance Requirements
1. ✅ Efficient document lookup and deletion
2. ✅ Replacement operations complete in reasonable time
3. ✅ No significant impact on search performance
4. ✅ Scalable to large numbers of documents

## Configuration Options

### Environment Variables
```env
# Document replacement settings
MORAG_ENABLE_DOCUMENT_REPLACEMENT=true
MORAG_AUTO_GENERATE_DOCUMENT_IDS=true
MORAG_DOCUMENT_ID_LENGTH=16

# Document management
MORAG_MAX_DOCUMENTS_PER_LIST=1000
MORAG_DOCUMENT_DELETION_BATCH_SIZE=100
```

### API Parameters
- `document_id`: Custom document identifier
- `replace_existing`: Whether to replace existing documents
- `auto_generate_id`: Whether to auto-generate IDs if not provided
- `deletion_strategy`: How to handle existing documents (replace, append, error)

## Risk Mitigation

### Data Integrity Risks
1. **Data loss during replacement**: Replacement might fail and lose data
   - Mitigation: Implement atomic replacement with rollback
   - Validation: Comprehensive testing of replacement scenarios

2. **Orphaned vectors**: Deletion might leave orphaned vectors
   - Mitigation: Implement thorough cleanup and validation
   - Monitoring: Track vector counts and document consistency

### Performance Risks
1. **Slow document lookup**: Finding existing documents might be slow
   - Mitigation: Implement efficient indexing and caching
   - Optimization: Use Qdrant's filtering capabilities effectively

2. **Large deletion operations**: Deleting many chunks might be slow
   - Mitigation: Implement batch deletion with progress tracking
   - Configuration: Configurable batch sizes for deletion

## Implementation Timeline

### Week 1: Infrastructure and Storage
- [ ] Add document ID support to storage layer
- [ ] Implement document lookup and deletion methods
- [ ] Add document replacement logic

### Week 2: API Integration
- [ ] Update ingestion endpoints with document ID support
- [ ] Add document management endpoints
- [ ] Implement replacement workflow

### Week 3: Testing and Validation
- [ ] Comprehensive testing of replacement functionality
- [ ] Concurrency and race condition testing
- [ ] Performance testing and optimization

### Week 4: Documentation and Deployment
- [ ] Document replacement API usage
- [ ] Create examples and tutorials
- [ ] Final testing and deployment preparation

## Next Steps

1. **Start with storage layer**: Implement document lookup and deletion in vector storage
2. **Add ID generation**: Implement consistent document ID generation
3. **Update ingestion**: Add document ID support to ingestion tasks
4. **Test thoroughly**: Validate replacement functionality with various scenarios

This task enables proper document lifecycle management and prevents duplicate entries in vector storage.
