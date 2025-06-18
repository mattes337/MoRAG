# Task 1.2: Document and Chunk ID Standardization

## Overview

Standardize document and chunk ID usage across Neo4j and Qdrant storage systems. This task builds upon the unified ID architecture to ensure consistent identification and cross-system compatibility.

## Objectives

- Update storage classes to use unified document and chunk IDs
- Implement consistent ID handling in Qdrant vector operations
- Ensure Neo4j queries work with new ID formats
- Create validation mechanisms for ID consistency
- Establish cross-system ID verification

## Dependencies

- Task 1.1: Unified ID Architecture (must be completed first)
- Existing `morag-graph` storage implementations

## Implementation Plan

### Step 1: Update Neo4j Storage Implementation

Modify `src/morag_graph/storage/neo4j_storage.py`:

```python
# Add imports
from ..utils.id_generation import UnifiedIDGenerator, IDValidator
from ..models.document import Document
from ..models.document_chunk import DocumentChunk

class Neo4jStorage:
    # ... existing methods ...
    
    async def store_document_with_unified_id(self, document: Document) -> str:
        """Store document with unified ID format.
        
        Args:
            document: Document instance with unified ID
            
        Returns:
            Document ID
        """
        # Validate ID format
        if not IDValidator.validate_document_id(document.id):
            raise ValueError(f"Invalid document ID format: {document.id}")
        
        # Store document with unified ID
        query = """
        MERGE (d:Document {id: $id})
        SET d.source_file = $source_file,
            d.file_name = $file_name,
            d.checksum = $checksum,
            d.ingestion_timestamp = $ingestion_timestamp,
            d.metadata = $metadata,
            d.unified_id_format = true
        RETURN d.id as document_id
        """
        
        result = await self.execute_query(
            query,
            id=document.id,
            source_file=document.source_file,
            file_name=document.file_name,
            checksum=document.checksum,
            ingestion_timestamp=document.ingestion_timestamp.isoformat(),
            metadata=document.metadata or {}
        )
        
        return result[0]['document_id']
    
    async def store_chunk_with_unified_id(self, chunk: DocumentChunk) -> str:
        """Store document chunk with unified ID format.
        
        Args:
            chunk: DocumentChunk instance with unified ID
            
        Returns:
            Chunk ID
        """
        # Validate ID formats
        if not IDValidator.validate_chunk_id(chunk.id):
            raise ValueError(f"Invalid chunk ID format: {chunk.id}")
        
        if not IDValidator.validate_document_id(chunk.document_id):
            raise ValueError(f"Invalid document ID format: {chunk.document_id}")
        
        # Verify document exists
        doc_check = await self.execute_query(
            "MATCH (d:Document {id: $doc_id}) RETURN d.id",
            doc_id=chunk.document_id
        )
        
        if not doc_check:
            raise ValueError(f"Document {chunk.document_id} not found")
        
        # Store chunk with unified ID
        query = """
        MATCH (d:Document {id: $document_id})
        MERGE (c:DocumentChunk {id: $id})
        SET c.document_id = $document_id,
            c.chunk_index = $chunk_index,
            c.text = $text,
            c.metadata = $metadata,
            c.unified_id_format = true
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN c.id as chunk_id
        """
        
        result = await self.execute_query(
            query,
            id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            metadata=chunk.metadata or {}
        )
        
        return result[0]['chunk_id']
    
    async def get_document_by_unified_id(self, document_id: str) -> Optional[Document]:
        """Retrieve document by unified ID.
        
        Args:
            document_id: Unified document ID
            
        Returns:
            Document instance or None
        """
        if not IDValidator.validate_document_id(document_id):
            raise ValueError(f"Invalid document ID format: {document_id}")
        
        query = """
        MATCH (d:Document {id: $id})
        RETURN d.id as id,
               d.source_file as source_file,
               d.file_name as file_name,
               d.checksum as checksum,
               d.ingestion_timestamp as ingestion_timestamp,
               d.metadata as metadata
        """
        
        result = await self.execute_query(query, id=document_id)
        
        if not result:
            return None
        
        doc_data = result[0]
        return Document(
            id=doc_data['id'],
            source_file=doc_data['source_file'],
            file_name=doc_data['file_name'],
            checksum=doc_data['checksum'],
            ingestion_timestamp=datetime.fromisoformat(doc_data['ingestion_timestamp']),
            metadata=doc_data['metadata']
        )
    
    async def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document by unified ID.
        
        Args:
            document_id: Unified document ID
            
        Returns:
            List of DocumentChunk instances
        """
        if not IDValidator.validate_document_id(document_id):
            raise ValueError(f"Invalid document ID format: {document_id}")
        
        query = """
        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:DocumentChunk)
        RETURN c.id as id,
               c.document_id as document_id,
               c.chunk_index as chunk_index,
               c.text as text,
               c.metadata as metadata
        ORDER BY c.chunk_index
        """
        
        result = await self.execute_query(query, document_id=document_id)
        
        chunks = []
        for chunk_data in result:
            chunks.append(DocumentChunk(
                id=chunk_data['id'],
                document_id=chunk_data['document_id'],
                chunk_index=chunk_data['chunk_index'],
                text=chunk_data['text'],
                metadata=chunk_data['metadata']
            ))
        
        return chunks
    
    async def validate_id_consistency(self) -> Dict[str, Any]:
        """Validate ID consistency across the database.
        
        Returns:
            Validation report
        """
        # Check for documents with invalid ID formats
        invalid_docs_query = """
        MATCH (d:Document)
        WHERE NOT d.id STARTS WITH 'doc_' OR size(d.id) <> 20
        RETURN count(d) as invalid_document_count
        """
        
        # Check for chunks with invalid ID formats
        invalid_chunks_query = """
        MATCH (c:DocumentChunk)
        WHERE NOT c.id CONTAINS ':chunk:' OR NOT c.id STARTS WITH 'doc_'
        RETURN count(c) as invalid_chunk_count
        """
        
        # Check for orphaned chunks
        orphaned_chunks_query = """
        MATCH (c:DocumentChunk)
        WHERE NOT EXISTS((c)<-[:HAS_CHUNK]-(:Document))
        RETURN count(c) as orphaned_chunk_count
        """
        
        # Check for ID mismatches between chunks and documents
        mismatch_query = """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:DocumentChunk)
        WHERE c.document_id <> d.id
        RETURN count(c) as mismatched_chunk_count
        """
        
        invalid_docs = await self.execute_query(invalid_docs_query)
        invalid_chunks = await self.execute_query(invalid_chunks_query)
        orphaned_chunks = await self.execute_query(orphaned_chunks_query)
        mismatched = await self.execute_query(mismatch_query)
        
        return {
            'invalid_documents': invalid_docs[0]['invalid_document_count'],
            'invalid_chunks': invalid_chunks[0]['invalid_chunk_count'],
            'orphaned_chunks': orphaned_chunks[0]['orphaned_chunk_count'],
            'mismatched_chunks': mismatched[0]['mismatched_chunk_count'],
            'is_consistent': all([
                invalid_docs[0]['invalid_document_count'] == 0,
                invalid_chunks[0]['invalid_chunk_count'] == 0,
                orphaned_chunks[0]['orphaned_chunk_count'] == 0,
                mismatched[0]['mismatched_chunk_count'] == 0
            ])
        }
```

### Step 2: Update Qdrant Storage Implementation

Modify `src/morag_graph/storage/qdrant_storage.py`:

```python
# Add imports
from ..utils.id_generation import UnifiedIDGenerator, IDValidator
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

class QdrantStorage:
    # ... existing methods ...
    
    async def store_chunk_vector_with_unified_id(self, 
                                                chunk_id: str,
                                                vector: List[float],
                                                metadata: Dict[str, Any]) -> str:
        """Store chunk vector with unified ID.
        
        Args:
            chunk_id: Unified chunk ID
            vector: Dense vector embedding
            metadata: Additional metadata
            
        Returns:
            Point ID (same as chunk_id)
        """
        # Validate chunk ID format
        if not IDValidator.validate_chunk_id(chunk_id):
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        
        # Extract document ID from chunk ID
        document_id = UnifiedIDGenerator.extract_document_id_from_chunk(chunk_id)
        chunk_index = UnifiedIDGenerator.extract_chunk_index_from_chunk(chunk_id)
        
        # Prepare enhanced metadata
        enhanced_metadata = {
            'document_id': document_id,
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'neo4j_chunk_id': chunk_id,  # Cross-reference
            'unified_id_format': True,
            **metadata
        }
        
        # Create point with chunk ID as point ID
        point = PointStruct(
            id=chunk_id,
            vector=vector,
            payload=enhanced_metadata
        )
        
        # Store in Qdrant
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        return chunk_id
    
    async def get_vectors_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all vectors for a document by unified ID.
        
        Args:
            document_id: Unified document ID
            
        Returns:
            List of vector points with metadata
        """
        if not IDValidator.validate_document_id(document_id):
            raise ValueError(f"Invalid document ID format: {document_id}")
        
        # Search for vectors with matching document_id
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
        
        result = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=search_filter,
            limit=1000,  # Adjust based on expected chunk count
            with_payload=True,
            with_vectors=True
        )
        
        points = []
        for point in result[0]:
            points.append({
                'id': point.id,
                'vector': point.vector,
                'payload': point.payload
            })
        
        # Sort by chunk_index if available
        points.sort(key=lambda x: x['payload'].get('chunk_index', 0))
        
        return points
    
    async def get_vector_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get vector by unified chunk ID.
        
        Args:
            chunk_id: Unified chunk ID
            
        Returns:
            Vector point data or None
        """
        if not IDValidator.validate_chunk_id(chunk_id):
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        
        try:
            result = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=True,
                with_vectors=True
            )
            
            if result:
                point = result[0]
                return {
                    'id': point.id,
                    'vector': point.vector,
                    'payload': point.payload
                }
            
            return None
            
        except Exception as e:
            # Handle case where point doesn't exist
            return None
    
    async def delete_vectors_by_document_id(self, document_id: str) -> int:
        """Delete all vectors for a document.
        
        Args:
            document_id: Unified document ID
            
        Returns:
            Number of vectors deleted
        """
        if not IDValidator.validate_document_id(document_id):
            raise ValueError(f"Invalid document ID format: {document_id}")
        
        # Get all vector IDs for the document
        vectors = await self.get_vectors_by_document_id(document_id)
        vector_ids = [v['id'] for v in vectors]
        
        if vector_ids:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=vector_ids
            )
        
        return len(vector_ids)
    
    async def validate_cross_system_consistency(self, neo4j_storage) -> Dict[str, Any]:
        """Validate consistency between Qdrant and Neo4j.
        
        Args:
            neo4j_storage: Neo4j storage instance
            
        Returns:
            Consistency report
        """
        # Get all chunk IDs from Neo4j
        neo4j_chunks = await neo4j_storage.execute_query(
            "MATCH (c:DocumentChunk) RETURN c.id as chunk_id"
        )
        neo4j_chunk_ids = {chunk['chunk_id'] for chunk in neo4j_chunks}
        
        # Get all point IDs from Qdrant
        qdrant_result = await self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,  # Adjust based on data size
            with_payload=True
        )
        qdrant_chunk_ids = {point.id for point in qdrant_result[0]}
        
        # Find inconsistencies
        missing_in_qdrant = neo4j_chunk_ids - qdrant_chunk_ids
        missing_in_neo4j = qdrant_chunk_ids - neo4j_chunk_ids
        
        # Check metadata consistency for common chunks
        common_chunks = neo4j_chunk_ids & qdrant_chunk_ids
        metadata_mismatches = []
        
        for chunk_id in list(common_chunks)[:100]:  # Sample check
            # Get Neo4j chunk
            neo4j_chunk = await neo4j_storage.execute_query(
                "MATCH (c:DocumentChunk {id: $id}) RETURN c.document_id as doc_id",
                id=chunk_id
            )
            
            # Get Qdrant point
            qdrant_point = await self.get_vector_by_chunk_id(chunk_id)
            
            if (neo4j_chunk and qdrant_point and 
                neo4j_chunk[0]['doc_id'] != qdrant_point['payload'].get('document_id')):
                metadata_mismatches.append({
                    'chunk_id': chunk_id,
                    'neo4j_doc_id': neo4j_chunk[0]['doc_id'],
                    'qdrant_doc_id': qdrant_point['payload'].get('document_id')
                })
        
        return {
            'neo4j_chunks': len(neo4j_chunk_ids),
            'qdrant_chunks': len(qdrant_chunk_ids),
            'missing_in_qdrant': len(missing_in_qdrant),
            'missing_in_neo4j': len(missing_in_neo4j),
            'metadata_mismatches': len(metadata_mismatches),
            'consistency_score': (
                len(common_chunks) / max(len(neo4j_chunk_ids | qdrant_chunk_ids), 1)
            ),
            'is_consistent': (
                len(missing_in_qdrant) == 0 and 
                len(missing_in_neo4j) == 0 and 
                len(metadata_mismatches) == 0
            ),
            'missing_details': {
                'missing_in_qdrant': list(missing_in_qdrant)[:10],  # Sample
                'missing_in_neo4j': list(missing_in_neo4j)[:10],   # Sample
                'metadata_mismatches': metadata_mismatches[:10]     # Sample
            }
        }
```

### Step 3: Create Document Processing Pipeline

Create `src/morag_graph/processing/unified_document_processor.py`:

```python
import asyncio
from typing import List, Dict, Any, Optional
from ..models.document import Document
from ..models.document_chunk import DocumentChunk
from ..storage.neo4j_storage import Neo4jStorage
from ..storage.qdrant_storage import QdrantStorage
from ..utils.id_generation import UnifiedIDGenerator

class UnifiedDocumentProcessor:
    """Process documents with unified ID architecture."""
    
    def __init__(self, neo4j_storage: Neo4jStorage, qdrant_storage: QdrantStorage):
        self.neo4j = neo4j_storage
        self.qdrant = qdrant_storage
    
    async def process_document(self, 
                             source_file: str,
                             chunks_text: List[str],
                             embeddings: List[List[float]],
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process document with unified ID system.
        
        Args:
            source_file: Path to source document
            chunks_text: List of chunk texts
            embeddings: List of chunk embeddings
            metadata: Optional document metadata
            
        Returns:
            Processing result with IDs and statistics
        """
        if len(chunks_text) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        # Generate document ID
        document_id = UnifiedIDGenerator.generate_document_id(
            source_file=source_file,
            checksum=self._calculate_checksum(source_file)
        )
        
        # Create document
        document = Document(
            id=document_id,
            source_file=source_file,
            file_name=source_file.split('/')[-1],
            checksum=self._calculate_checksum(source_file),
            metadata=metadata or {}
        )
        
        # Store document in Neo4j
        stored_doc_id = await self.neo4j.store_document_with_unified_id(document)
        
        # Process chunks
        chunk_ids = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks_text, embeddings)):
            # Generate chunk ID
            chunk_id = UnifiedIDGenerator.generate_chunk_id(document_id, i)
            
            # Create chunk
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                chunk_index=i,
                text=chunk_text,
                metadata={'embedding_model': 'text-embedding-004'}
            )
            
            # Store chunk in Neo4j
            stored_chunk_id = await self.neo4j.store_chunk_with_unified_id(chunk)
            
            # Store vector in Qdrant
            await self.qdrant.store_chunk_vector_with_unified_id(
                chunk_id=chunk_id,
                vector=embedding,
                metadata={
                    'text': chunk_text,
                    'chunk_length': len(chunk_text),
                    'processing_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            chunk_ids.append(chunk_id)
        
        return {
            'document_id': document_id,
            'chunk_ids': chunk_ids,
            'chunks_processed': len(chunk_ids),
            'neo4j_stored': True,
            'qdrant_stored': True,
            'unified_format': True
        }
    
    async def verify_processing_integrity(self, document_id: str) -> Dict[str, Any]:
        """Verify integrity of processed document.
        
        Args:
            document_id: Unified document ID
            
        Returns:
            Integrity report
        """
        # Check Neo4j storage
        neo4j_doc = await self.neo4j.get_document_by_unified_id(document_id)
        neo4j_chunks = await self.neo4j.get_chunks_by_document_id(document_id)
        
        # Check Qdrant storage
        qdrant_vectors = await self.qdrant.get_vectors_by_document_id(document_id)
        
        # Verify consistency
        neo4j_chunk_ids = {chunk.id for chunk in neo4j_chunks}
        qdrant_chunk_ids = {vector['id'] for vector in qdrant_vectors}
        
        return {
            'document_exists_neo4j': neo4j_doc is not None,
            'neo4j_chunk_count': len(neo4j_chunks),
            'qdrant_vector_count': len(qdrant_vectors),
            'chunk_ids_match': neo4j_chunk_ids == qdrant_chunk_ids,
            'missing_in_qdrant': list(neo4j_chunk_ids - qdrant_chunk_ids),
            'missing_in_neo4j': list(qdrant_chunk_ids - neo4j_chunk_ids),
            'integrity_score': len(neo4j_chunk_ids & qdrant_chunk_ids) / max(len(neo4j_chunk_ids | qdrant_chunk_ids), 1)
        }
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum for deterministic ID generation."""
        import hashlib
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            # For testing or when file is not accessible
            return hashlib.md5(file_path.encode()).hexdigest()
```

## Testing

### Unit Tests

Create `tests/test_document_chunk_standardization.py`:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.morag_graph.processing.unified_document_processor import UnifiedDocumentProcessor
from src.morag_graph.storage.neo4j_storage import Neo4jStorage
from src.morag_graph.storage.qdrant_storage import QdrantStorage

@pytest.mark.asyncio
class TestUnifiedDocumentProcessor:
    
    @pytest.fixture
    def mock_storages(self):
        neo4j_storage = AsyncMock(spec=Neo4jStorage)
        qdrant_storage = AsyncMock(spec=QdrantStorage)
        return neo4j_storage, qdrant_storage
    
    async def test_document_processing(self, mock_storages):
        neo4j_storage, qdrant_storage = mock_storages
        processor = UnifiedDocumentProcessor(neo4j_storage, qdrant_storage)
        
        # Mock successful storage
        neo4j_storage.store_document_with_unified_id.return_value = "doc_1234567890123456"
        neo4j_storage.store_chunk_with_unified_id.return_value = "doc_1234567890123456:chunk:0000"
        qdrant_storage.store_chunk_vector_with_unified_id.return_value = "doc_1234567890123456:chunk:0000"
        
        # Process test document
        result = await processor.process_document(
            source_file="test.pdf",
            chunks_text=["Test chunk 1", "Test chunk 2"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        
        # Verify results
        assert result['chunks_processed'] == 2
        assert result['unified_format'] is True
        assert len(result['chunk_ids']) == 2
        
        # Verify storage calls
        neo4j_storage.store_document_with_unified_id.assert_called_once()
        assert neo4j_storage.store_chunk_with_unified_id.call_count == 2
        assert qdrant_storage.store_chunk_vector_with_unified_id.call_count == 2
    
    async def test_integrity_verification(self, mock_storages):
        neo4j_storage, qdrant_storage = mock_storages
        processor = UnifiedDocumentProcessor(neo4j_storage, qdrant_storage)
        
        # Mock data
        from src.morag_graph.models.document import Document
        from src.morag_graph.models.document_chunk import DocumentChunk
        
        mock_doc = Document(id="doc_1234567890123456", source_file="test.pdf")
        mock_chunks = [
            DocumentChunk(id="doc_1234567890123456:chunk:0000", document_id="doc_1234567890123456", chunk_index=0, text="chunk 1"),
            DocumentChunk(id="doc_1234567890123456:chunk:0001", document_id="doc_1234567890123456", chunk_index=1, text="chunk 2")
        ]
        mock_vectors = [
            {'id': "doc_1234567890123456:chunk:0000", 'vector': [0.1, 0.2], 'payload': {}},
            {'id': "doc_1234567890123456:chunk:0001", 'vector': [0.3, 0.4], 'payload': {}}
        ]
        
        neo4j_storage.get_document_by_unified_id.return_value = mock_doc
        neo4j_storage.get_chunks_by_document_id.return_value = mock_chunks
        qdrant_storage.get_vectors_by_document_id.return_value = mock_vectors
        
        # Verify integrity
        result = await processor.verify_processing_integrity("doc_1234567890123456")
        
        assert result['document_exists_neo4j'] is True
        assert result['neo4j_chunk_count'] == 2
        assert result['qdrant_vector_count'] == 2
        assert result['chunk_ids_match'] is True
        assert result['integrity_score'] == 1.0

@pytest.mark.asyncio
class TestNeo4jStorageStandardization:
    
    async def test_id_validation(self):
        storage = Neo4jStorage(uri="bolt://localhost:7687", user="neo4j", password="test")
        
        # Test invalid document ID
        with pytest.raises(ValueError, match="Invalid document ID format"):
            await storage.store_document_with_unified_id(
                Document(id="invalid_id", source_file="test.pdf")
            )
        
        # Test invalid chunk ID
        with pytest.raises(ValueError, match="Invalid chunk ID format"):
            await storage.store_chunk_with_unified_id(
                DocumentChunk(id="invalid_chunk_id", document_id="doc_1234567890123456", chunk_index=0, text="test")
            )

@pytest.mark.asyncio
class TestQdrantStorageStandardization:
    
    async def test_chunk_vector_storage(self):
        storage = QdrantStorage(host="localhost", port=6333)
        storage.client = AsyncMock()
        
        # Test valid chunk ID
        result = await storage.store_chunk_vector_with_unified_id(
            chunk_id="doc_1234567890123456:chunk:0000",
            vector=[0.1, 0.2, 0.3],
            metadata={'text': 'test chunk'}
        )
        
        assert result == "doc_1234567890123456:chunk:0000"
        storage.client.upsert.assert_called_once()
        
        # Verify point structure
        call_args = storage.client.upsert.call_args
        points = call_args[1]['points']
        assert len(points) == 1
        assert points[0].id == "doc_1234567890123456:chunk:0000"
        assert points[0].payload['document_id'] == "doc_1234567890123456"
        assert points[0].payload['chunk_index'] == 0
```

### Integration Tests

Create `tests/test_cross_system_consistency.py`:

```python
import pytest
import asyncio
from src.morag_graph.storage.neo4j_storage import Neo4jStorage
from src.morag_graph.storage.qdrant_storage import QdrantStorage
from src.morag_graph.processing.unified_document_processor import UnifiedDocumentProcessor

@pytest.mark.asyncio
@pytest.mark.integration
class TestCrossSystemConsistency:
    
    @pytest.fixture
    async def storages(self):
        # Setup real storage connections for integration testing
        neo4j = Neo4jStorage(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test"
        )
        qdrant = QdrantStorage(
            host="localhost",
            port=6333,
            collection_name="test_collection"
        )
        
        await neo4j.connect()
        await qdrant.ensure_collection()
        
        yield neo4j, qdrant
        
        # Cleanup
        await neo4j.disconnect()
    
    async def test_end_to_end_processing(self, storages):
        neo4j, qdrant = storages
        processor = UnifiedDocumentProcessor(neo4j, qdrant)
        
        # Process test document
        result = await processor.process_document(
            source_file="integration_test.pdf",
            chunks_text=["Integration test chunk 1", "Integration test chunk 2"],
            embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )
        
        # Verify processing
        assert result['chunks_processed'] == 2
        document_id = result['document_id']
        
        # Verify integrity
        integrity = await processor.verify_processing_integrity(document_id)
        assert integrity['integrity_score'] == 1.0
        assert integrity['chunk_ids_match'] is True
        
        # Verify cross-system consistency
        consistency = await qdrant.validate_cross_system_consistency(neo4j)
        assert consistency['is_consistent'] is True
        
        # Cleanup
        await qdrant.delete_vectors_by_document_id(document_id)
        await neo4j.execute_query(
            "MATCH (d:Document {id: $id}) DETACH DELETE d",
            id=document_id
        )
```

## Performance Benchmarks

Create `benchmarks/test_id_standardization_performance.py`:

```python
import asyncio
import time
from typing import List
from src.morag_graph.processing.unified_document_processor import UnifiedDocumentProcessor

class IDStandardizationBenchmark:
    
    def __init__(self, processor: UnifiedDocumentProcessor):
        self.processor = processor
    
    async def benchmark_document_processing(self, num_documents: int, chunks_per_doc: int) -> Dict[str, float]:
        """Benchmark document processing with unified IDs."""
        start_time = time.time()
        
        tasks = []
        for i in range(num_documents):
            chunks_text = [f"Chunk {j} of document {i}" for j in range(chunks_per_doc)]
            embeddings = [[0.1] * 384 for _ in range(chunks_per_doc)]
            
            task = self.processor.process_document(
                source_file=f"benchmark_doc_{i}.pdf",
                chunks_text=chunks_text,
                embeddings=embeddings
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        total_chunks = num_documents * chunks_per_doc
        
        return {
            'total_time': total_time,
            'documents_per_second': num_documents / total_time,
            'chunks_per_second': total_chunks / total_time,
            'avg_time_per_document': total_time / num_documents,
            'avg_time_per_chunk': total_time / total_chunks
        }
```

## Success Criteria

- [ ] All storage classes use unified ID formats
- [ ] Cross-system ID consistency validation passes
- [ ] Document processing pipeline works with unified IDs
- [ ] Performance benchmarks meet requirements (>100 chunks/second)
- [ ] Integration tests pass with real databases
- [ ] ID validation prevents invalid formats
- [ ] Backward compatibility maintained for existing data

## Rollback Plan

1. **Database Backup**: Create full backups before deployment
2. **Feature Flags**: Use configuration to enable/disable unified ID format
3. **Gradual Migration**: Process new documents with unified IDs while maintaining old format support
4. **Monitoring**: Track ID format distribution and consistency metrics

## Next Steps

After completing this task:
1. Proceed to Task 1.3: Entity ID Integration
2. Update all existing processing pipelines
3. Plan production migration timeline
4. Implement monitoring and alerting

---

**Estimated Time**: 3-4 days  
**Dependencies**: Task 1.1 (Unified ID Architecture)  
**Risk Level**: Medium (requires careful testing with real data)