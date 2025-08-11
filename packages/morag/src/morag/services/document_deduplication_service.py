"""Document deduplication service using user-provided document IDs."""

import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import structlog

from morag_services.storage import QdrantVectorStorage
from morag_graph.storage.neo4j_storage import Neo4jStorage

logger = structlog.get_logger(__name__)


class DocumentDeduplicationService:
    """Service for managing document deduplication using user-provided IDs."""
    
    def __init__(self, vector_storage: QdrantVectorStorage, graph_storage: Optional[Neo4jStorage] = None):
        """Initialize deduplication service.
        
        Args:
            vector_storage: Qdrant vector storage instance
            graph_storage: Optional Neo4j graph storage instance
        """
        self.vector_storage = vector_storage
        self.graph_storage = graph_storage
    
    def validate_document_id(self, document_id: str) -> bool:
        """Validate document ID format.
        
        Args:
            document_id: Document ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not document_id or not isinstance(document_id, str):
            return False
        
        # Basic validation rules
        if len(document_id) < 1 or len(document_id) > 255:
            return False
        
        # Check for invalid characters (basic security)
        invalid_chars = ['<', '>', '"', "'", '&', '\n', '\r', '\t']
        if any(char in document_id for char in invalid_chars):
            return False
        
        return True
    
    def generate_document_id(self) -> str:
        """Generate a new UUID-based document ID.
        
        Returns:
            Generated document ID
        """
        return str(uuid.uuid4())
    
    async def check_document_exists(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Check if a document with the given ID already exists.
        
        Args:
            document_id: Document ID to check
            
        Returns:
            Document metadata if exists, None otherwise
        """
        try:
            # Search in Qdrant for documents with this ID
            search_results = await self.vector_storage.search_by_metadata(
                filter_dict={"document_id": document_id},
                limit=1
            )
            
            if search_results:
                point = search_results[0]
                return {
                    "document_id": document_id,
                    "created_at": point.payload.get("created_at"),
                    "status": "completed",  # If it exists in vector DB, it's completed
                    "chunks_count": await self._count_document_chunks(document_id),
                    "metadata": {
                        "original_filename": point.payload.get("original_filename"),
                        "content_type": point.payload.get("content_type"),
                        "file_size_bytes": point.payload.get("file_size_bytes")
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error("Failed to check document existence",
                        document_id=document_id,
                        error=str(e))
            return None
    
    async def _count_document_chunks(self, document_id: str) -> int:
        """Count the number of chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of chunks
        """
        try:
            search_results = await self.vector_storage.search_by_metadata(
                filter_dict={"document_id": document_id},
                limit=1000  # Reasonable limit for counting
            )
            return len(search_results)
        except Exception:
            return 0
    
    async def get_document_stats(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive statistics for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document statistics if exists, None otherwise
        """
        existing_doc = await self.check_document_exists(document_id)
        if not existing_doc:
            return None
        
        try:
            # Get all chunks for this document
            chunks = await self.vector_storage.search_by_metadata(
                filter_dict={"document_id": document_id},
                limit=1000
            )
            
            # Calculate statistics
            total_text_length = 0
            chunk_types = set()
            
            for chunk in chunks:
                if "text" in chunk.payload:
                    total_text_length += len(chunk.payload["text"])
                if "chunk_type" in chunk.payload:
                    chunk_types.add(chunk.payload["chunk_type"])
            
            # Get facts count from graph database if available
            facts_count = 0
            keywords_count = 0
            
            if self.graph_storage:
                try:
                    # Query Neo4j for facts related to this document
                    facts_query = """
                    MATCH (d:Document {id: $document_id})-[:CONTAINS]->(c:DocumentChunk)-[:SUPPORTS]->(f:Fact)
                    RETURN count(f) as facts_count
                    """
                    facts_result = await self.graph_storage.execute_query(facts_query, {"document_id": document_id})
                    if facts_result:
                        facts_count = facts_result[0].get("facts_count", 0)
                    
                    # Query for keywords/entities
                    keywords_query = """
                    MATCH (d:Document {id: $document_id})-[:CONTAINS]->(c:DocumentChunk)-[:MENTIONS]->(e:Entity)
                    RETURN count(DISTINCT e) as keywords_count
                    """
                    keywords_result = await self.graph_storage.execute_query(keywords_query, {"document_id": document_id})
                    if keywords_result:
                        keywords_count = keywords_result[0].get("keywords_count", 0)
                        
                except Exception as e:
                    logger.warning("Failed to get graph statistics", document_id=document_id, error=str(e))
            
            return {
                **existing_doc,
                "facts_count": facts_count,
                "keywords_count": keywords_count,
                "chunks_count": len(chunks),
                "total_text_length": total_text_length,
                "chunk_types": list(chunk_types)
            }
            
        except Exception as e:
            logger.error("Failed to get document statistics",
                        document_id=document_id,
                        error=str(e))
            return existing_doc
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all data for a document.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from vector storage
            await self.vector_storage.delete_by_metadata({"document_id": document_id})
            
            # Delete from graph storage if available
            if self.graph_storage:
                delete_query = """
                MATCH (d:Document {id: $document_id})
                DETACH DELETE d
                """
                await self.graph_storage.execute_query(delete_query, {"document_id": document_id})
            
            logger.info("Document deleted successfully", document_id=document_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete document",
                        document_id=document_id,
                        error=str(e))
            return False
    
    def create_duplicate_error_response(self, existing_document: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standardized duplicate document error response.
        
        Args:
            existing_document: Existing document metadata
            
        Returns:
            Error response dictionary
        """
        document_id = existing_document["document_id"]
        
        return {
            "error": "duplicate_document",
            "message": f"Document with ID '{document_id}' already exists",
            "existing_document": {
                "document_id": document_id,
                "created_at": existing_document.get("created_at"),
                "status": existing_document.get("status", "completed"),
                "facts_count": existing_document.get("facts_count", 0),
                "keywords_count": existing_document.get("keywords_count", 0),
                "chunks_count": existing_document.get("chunks_count", 0)
            },
            "options": {
                "update_url": f"/api/process/update/{document_id}",
                "version_url": f"/api/process/version/{document_id}",
                "delete_url": f"/api/process/delete/{document_id}"
            }
        }


# Global service instance
_deduplication_service = None


async def get_deduplication_service() -> DocumentDeduplicationService:
    """Get global deduplication service instance."""
    global _deduplication_service
    if _deduplication_service is None:
        # Initialize with default storage connections
        from morag_services.storage import QdrantVectorStorage
        import os
        
        # Initialize Qdrant storage
        qdrant_url = os.getenv('QDRANT_URL')
        if qdrant_url:
            vector_storage = QdrantVectorStorage(
                host=qdrant_url,
                api_key=os.getenv('QDRANT_API_KEY'),
                collection_name=os.getenv('QDRANT_COLLECTION_NAME', 'morag_documents')
            )
        else:
            vector_storage = QdrantVectorStorage(
                host=os.getenv('QDRANT_HOST', 'localhost'),
                port=int(os.getenv('QDRANT_PORT', '6333')),
                api_key=os.getenv('QDRANT_API_KEY'),
                collection_name=os.getenv('QDRANT_COLLECTION_NAME', 'morag_documents')
            )
        
        await vector_storage.connect()
        
        # Initialize Neo4j storage if available
        graph_storage = None
        neo4j_uri = os.getenv('NEO4J_URI')
        if neo4j_uri:
            try:
                from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
                neo4j_config = Neo4jConfig(
                    uri=neo4j_uri,
                    username=os.getenv('NEO4J_USERNAME', 'neo4j'),
                    password=os.getenv('NEO4J_PASSWORD', ''),
                    database=os.getenv('NEO4J_DATABASE', 'neo4j')
                )
                graph_storage = Neo4jStorage(neo4j_config)
                await graph_storage.connect()
            except Exception as e:
                logger.warning("Failed to initialize Neo4j storage for deduplication", error=str(e))
        
        _deduplication_service = DocumentDeduplicationService(vector_storage, graph_storage)
    
    return _deduplication_service
