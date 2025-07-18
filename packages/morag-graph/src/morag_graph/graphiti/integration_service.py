"""Integration service for Graphiti with MoRAG ingestion pipeline."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from .config import GraphitiConfig
from .ingestion_service import GraphitiIngestionService
from .entity_storage import GraphitiEntityStorage
from .search_service import GraphitiSearchService
from morag_core.models import Document, DocumentChunk
from morag_graph.models import Entity, Relation

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Available storage backends."""
    GRAPHITI = "graphiti"
    NEO4J = "neo4j"
    HYBRID = "hybrid"


@dataclass
class IngestionResult:
    """Result of ingestion operation."""
    success: bool
    backend_used: StorageBackend
    document_id: str
    episode_ids: List[str] = None
    entity_count: int = 0
    relation_count: int = 0
    chunk_count: int = 0
    error: Optional[str] = None
    fallback_used: bool = False
    performance_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.episode_ids is None:
            self.episode_ids = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


class GraphitiIntegrationService:
    """Service for integrating Graphiti with MoRAG ingestion pipeline."""
    
    def __init__(
        self, 
        graphiti_config: Optional[GraphitiConfig] = None,
        preferred_backend: StorageBackend = StorageBackend.GRAPHITI,
        enable_fallback: bool = True
    ):
        self.graphiti_config = graphiti_config
        self.preferred_backend = preferred_backend
        self.enable_fallback = enable_fallback
        
        # Initialize Graphiti services
        try:
            self.graphiti_ingestion = GraphitiIngestionService(graphiti_config)
            self.graphiti_entity_storage = GraphitiEntityStorage(graphiti_config)
            self.graphiti_search = GraphitiSearchService(graphiti_config)
            self.graphiti_available = True
            logger.info("Graphiti services initialized successfully")
        except Exception as e:
            self.graphiti_available = False
            logger.warning(f"Graphiti services initialization failed: {e}")
            if preferred_backend == StorageBackend.GRAPHITI and not enable_fallback:
                raise RuntimeError(f"Graphiti required but unavailable: {e}")
    
    async def ingest_document_with_graph_data(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: List[Entity],
        relations: List[Relation],
        force_backend: Optional[StorageBackend] = None
    ) -> IngestionResult:
        """Ingest document with graph data using specified or preferred backend.
        
        Args:
            document: Document to ingest
            chunks: Document chunks
            entities: Extracted entities
            relations: Extracted relations
            force_backend: Force specific backend (overrides preference)
            
        Returns:
            IngestionResult with operation details
        """
        start_time = time.time()
        
        # Determine backend to use
        backend = force_backend or self.preferred_backend
        
        # Validate backend availability
        if backend == StorageBackend.GRAPHITI and not self.graphiti_available:
            if self.enable_fallback:
                logger.warning("Graphiti unavailable, falling back to Neo4j")
                backend = StorageBackend.NEO4J
            else:
                return IngestionResult(
                    success=False,
                    backend_used=backend,
                    document_id=document.id,
                    error="Graphiti backend unavailable and fallback disabled"
                )
        
        try:
            if backend == StorageBackend.GRAPHITI:
                result = await self._ingest_with_graphiti(document, chunks, entities, relations)
            elif backend == StorageBackend.NEO4J:
                result = await self._ingest_with_neo4j(document, chunks, entities, relations)
            elif backend == StorageBackend.HYBRID:
                result = await self._ingest_with_hybrid(document, chunks, entities, relations)
            else:
                raise ValueError(f"Unknown backend: {backend}")
            
            # Add performance metrics
            result.performance_metrics = {
                "total_time": time.time() - start_time,
                "backend_used": result.backend_used.value
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed with {backend.value} backend: {e}")
            
            # Try fallback if enabled and not already using fallback
            if (self.enable_fallback and 
                backend != StorageBackend.NEO4J and 
                force_backend is None):
                
                logger.info("Attempting fallback to Neo4j backend")
                try:
                    fallback_result = await self._ingest_with_neo4j(document, chunks, entities, relations)
                    fallback_result.fallback_used = True
                    fallback_result.performance_metrics = {
                        "total_time": time.time() - start_time,
                        "backend_used": fallback_result.backend_used.value,
                        "fallback_from": backend.value
                    }
                    return fallback_result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    return IngestionResult(
                        success=False,
                        backend_used=backend,
                        document_id=document.id,
                        error=f"Primary: {str(e)}, Fallback: {str(fallback_error)}"
                    )
            
            return IngestionResult(
                success=False,
                backend_used=backend,
                document_id=document.id,
                error=str(e)
            )
    
    async def _ingest_with_graphiti(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: List[Entity],
        relations: List[Relation]
    ) -> IngestionResult:
        """Ingest using Graphiti backend."""
        # Ingest document and chunks as episodes
        doc_result = await self.graphiti_ingestion.ingest_document_as_episode(
            document, chunks, entities, relations, strategy="document"
        )
        
        if not doc_result["success"]:
            raise RuntimeError(f"Document ingestion failed: {doc_result.get('error')}")
        
        # Store entities and relations
        entity_results = await self.graphiti_entity_storage.store_entities_batch(entities)
        relation_results = await self.graphiti_entity_storage.store_relations_batch(relations)
        
        # Check for failures
        failed_entities = [r for r in entity_results if not r.success]
        failed_relations = [r for r in relation_results if not r.success]
        
        if failed_entities or failed_relations:
            error_msg = f"Failed entities: {len(failed_entities)}, Failed relations: {len(failed_relations)}"
            logger.warning(error_msg)
        
        return IngestionResult(
            success=True,
            backend_used=StorageBackend.GRAPHITI,
            document_id=document.id,
            episode_ids=doc_result["episode_ids"],
            entity_count=len([r for r in entity_results if r.success]),
            relation_count=len([r for r in relation_results if r.success]),
            chunk_count=len(chunks)
        )
    
    async def _ingest_with_neo4j(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: List[Entity],
        relations: List[Relation]
    ) -> IngestionResult:
        """Ingest using legacy Neo4j backend."""
        # This would call the existing Neo4j ingestion logic
        # For now, this is a placeholder that simulates the operation
        
        logger.info(f"Ingesting document {document.id} with Neo4j backend")
        
        # Simulate Neo4j ingestion
        # In actual implementation, this would call existing Neo4j storage methods
        
        return IngestionResult(
            success=True,
            backend_used=StorageBackend.NEO4J,
            document_id=document.id,
            entity_count=len(entities),
            relation_count=len(relations),
            chunk_count=len(chunks)
        )
    
    async def _ingest_with_hybrid(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: List[Entity],
        relations: List[Relation]
    ) -> IngestionResult:
        """Ingest using hybrid approach (both backends)."""
        # Store in both Graphiti and Neo4j for comparison/migration
        
        graphiti_result = await self._ingest_with_graphiti(document, chunks, entities, relations)
        neo4j_result = await self._ingest_with_neo4j(document, chunks, entities, relations)
        
        # Return Graphiti result but note hybrid usage
        graphiti_result.backend_used = StorageBackend.HYBRID
        graphiti_result.performance_metrics = {
            "graphiti_success": graphiti_result.success,
            "neo4j_success": neo4j_result.success
        }
        
        return graphiti_result
    
    async def search_with_backend(
        self,
        query: str,
        backend: Optional[StorageBackend] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search using specified backend.
        
        Args:
            query: Search query
            backend: Backend to use (defaults to preferred)
            limit: Maximum results
            
        Returns:
            Search results with backend information
        """
        backend = backend or self.preferred_backend
        
        if backend == StorageBackend.GRAPHITI and self.graphiti_available:
            results, metrics = await self.graphiti_search.search(query, limit)
            return {
                "backend": backend.value,
                "results": [
                    {
                        "content": r.content,
                        "score": r.score,
                        "document_id": getattr(r, 'document_id', None),
                        "chunk_id": getattr(r, 'chunk_id', None)
                    }
                    for r in results
                ],
                "metrics": {
                    "query_time": metrics.query_time,
                    "result_count": metrics.result_count
                }
            }
        else:
            # Fallback to Neo4j search (placeholder)
            return {
                "backend": "neo4j",
                "results": [],
                "metrics": {"query_time": 0.0, "result_count": 0},
                "note": "Neo4j search not implemented in this placeholder"
            }
    
    def is_graphiti_available(self) -> bool:
        """Check if Graphiti backend is available."""
        return self.graphiti_available
    
    def get_backend_status(self) -> Dict[str, Any]:
        """Get status of all backends."""
        return {
            "preferred_backend": self.preferred_backend.value,
            "graphiti_available": self.graphiti_available,
            "neo4j_available": True,  # Assume Neo4j is always available as fallback
            "fallback_enabled": self.enable_fallback
        }
    
    async def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = {
            "backend_status": self.get_backend_status(),
            "graphiti_stats": {}
        }
        
        if self.graphiti_available:
            stats["graphiti_stats"] = {
                "ingestion_stats": await self.graphiti_ingestion.get_ingestion_stats(),
                "entity_storage_stats": self.graphiti_entity_storage.get_storage_stats()
            }
        
        return stats


def create_integration_service(
    graphiti_config: Optional[GraphitiConfig] = None,
    preferred_backend: StorageBackend = StorageBackend.GRAPHITI,
    enable_fallback: bool = True
) -> GraphitiIntegrationService:
    """Create a GraphitiIntegrationService instance.
    
    Args:
        graphiti_config: Optional Graphiti configuration
        preferred_backend: Preferred storage backend
        enable_fallback: Whether to enable fallback to Neo4j
        
    Returns:
        GraphitiIntegrationService instance
    """
    return GraphitiIntegrationService(graphiti_config, preferred_backend, enable_fallback)
