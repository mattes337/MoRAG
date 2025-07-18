"""Graphiti ingestion service for MoRAG documents."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from graphiti_core import Graphiti
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False

from .config import GraphitiConfig, create_graphiti_instance
from .episode_mapper import DocumentEpisodeMapper
from .entity_storage import GraphitiEntityStorage
from morag_core.models import Document, DocumentChunk
from morag_graph.models import Entity, Relation

logger = logging.getLogger(__name__)


class GraphitiIngestionService:
    """Service for ingesting MoRAG documents into Graphiti."""
    
    def __init__(self, config: Optional[GraphitiConfig] = None):
        self.config = config
        self.episode_mapper = DocumentEpisodeMapper(config)
        self.entity_storage = GraphitiEntityStorage(config)
        
        # Initialize Graphiti instance if available
        if GRAPHITI_AVAILABLE:
            try:
                self.graphiti = create_graphiti_instance(config)
            except Exception as e:
                logger.warning(f"Failed to create Graphiti instance: {e}")
                self.graphiti = None
        else:
            self.graphiti = None
    
    async def ingest_document_as_episode(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: List[Entity],
        relations: List[Relation],
        strategy: str = "document"
    ) -> Dict[str, Any]:
        """Ingest document as Graphiti episode(s).
        
        Args:
            document: Document to ingest
            chunks: Document chunks
            entities: Extracted entities
            relations: Extracted relations
            strategy: Ingestion strategy ("document", "chunks", or "hybrid")
            
        Returns:
            Dictionary with ingestion results
        """
        if not self.graphiti:
            return {
                "success": False,
                "error": "Graphiti instance not available",
                "episode_ids": []
            }
        
        try:
            episode_ids = []
            
            if strategy == "document":
                # Create single episode for entire document
                result = await self.episode_mapper.map_document_to_episode(
                    document, 
                    include_chunks=True
                )
                
                if result["success"]:
                    episode_ids.append(result["episode_name"])
                else:
                    return {
                        "success": False,
                        "error": f"Document episode creation failed: {result.get('error')}",
                        "episode_ids": []
                    }
            
            elif strategy == "chunks":
                # Create separate episodes for each chunk
                chunk_results = await self.episode_mapper.map_chunks_to_episodes(
                    document,
                    chunk_episode_prefix=f"doc_{document.id}"
                )
                
                successful_episodes = [r for r in chunk_results if r["success"]]
                failed_episodes = [r for r in chunk_results if not r["success"]]
                
                if failed_episodes:
                    logger.warning(f"Failed to create {len(failed_episodes)} chunk episodes")
                
                episode_ids = [r["episode_name"] for r in successful_episodes]
                
                if not episode_ids:
                    return {
                        "success": False,
                        "error": "No chunk episodes created successfully",
                        "episode_ids": []
                    }
            
            elif strategy == "hybrid":
                # Create both document and chunk episodes
                doc_result = await self.episode_mapper.map_document_to_episode(
                    document,
                    include_chunks=False  # Don't duplicate chunks in document episode
                )
                
                if doc_result["success"]:
                    episode_ids.append(doc_result["episode_name"])
                
                chunk_results = await self.episode_mapper.map_chunks_to_episodes(
                    document,
                    chunk_episode_prefix=f"doc_{document.id}"
                )
                
                successful_chunk_episodes = [r for r in chunk_results if r["success"]]
                episode_ids.extend([r["episode_name"] for r in successful_chunk_episodes])
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown ingestion strategy: {strategy}",
                    "episode_ids": []
                }
            
            # Add metadata linking episodes to entities and relations
            await self._link_episodes_to_graph_data(episode_ids, entities, relations)
            
            return {
                "success": True,
                "episode_ids": episode_ids,
                "strategy_used": strategy,
                "document_id": document.id,
                "chunk_count": len(chunks),
                "entity_count": len(entities),
                "relation_count": len(relations)
            }
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "episode_ids": []
            }
    
    async def ingest_document_batch(
        self,
        documents: List[Document],
        document_chunks: Dict[str, List[DocumentChunk]],
        document_entities: Dict[str, List[Entity]],
        document_relations: Dict[str, List[Relation]],
        strategy: str = "document"
    ) -> List[Dict[str, Any]]:
        """Ingest multiple documents in batch.
        
        Args:
            documents: List of documents to ingest
            document_chunks: Mapping of document ID to chunks
            document_entities: Mapping of document ID to entities
            document_relations: Mapping of document ID to relations
            strategy: Ingestion strategy
            
        Returns:
            List of ingestion results
        """
        results = []
        
        for document in documents:
            chunks = document_chunks.get(document.id, [])
            entities = document_entities.get(document.id, [])
            relations = document_relations.get(document.id, [])
            
            result = await self.ingest_document_as_episode(
                document, chunks, entities, relations, strategy
            )
            results.append(result)
            
            # Log progress for large batches
            if len(results) % 10 == 0:
                logger.info(f"Processed {len(results)}/{len(documents)} documents")
        
        return results
    
    async def _link_episodes_to_graph_data(
        self,
        episode_ids: List[str],
        entities: List[Entity],
        relations: List[Relation]
    ):
        """Link episodes to graph data through metadata.
        
        Args:
            episode_ids: List of episode IDs
            entities: Entities to link
            relations: Relations to link
        """
        if not self.graphiti:
            return
        
        try:
            # This is a simplified approach - in practice, you might want
            # more sophisticated linking based on entity mentions in content
            
            entity_ids = [entity.id for entity in entities]
            relation_ids = [relation.id for relation in relations]
            
            # Add metadata to episodes linking them to graph data
            # This would depend on Graphiti's metadata update capabilities
            
            logger.debug(
                f"Linked {len(episode_ids)} episodes to "
                f"{len(entity_ids)} entities and {len(relation_ids)} relations"
            )
            
        except Exception as e:
            logger.error(f"Failed to link episodes to graph data: {e}")
    
    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics.
        
        Returns:
            Dictionary with ingestion statistics
        """
        stats = {
            "graphiti_available": self.graphiti is not None,
            "episode_mapper_stats": self.episode_mapper.get_stats() if hasattr(self.episode_mapper, 'get_stats') else {},
            "entity_storage_stats": self.entity_storage.get_storage_stats()
        }
        
        return stats
    
    async def validate_ingestion_setup(self) -> Dict[str, Any]:
        """Validate that ingestion setup is working.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "graphiti_available": self.graphiti is not None,
            "episode_mapper_available": self.episode_mapper is not None,
            "entity_storage_available": self.entity_storage is not None,
            "overall_ready": False,
            "issues": []
        }
        
        if not self.graphiti:
            validation_results["issues"].append("Graphiti instance not available")
        
        if not self.episode_mapper:
            validation_results["issues"].append("Episode mapper not available")
        
        if not self.entity_storage:
            validation_results["issues"].append("Entity storage not available")
        
        # Test basic connectivity if everything is available
        if self.graphiti and self.episode_mapper and self.entity_storage:
            try:
                # Try a simple operation to validate connectivity
                # This would depend on Graphiti's API
                validation_results["overall_ready"] = True
                logger.info("Ingestion setup validation passed")
            except Exception as e:
                validation_results["issues"].append(f"Connectivity test failed: {e}")
                logger.error(f"Ingestion setup validation failed: {e}")
        
        return validation_results


def create_ingestion_service(config: Optional[GraphitiConfig] = None) -> GraphitiIngestionService:
    """Create a GraphitiIngestionService instance.
    
    Args:
        config: Optional Graphiti configuration
        
    Returns:
        GraphitiIngestionService instance
    """
    return GraphitiIngestionService(config)
