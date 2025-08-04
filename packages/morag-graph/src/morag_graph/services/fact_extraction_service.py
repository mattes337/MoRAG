"""Fact extraction service that integrates with the existing pipeline."""

import asyncio
from typing import List, Dict, Any, Optional
import structlog

from ..extraction.fact_extractor import FactExtractor
from ..extraction.fact_graph_builder import FactGraphBuilder
from ..models.fact import Fact, FactRelation
from ..models.document_chunk import DocumentChunk
from ..storage.neo4j_operations.fact_operations import FactOperations
from ..storage.neo4j_storage import Neo4jStorage


class FactExtractionService:
    """Service for extracting and storing facts from document chunks."""
    
    def __init__(
        self,
        neo4j_storage: Neo4jStorage,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        min_confidence: float = 0.7,
        max_facts_per_chunk: int = 10,
        enable_relationships: bool = True
    ):
        """Initialize fact extraction service.
        
        Args:
            neo4j_storage: Neo4j storage instance
            model_id: LLM model for extraction
            api_key: API key for LLM service
            min_confidence: Minimum confidence threshold
            max_facts_per_chunk: Maximum facts per chunk
            enable_relationships: Whether to extract fact relationships
        """
        self.neo4j_storage = neo4j_storage
        self.enable_relationships = enable_relationships
        
        self.logger = structlog.get_logger(__name__)
        
        # Initialize extraction components
        self.fact_extractor = FactExtractor(
            model_id=model_id,
            api_key=api_key,
            min_confidence=min_confidence,
            max_facts_per_chunk=max_facts_per_chunk
        )
        
        if enable_relationships:
            self.graph_builder = FactGraphBuilder(
                model_id=model_id,
                api_key=api_key
            )
        
        # Initialize storage operations
        self.fact_operations = FactOperations(neo4j_storage.driver)
    
    async def extract_facts_from_chunks(
        self,
        chunks: List[DocumentChunk],
        domain: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """Extract facts from document chunks.
        
        Args:
            chunks: List of document chunks to process
            domain: Optional domain context
            language: Language of the content
            
        Returns:
            Dictionary with extraction results
        """
        if not chunks:
            return {
                'facts': [],
                'relationships': [],
                'statistics': {
                    'chunks_processed': 0,
                    'facts_extracted': 0,
                    'relationships_created': 0
                }
            }
        
        self.logger.info(
            "Starting fact extraction from chunks",
            num_chunks=len(chunks),
            domain=domain,
            language=language
        )
        
        try:
            # Extract facts from all chunks
            all_facts = []
            extraction_context = {
                'domain': domain,
                'language': language
            }
            
            # Process chunks in parallel batches
            batch_size = 5
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_tasks = [
                    self._extract_facts_from_chunk(chunk, extraction_context)
                    for chunk in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        self.logger.warning(
                            "Chunk fact extraction failed",
                            error=str(result)
                        )
                        continue
                    
                    if isinstance(result, list):
                        all_facts.extend(result)
            
            # Store facts in Neo4j
            stored_fact_ids = []
            if all_facts:
                stored_fact_ids = await self.fact_operations.store_facts(all_facts)
            
            # Extract relationships if enabled
            relationships = []
            if self.enable_relationships and len(all_facts) > 1:
                try:
                    graph = await self.graph_builder.build_fact_graph(all_facts)
                    relationships = self._extract_relationships_from_graph(graph)
                    
                    if relationships:
                        await self.fact_operations.store_fact_relations(relationships)
                        
                except Exception as e:
                    self.logger.warning(
                        "Relationship extraction failed",
                        error=str(e)
                    )
            
            statistics = {
                'chunks_processed': len(chunks),
                'facts_extracted': len(all_facts),
                'facts_stored': len(stored_fact_ids),
                'relationships_created': len(relationships)
            }
            
            self.logger.info(
                "Fact extraction completed",
                **statistics
            )
            
            return {
                'facts': all_facts,
                'relationships': relationships,
                'statistics': statistics
            }
            
        except Exception as e:
            self.logger.error(
                "Fact extraction service failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    async def _extract_facts_from_chunk(
        self,
        chunk: DocumentChunk,
        context: Dict[str, Any]
    ) -> List[Fact]:
        """Extract facts from a single chunk.
        
        Args:
            chunk: Document chunk to process
            context: Extraction context
            
        Returns:
            List of extracted facts
        """
        try:
            facts = await self.fact_extractor.extract_facts(
                chunk_text=chunk.content,
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                context=context
            )
            
            self.logger.debug(
                "Facts extracted from chunk",
                chunk_id=chunk.id,
                facts_count=len(facts)
            )
            
            return facts
            
        except Exception as e:
            self.logger.warning(
                "Failed to extract facts from chunk",
                chunk_id=chunk.id,
                error=str(e)
            )
            return []
    
    def _extract_relationships_from_graph(self, graph) -> List[FactRelation]:
        """Extract FactRelation objects from graph edges.
        
        Args:
            graph: Graph object with edges
            
        Returns:
            List of FactRelation objects
        """
        relationships = []
        
        for edge in graph.edges:
            try:
                # Convert graph edge to FactRelation
                relation = FactRelation(
                    source_fact_id=edge.source,
                    target_fact_id=edge.target,
                    relation_type=edge.type,
                    confidence=edge.properties.get('confidence', 0.7),
                    context=edge.properties.get('context', '')
                )
                relationships.append(relation)
                
            except Exception as e:
                self.logger.warning(
                    "Failed to convert graph edge to relation",
                    edge=edge,
                    error=str(e)
                )
                continue
        
        return relationships
    
    async def extract_facts_from_document(
        self,
        document_id: str,
        domain: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """Extract facts from all chunks of a document.
        
        Args:
            document_id: Document ID to process
            domain: Optional domain context
            language: Language of the content
            
        Returns:
            Dictionary with extraction results
        """
        # Get document chunks from storage
        chunks = await self._get_document_chunks(document_id)
        
        if not chunks:
            self.logger.warning(
                "No chunks found for document",
                document_id=document_id
            )
            return {
                'facts': [],
                'relationships': [],
                'statistics': {
                    'chunks_processed': 0,
                    'facts_extracted': 0,
                    'relationships_created': 0
                }
            }
        
        return await self.extract_facts_from_chunks(chunks, domain, language)
    
    async def _get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get document chunks from storage.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of document chunks
        """
        try:
            # Use Neo4j storage to get chunks
            async with self.neo4j_storage.driver.session() as session:
                query = """
                MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:DocumentChunk)
                RETURN c
                ORDER BY c.index
                """
                
                result = await session.run(query, document_id=document_id)
                chunks = []
                
                async for record in result:
                    chunk_data = dict(record["c"])
                    chunk = DocumentChunk(
                        id=chunk_data['id'],
                        document_id=chunk_data['document_id'],
                        content=chunk_data['content'],
                        index=chunk_data.get('index', 0),
                        metadata=chunk_data.get('metadata', {})
                    )
                    chunks.append(chunk)
                
                return chunks
                
        except Exception as e:
            self.logger.error(
                "Failed to get document chunks",
                document_id=document_id,
                error=str(e)
            )
            return []
    
    async def get_facts_by_document(self, document_id: str) -> List[Fact]:
        """Get all facts extracted from a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of facts from the document
        """
        return await self.fact_operations.get_facts_by_document(document_id)
    
    async def search_facts(
        self,
        query_text: str,
        fact_type: Optional[str] = None,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 50
    ) -> List[Fact]:
        """Search facts by text content.
        
        Args:
            query_text: Text to search for
            fact_type: Optional fact type filter
            domain: Optional domain filter
            min_confidence: Minimum confidence threshold
            limit: Maximum results to return
            
        Returns:
            List of matching facts
        """
        return await self.fact_operations.search_facts(
            query_text=query_text,
            fact_type=fact_type,
            domain=domain,
            min_confidence=min_confidence,
            limit=limit
        )
    
    async def get_fact_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored facts.
        
        Returns:
            Dictionary with fact statistics
        """
        return await self.fact_operations.get_fact_statistics()
    
    async def cleanup_low_quality_facts(self, min_confidence: float = 0.5) -> int:
        """Remove facts below confidence threshold.
        
        Args:
            min_confidence: Minimum confidence to keep
            
        Returns:
            Number of facts removed
        """
        try:
            async with self.neo4j_storage.driver.session() as session:
                query = """
                MATCH (f:Fact)
                WHERE f.confidence < $min_confidence
                DETACH DELETE f
                RETURN count(f) as deleted_count
                """
                
                result = await session.run(query, min_confidence=min_confidence)
                record = await result.single()
                
                deleted_count = record["deleted_count"] if record else 0
                
                self.logger.info(
                    "Low quality facts cleaned up",
                    deleted_count=deleted_count,
                    min_confidence=min_confidence
                )
                
                return deleted_count
                
        except Exception as e:
            self.logger.error(
                "Failed to cleanup low quality facts",
                error=str(e)
            )
            return 0
