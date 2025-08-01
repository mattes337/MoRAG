"""Hybrid retrieval coordinator for combining vector and graph search."""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from ..query import QueryEntityExtractor
from ..query.models import QueryAnalysis
from ..models import Entity
from .models import (
    RetrievalResult, HybridRetrievalConfig, RetrievalError, 
    VectorRetriever, DocumentResult, ExpandedContext
)
from .context_expansion import ContextExpansionEngine

logger = logging.getLogger(__name__)


class HybridRetrievalCoordinator:
    """Coordinator for hybrid retrieval combining vector and graph methods."""
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        context_expansion_engine: ContextExpansionEngine,
        query_entity_extractor: QueryEntityExtractor,
        config: Optional[HybridRetrievalConfig] = None
    ):
        """Initialize the hybrid retrieval coordinator.
        
        Args:
            vector_retriever: Vector-based retrieval system
            context_expansion_engine: Graph context expansion engine
            query_entity_extractor: Query entity extraction and linking
            config: Configuration for hybrid retrieval
        """
        self.vector_retriever = vector_retriever
        self.context_expansion_engine = context_expansion_engine
        self.query_entity_extractor = query_entity_extractor
        self.config = config or HybridRetrievalConfig()
        self.logger = logging.getLogger(__name__)
    
    async def retrieve(
        self, 
        query: str, 
        max_results: int = 10
    ) -> List[RetrievalResult]:
        """Perform hybrid retrieval combining vector and graph methods.
        
        Args:
            query: User query string
            max_results: Maximum number of results to return
            
        Returns:
            List of retrieval results ranked by relevance
            
        Raises:
            RetrievalError: If all retrieval methods fail
        """
        try:
            self.logger.info(f"Starting retrieval for query: {query}")
            
            # Analyze query and extract entities
            query_analysis = await self.query_entity_extractor.extract_and_link_entities(query)
            
            # Perform parallel retrieval
            vector_results, graph_results = await asyncio.gather(
                self._vector_retrieval(query),
                self._graph_retrieval(query_analysis),
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_results, Exception):
                self.logger.warning(f"Vector retrieval failed: {vector_results}")
                vector_results = []
            
            if isinstance(graph_results, Exception):
                self.logger.warning(f"Graph retrieval failed: {graph_results}")
                graph_results = []
            
            self.logger.info(f"Retrieved {len(vector_results)} vector results, {len(graph_results)} graph results")
            
            # Fuse results
            fused_results = await self._fuse_results(
                vector_results, graph_results, query_analysis
            )
            
            # Rank and filter final results
            final_results = self._rank_and_filter_results(
                fused_results, max_results
            )
            
            self.logger.info(f"Returning {len(final_results)} final results")
            return final_results
        
        except Exception as e:
            self.logger.error(f"Error in hybrid retrieval: {str(e)}")
            # Fallback to vector retrieval only
            try:
                self.logger.info("Attempting fallback to vector-only retrieval")
                vector_results = await self._vector_retrieval(query)
                return vector_results[:max_results]
            except Exception as fallback_error:
                self.logger.error(f"Fallback retrieval also failed: {fallback_error}")
                raise RetrievalError(f"All retrieval methods failed: {str(e)}")
    
    async def _vector_retrieval(self, query: str) -> List[RetrievalResult]:
        """Perform traditional vector-based retrieval.
        
        Args:
            query: Search query
            
        Returns:
            List of vector retrieval results
        """
        try:
            vector_docs = await self.vector_retriever.search(
                query, limit=self.config.max_vector_results
            )
            
            results = []
            for doc in vector_docs:
                result = RetrievalResult(
                    content=doc.get('content', ''),
                    source="vector",
                    score=doc.get('score', 0.0),
                    metadata=doc.get('metadata', {}),
                    reasoning="Retrieved via semantic similarity"
                )
                results.append(result)
            
            self.logger.debug(f"Vector retrieval returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {e}")
            return []
    
    async def _graph_retrieval(self, query_analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Perform graph-guided retrieval.
        
        Args:
            query_analysis: Analysis of the user query
            
        Returns:
            List of graph retrieval results
        """
        try:
            if not query_analysis.entities:
                self.logger.debug("No entities found in query, skipping graph retrieval")
                return []
            
            # Expand context using graph
            expanded_context = await self.context_expansion_engine.expand_context(query_analysis)
            
            if not expanded_context.expanded_entities:
                self.logger.debug("No expanded entities found, skipping graph retrieval")
                return []
            
            # Convert expanded entities to retrieval results
            results = []
            for entity in expanded_context.expanded_entities:
                # Get documents associated with this entity
                entity_docs = await self._get_entity_documents(entity)
                
                for doc in entity_docs:
                    result = RetrievalResult(
                        content=doc.content,
                        source="graph",
                        score=self._calculate_graph_relevance_score(entity, expanded_context),
                        metadata={
                            **doc.metadata,
                            "entity_id": entity.id,
                            "entity_type": str(entity.type),
                            "expansion_reasoning": expanded_context.expansion_reasoning
                        },
                        entities=[entity.id],
                        reasoning=f"Retrieved via graph expansion: {expanded_context.expansion_reasoning}"
                    )
                    results.append(result)
            
            # Limit results
            limited_results = results[:self.config.max_graph_results]
            self.logger.debug(f"Graph retrieval returned {len(limited_results)} results")
            return limited_results
            
        except Exception as e:
            self.logger.error(f"Graph retrieval failed: {e}")
            return []
    
    async def _get_entity_documents(self, entity: Entity) -> List[DocumentResult]:
        """Get documents associated with an entity.
        
        Args:
            entity: Entity to find documents for
            
        Returns:
            List of documents mentioning the entity
        """
        # This is a placeholder implementation
        # In a real system, this would query the document store for documents
        # that mention this entity, possibly using the entity's mentioned_in_chunks
        
        # For now, return a mock document
        if hasattr(entity, 'source_doc_id') and entity.source_doc_id:
            return [DocumentResult(
                content=f"Document content mentioning {entity.name}",
                score=0.8,
                metadata={
                    "source_doc_id": entity.source_doc_id,
                    "entity_name": entity.name,
                    "entity_type": str(entity.type)
                }
            )]
        
        return []
    
    async def _fuse_results(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        query_analysis: QueryAnalysis
    ) -> List[RetrievalResult]:
        """Fuse vector and graph retrieval results.
        
        Args:
            vector_results: Results from vector retrieval
            graph_results: Results from graph retrieval
            query_analysis: Original query analysis
            
        Returns:
            Fused results
        """
        if self.config.fusion_strategy == "weighted_combination":
            return await self._weighted_combination_fusion(
                vector_results, graph_results, query_analysis
            )
        elif self.config.fusion_strategy == "rank_fusion":
            return await self._rank_fusion(vector_results, graph_results)
        elif self.config.fusion_strategy == "adaptive":
            return await self._adaptive_fusion(
                vector_results, graph_results, query_analysis
            )
        else:
            # Default: simple concatenation with deduplication
            return self._simple_fusion(vector_results, graph_results)
    
    async def _weighted_combination_fusion(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        query_analysis: QueryAnalysis
    ) -> List[RetrievalResult]:
        """Fuse results using weighted combination of scores.
        
        Args:
            vector_results: Vector retrieval results
            graph_results: Graph retrieval results
            query_analysis: Query analysis for weight adjustment
            
        Returns:
            Weighted combination of results
        """
        # Adjust weights based on query characteristics
        vector_weight = self.config.vector_weight
        graph_weight = self.config.graph_weight
        
        # Boost graph weight for entity-rich queries
        if len(query_analysis.entities) >= 2:
            graph_weight *= 1.3
            vector_weight *= 0.8
        
        # Normalize weights
        total_weight = vector_weight + graph_weight
        vector_weight /= total_weight
        graph_weight /= total_weight
        
        # Combine and deduplicate results
        all_results = []
        content_seen = set()
        
        # Add vector results with adjusted scores
        for result in vector_results:
            if result.content not in content_seen:
                result.score *= vector_weight
                result.source = "hybrid_vector"
                all_results.append(result)
                content_seen.add(result.content)
        
        # Add graph results with adjusted scores
        for result in graph_results:
            if result.content not in content_seen:
                result.score *= graph_weight
                result.source = "hybrid_graph"
                all_results.append(result)
                content_seen.add(result.content)
            else:
                # Boost score for content found in both methods
                for existing_result in all_results:
                    if existing_result.content == result.content:
                        existing_result.score += result.score * graph_weight * 0.5
                        existing_result.source = "hybrid_both"
                        existing_result.reasoning += f" + {result.reasoning}"
                        break
        
        return all_results
    
    async def _rank_fusion(
        self, 
        vector_results: List[RetrievalResult], 
        graph_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Fuse results using reciprocal rank fusion.
        
        Args:
            vector_results: Vector retrieval results
            graph_results: Graph retrieval results
            
        Returns:
            Rank-fused results
        """
        # Implement reciprocal rank fusion
        k = 60  # RRF parameter
        content_to_results = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            if result.content not in content_to_results:
                content_to_results[result.content] = {
                    'result': result,
                    'vector_rank': rank + 1,
                    'graph_rank': None
                }
        
        # Process graph results
        for rank, result in enumerate(graph_results):
            if result.content in content_to_results:
                content_to_results[result.content]['graph_rank'] = rank + 1
            else:
                content_to_results[result.content] = {
                    'result': result,
                    'vector_rank': None,
                    'graph_rank': rank + 1
                }
        
        # Calculate RRF scores
        fused_results = []
        for content, data in content_to_results.items():
            rrf_score = 0.0
            
            if data['vector_rank'] is not None:
                rrf_score += 1.0 / (k + data['vector_rank'])
            
            if data['graph_rank'] is not None:
                rrf_score += 1.0 / (k + data['graph_rank'])
            
            result = data['result']
            result.score = rrf_score
            result.source = "rrf_fusion"
            fused_results.append(result)
        
        return fused_results
    
    async def _adaptive_fusion(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        query_analysis: QueryAnalysis
    ) -> List[RetrievalResult]:
        """Adaptively fuse results based on query characteristics.
        
        Args:
            vector_results: Vector retrieval results
            graph_results: Graph retrieval results
            query_analysis: Query analysis
            
        Returns:
            Adaptively fused results
        """
        # Simple adaptive strategy - use weighted combination for complex queries
        if query_analysis.complexity_score > 0.6:
            return await self._weighted_combination_fusion(
                vector_results, graph_results, query_analysis
            )
        else:
            return await self._rank_fusion(vector_results, graph_results)
    
    def _simple_fusion(
        self, 
        vector_results: List[RetrievalResult], 
        graph_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Simple fusion by concatenation with deduplication.
        
        Args:
            vector_results: Vector retrieval results
            graph_results: Graph retrieval results
            
        Returns:
            Concatenated and deduplicated results
        """
        all_results = []
        content_seen = set()
        
        # Add all results, avoiding duplicates
        for result in vector_results + graph_results:
            if result.content not in content_seen:
                all_results.append(result)
                content_seen.add(result.content)
        
        return all_results
    
    def _calculate_graph_relevance_score(self, entity: Entity, context: ExpandedContext) -> float:
        """Calculate relevance score for graph-retrieved content.
        
        Args:
            entity: Entity associated with the content
            context: Expanded context information
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        base_score = 0.5
        
        # Boost for entities in original query
        if entity.id in context.original_entities:
            base_score += 0.3
        
        # Boost based on entity confidence
        if hasattr(entity, 'confidence'):
            base_score += entity.confidence * 0.1
        
        # Context quality bonus
        base_score += context.context_score * 0.2
        
        return min(base_score, 1.0)
    
    def _rank_and_filter_results(
        self, 
        results: List[RetrievalResult], 
        max_results: int
    ) -> List[RetrievalResult]:
        """Rank and filter final results.
        
        Args:
            results: Results to rank and filter
            max_results: Maximum number of results to return
            
        Returns:
            Ranked and filtered results
        """
        # Filter by minimum confidence
        filtered_results = [
            r for r in results 
            if r.score >= self.config.min_confidence_threshold
        ]
        
        # Sort by score
        sorted_results = sorted(
            filtered_results, 
            key=lambda r: r.score, 
            reverse=True
        )
        
        return sorted_results[:max_results]
