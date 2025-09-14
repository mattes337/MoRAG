"""Response builder utilities for enhanced API."""

import uuid
from typing import Dict, Any, List, Optional
import structlog

from morag.models.enhanced_query import (
    EnhancedQueryRequest, EnhancedQueryResponse, EnhancedResult,
    GraphContext, EntityInfo, RelationInfo, FusionStrategy, ExpansionStrategy
)

logger = structlog.get_logger(__name__)


class EnhancedResponseBuilder:
    """Builder for enhanced query responses."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    async def build_response(
        self,
        query_id: str,
        request: EnhancedQueryRequest,
        retrieval_result: Any,  # Result from hybrid retrieval system
        processing_time: float
    ) -> EnhancedQueryResponse:
        """Build enhanced query response from retrieval results.
        
        Args:
            query_id: Unique query identifier
            request: Original query request
            retrieval_result: Results from hybrid retrieval
            processing_time: Processing time in milliseconds
            
        Returns:
            Enhanced query response
        """
        try:
            # Extract results from retrieval_result
            if hasattr(retrieval_result, 'results'):
                raw_results = retrieval_result.results
            elif isinstance(retrieval_result, list):
                raw_results = retrieval_result
            else:
                raw_results = []
            
            # Build enhanced results
            enhanced_results = []
            for i, result in enumerate(raw_results[:request.max_results]):
                enhanced_result = await self._build_enhanced_result(result, i)
                if enhanced_result.relevance_score >= request.min_relevance_score:
                    enhanced_results.append(enhanced_result)
            
            # Build graph context
            graph_context = await self._build_graph_context(
                enhanced_results, 
                request,
                retrieval_result
            )
            
            # Calculate quality metrics
            confidence_score = self._calculate_confidence_score(enhanced_results, graph_context)
            completeness_score = self._calculate_completeness_score(enhanced_results, request)
            
            # Determine actual strategies used
            fusion_strategy_used = self._determine_fusion_strategy_used(request, retrieval_result)
            expansion_strategy_used = self._determine_expansion_strategy_used(request, retrieval_result)
            
            # Build debug info if needed
            debug_info = None
            if hasattr(retrieval_result, 'debug_info'):
                debug_info = retrieval_result.debug_info
            
            return EnhancedQueryResponse(
                query_id=query_id,
                query=request.query,
                results=enhanced_results,
                graph_context=graph_context,
                total_results=len(enhanced_results),
                processing_time_ms=processing_time,
                fusion_strategy_used=fusion_strategy_used,
                expansion_strategy_used=expansion_strategy_used,
                confidence_score=confidence_score,
                completeness_score=completeness_score,
                debug_info=debug_info
            )
            
        except Exception as e:
            self.logger.error("Error building enhanced response", error=str(e))
            # Return minimal response on error
            return EnhancedQueryResponse(
                query_id=query_id,
                query=request.query,
                results=[],
                graph_context=GraphContext(),
                total_results=0,
                processing_time_ms=processing_time,
                fusion_strategy_used=request.fusion_strategy,
                expansion_strategy_used=request.expansion_strategy,
                confidence_score=0.0,
                completeness_score=0.0,
                debug_info={"error": str(e)}
            )
    
    async def _build_enhanced_result(self, raw_result: Any, index: int) -> EnhancedResult:
        """Build enhanced result from raw retrieval result."""
        try:
            # Handle different result formats
            if isinstance(raw_result, dict):
                result_id = raw_result.get('id', f"result_{index}")
                content = raw_result.get('content', raw_result.get('text', ''))
                score = raw_result.get('score', raw_result.get('relevance_score', 0.5))
                document_id = raw_result.get('document_id', raw_result.get('source', f"doc_{index}"))
                metadata = raw_result.get('metadata', {})
                source_type = raw_result.get('source_type', 'vector')
            else:
                # Handle object-like results
                result_id = getattr(raw_result, 'id', f"result_{index}")
                content = getattr(raw_result, 'content', getattr(raw_result, 'text', ''))
                score = getattr(raw_result, 'score', getattr(raw_result, 'relevance_score', 0.5))
                document_id = getattr(raw_result, 'document_id', getattr(raw_result, 'source', f"doc_{index}"))
                metadata = getattr(raw_result, 'metadata', {})
                source_type = getattr(raw_result, 'source_type', 'vector')
            
            # Extract graph-specific information if available
            connected_entities = []
            relation_context = []
            reasoning_path = None
            
            if isinstance(raw_result, dict):
                connected_entities = raw_result.get('connected_entities', [])
                relation_context = [
                    RelationInfo(**rel) if isinstance(rel, dict) else rel
                    for rel in raw_result.get('relation_context', [])
                ]
                reasoning_path = raw_result.get('reasoning_path')
            
            return EnhancedResult(
                id=result_id,
                content=content,
                relevance_score=min(max(float(score), 0.0), 1.0),  # Clamp to [0,1]
                source_type=source_type,
                document_id=document_id,
                metadata=metadata,
                connected_entities=connected_entities,
                relation_context=relation_context,
                reasoning_path=reasoning_path
            )
            
        except Exception as e:
            self.logger.warning("Error building enhanced result", error=str(e), index=index)
            return EnhancedResult(
                id=f"result_{index}",
                content="",
                relevance_score=0.0,
                source_type="unknown",
                document_id=f"doc_{index}",
                metadata={"error": str(e)}
            )
    
    async def _build_graph_context(
        self,
        results: List[EnhancedResult],
        request: EnhancedQueryRequest,
        retrieval_result: Any
    ) -> GraphContext:
        """Build graph context from results and request."""
        if not request.include_graph_context:
            return GraphContext()
        
        try:
            entities = {}
            relations = []
            expansion_path = []
            reasoning_steps = None
            
            # Extract entities from results
            for result in results:
                for entity_id in result.connected_entities:
                    if entity_id not in entities:
                        entities[entity_id] = EntityInfo(
                            id=entity_id,
                            name=entity_id,  # Simplified - would normally look up actual name
                            type="unknown",
                            relevance_score=0.5,
                            source_documents=[result.document_id]
                        )
                
                # Add relations from result context
                relations.extend(result.relation_context)
            
            # Extract from retrieval result if available
            if hasattr(retrieval_result, 'graph_context'):
                graph_ctx = retrieval_result.graph_context
                if hasattr(graph_ctx, 'entities'):
                    entities.update(graph_ctx.entities)
                if hasattr(graph_ctx, 'relations'):
                    relations.extend(graph_ctx.relations)
                if hasattr(graph_ctx, 'expansion_path'):
                    expansion_path = graph_ctx.expansion_path
            
            # Build reasoning steps if requested
            if request.include_reasoning_path:
                reasoning_steps = self._build_reasoning_steps(results, request)
            
            return GraphContext(
                entities=entities,
                relations=relations,
                expansion_path=expansion_path,
                reasoning_steps=reasoning_steps
            )
            
        except Exception as e:
            self.logger.warning("Error building graph context", error=str(e))
            return GraphContext()
    
    def _build_reasoning_steps(
        self,
        results: List[EnhancedResult],
        request: EnhancedQueryRequest
    ) -> List[str]:
        """Build reasoning steps for the query."""
        steps = []
        
        steps.append(f"1. Analyzed query: '{request.query}' (type: {request.query_type})")
        
        if request.expansion_strategy != "none":
            steps.append(f"2. Applied {request.expansion_strategy} expansion strategy with depth {request.expansion_depth}")
        
        vector_results = [r for r in results if r.source_type == "vector"]
        graph_results = [r for r in results if r.source_type == "graph"]
        hybrid_results = [r for r in results if r.source_type == "hybrid"]
        
        if vector_results:
            steps.append(f"3. Found {len(vector_results)} vector-based results")
        if graph_results:
            steps.append(f"4. Found {len(graph_results)} graph-based results")
        if hybrid_results:
            steps.append(f"5. Found {len(hybrid_results)} hybrid results")
        
        steps.append(f"6. Applied {request.fusion_strategy} fusion strategy")
        steps.append(f"7. Filtered results by minimum relevance score {request.min_relevance_score}")
        steps.append(f"8. Returned top {len(results)} results")
        
        return steps
    
    def _calculate_confidence_score(
        self,
        results: List[EnhancedResult],
        graph_context: GraphContext
    ) -> float:
        """Calculate overall confidence score for the response."""
        if not results:
            return 0.0
        
        # Base confidence from result scores
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        
        # Boost confidence if we have graph context
        graph_boost = 0.0
        if graph_context.entities:
            graph_boost += 0.1
        if graph_context.relations:
            graph_boost += 0.1
        if graph_context.reasoning_steps:
            graph_boost += 0.05
        
        # Penalty for very few results
        result_penalty = 0.0
        if len(results) < 3:
            result_penalty = 0.1
        
        confidence = min(avg_relevance + graph_boost - result_penalty, 1.0)
        return max(confidence, 0.0)
    
    def _calculate_completeness_score(
        self,
        results: List[EnhancedResult],
        request: EnhancedQueryRequest
    ) -> float:
        """Calculate completeness score based on request fulfillment."""
        score = 0.0
        
        # Base score from number of results vs requested
        if results:
            result_ratio = len(results) / request.max_results
            score += min(result_ratio, 1.0) * 0.5
        
        # Bonus for diverse source types
        source_types = set(r.source_type for r in results)
        if len(source_types) > 1:
            score += 0.2
        
        # Bonus for graph context if requested
        if request.include_graph_context:
            score += 0.2
        
        # Bonus for reasoning path if requested
        if request.include_reasoning_path:
            score += 0.1
        
        return min(score, 1.0)
    
    def _determine_fusion_strategy_used(
        self,
        request: EnhancedQueryRequest,
        retrieval_result: Any
    ) -> FusionStrategy:
        """Determine which fusion strategy was actually used."""
        if hasattr(retrieval_result, 'fusion_strategy_used'):
            return retrieval_result.fusion_strategy_used
        return request.fusion_strategy
    
    def _determine_expansion_strategy_used(
        self,
        request: EnhancedQueryRequest,
        retrieval_result: Any
    ) -> ExpansionStrategy:
        """Determine which expansion strategy was actually used."""
        if hasattr(retrieval_result, 'expansion_strategy_used'):
            return retrieval_result.expansion_strategy_used
        return request.expansion_strategy
