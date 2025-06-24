"""Advanced result fusion strategies for hybrid retrieval."""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..query.models import QueryAnalysis
from .models import RetrievalResult, HybridRetrievalConfig

logger = logging.getLogger(__name__)


class FusionStrategy(ABC):
    """Abstract base class for result fusion strategies."""
    
    @abstractmethod
    async def fuse(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        query_analysis: QueryAnalysis,
        config: HybridRetrievalConfig
    ) -> List[RetrievalResult]:
        """Fuse vector and graph results.
        
        Args:
            vector_results: Results from vector retrieval
            graph_results: Results from graph retrieval
            query_analysis: Original query analysis
            config: Retrieval configuration
            
        Returns:
            Fused results
        """
        pass


class WeightedCombinationFusion(FusionStrategy):
    """Weighted combination fusion strategy."""
    
    async def fuse(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        query_analysis: QueryAnalysis,
        config: HybridRetrievalConfig
    ) -> List[RetrievalResult]:
        """Fuse results using weighted combination of scores."""
        # Adjust weights based on query characteristics
        vector_weight = config.vector_weight
        graph_weight = config.graph_weight
        
        # Boost graph weight for entity-rich queries
        if len(query_analysis.entities) >= 2:
            graph_weight *= 1.3
            vector_weight *= 0.8
        
        # Boost graph weight for complex queries
        if query_analysis.complexity_score > 0.7:
            graph_weight *= 1.2
            vector_weight *= 0.9
        
        # Normalize weights
        total_weight = vector_weight + graph_weight
        vector_weight /= total_weight
        graph_weight /= total_weight
        
        logger.debug(f"Using weights: vector={vector_weight:.3f}, graph={graph_weight:.3f}")
        
        # Combine and deduplicate results
        all_results = []
        content_seen = set()
        
        # Add vector results with adjusted scores
        for result in vector_results:
            if result.content not in content_seen:
                new_result = RetrievalResult(
                    content=result.content,
                    source="hybrid_vector",
                    score=result.score * vector_weight,
                    metadata=result.metadata,
                    entities=result.entities,
                    reasoning=result.reasoning
                )
                all_results.append(new_result)
                content_seen.add(result.content)
        
        # Add graph results with adjusted scores
        for result in graph_results:
            if result.content not in content_seen:
                new_result = RetrievalResult(
                    content=result.content,
                    source="hybrid_graph",
                    score=result.score * graph_weight,
                    metadata=result.metadata,
                    entities=result.entities,
                    reasoning=result.reasoning
                )
                all_results.append(new_result)
                content_seen.add(result.content)
            else:
                # Boost score for content found in both methods
                for existing_result in all_results:
                    if existing_result.content == result.content:
                        existing_result.score += result.score * graph_weight * 0.5
                        existing_result.source = "hybrid_both"
                        existing_result.reasoning += f" + {result.reasoning}"
                        if result.entities:
                            existing_result.entities = list(set(
                                (existing_result.entities or []) + result.entities
                            ))
                        break
        
        return all_results


class ReciprocalRankFusion(FusionStrategy):
    """Reciprocal Rank Fusion (RRF) strategy."""
    
    def __init__(self, k: int = 60):
        """Initialize RRF with parameter k.
        
        Args:
            k: RRF parameter (typically 60)
        """
        self.k = k
    
    async def fuse(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        query_analysis: QueryAnalysis,
        config: HybridRetrievalConfig
    ) -> List[RetrievalResult]:
        """Fuse results using reciprocal rank fusion."""
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
                # Merge entities from both sources
                existing_entities = content_to_results[result.content]['result'].entities or []
                new_entities = result.entities or []
                content_to_results[result.content]['result'].entities = list(set(
                    existing_entities + new_entities
                ))
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
                rrf_score += 1.0 / (self.k + data['vector_rank'])
            
            if data['graph_rank'] is not None:
                rrf_score += 1.0 / (self.k + data['graph_rank'])
            
            result = data['result']
            new_result = RetrievalResult(
                content=result.content,
                source="rrf_fusion",
                score=rrf_score,
                metadata=result.metadata,
                entities=result.entities,
                reasoning=f"RRF fusion (k={self.k}): {result.reasoning}"
            )
            fused_results.append(new_result)

        # Sort by RRF score (descending)
        return sorted(fused_results, key=lambda r: r.score, reverse=True)


class AdaptiveFusion(FusionStrategy):
    """Adaptive fusion strategy that chooses method based on query characteristics."""
    
    def __init__(self):
        """Initialize adaptive fusion with sub-strategies."""
        self.weighted_fusion = WeightedCombinationFusion()
        self.rrf_fusion = ReciprocalRankFusion()
    
    async def fuse(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        query_analysis: QueryAnalysis,
        config: HybridRetrievalConfig
    ) -> List[RetrievalResult]:
        """Adaptively choose fusion strategy based on query characteristics."""
        # Use weighted combination for complex queries with many entities
        if query_analysis.complexity_score > 0.6 and len(query_analysis.entities) >= 2:
            logger.debug("Using weighted combination for complex multi-entity query")
            return await self.weighted_fusion.fuse(
                vector_results, graph_results, query_analysis, config
            )
        
        # Use RRF for simpler queries or when results are similar in quality
        elif len(vector_results) > 0 and len(graph_results) > 0:
            logger.debug("Using RRF for balanced result sets")
            return await self.rrf_fusion.fuse(
                vector_results, graph_results, query_analysis, config
            )
        
        # Fallback to weighted combination
        else:
            logger.debug("Using weighted combination as fallback")
            return await self.weighted_fusion.fuse(
                vector_results, graph_results, query_analysis, config
            )


class ResultFusionEngine:
    """Engine for managing and executing result fusion strategies."""
    
    def __init__(self):
        """Initialize fusion engine with available strategies."""
        self.strategies = {
            "weighted_combination": WeightedCombinationFusion(),
            "rank_fusion": ReciprocalRankFusion(),
            "rrf": ReciprocalRankFusion(),
            "adaptive": AdaptiveFusion()
        }
        self.logger = logging.getLogger(__name__)
    
    async def fuse_results(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        query_analysis: QueryAnalysis,
        config: HybridRetrievalConfig
    ) -> List[RetrievalResult]:
        """Fuse results using the configured strategy.
        
        Args:
            vector_results: Results from vector retrieval
            graph_results: Results from graph retrieval
            query_analysis: Original query analysis
            config: Retrieval configuration
            
        Returns:
            Fused and ranked results
        """
        strategy_name = config.fusion_strategy
        
        if strategy_name not in self.strategies:
            self.logger.warning(f"Unknown fusion strategy: {strategy_name}, using weighted_combination")
            strategy_name = "weighted_combination"
        
        strategy = self.strategies[strategy_name]
        
        try:
            self.logger.info(f"Fusing results using {strategy_name} strategy")
            fused_results = await strategy.fuse(
                vector_results, graph_results, query_analysis, config
            )
            
            # Post-process: rank and filter results
            final_results = self._rank_and_filter_results(fused_results, config)
            
            self.logger.info(f"Fusion complete: {len(final_results)} final results")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in fusion strategy {strategy_name}: {e}")
            # Fallback to simple concatenation
            return self._simple_fallback_fusion(vector_results, graph_results)
    
    def _rank_and_filter_results(
        self, 
        results: List[RetrievalResult], 
        config: HybridRetrievalConfig
    ) -> List[RetrievalResult]:
        """Rank and filter results based on configuration.
        
        Args:
            results: Results to rank and filter
            config: Configuration with thresholds and limits
            
        Returns:
            Ranked and filtered results
        """
        # Filter by minimum confidence
        filtered_results = [
            r for r in results 
            if r.score >= config.min_confidence_threshold
        ]
        
        # Sort by score (descending)
        sorted_results = sorted(
            filtered_results, 
            key=lambda r: r.score, 
            reverse=True
        )
        
        return sorted_results
    
    def _simple_fallback_fusion(
        self, 
        vector_results: List[RetrievalResult], 
        graph_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Simple fallback fusion by concatenation with deduplication.
        
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
        
        return sorted(all_results, key=lambda r: r.score, reverse=True)
