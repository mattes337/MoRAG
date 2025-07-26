"""LLM-guided path selection for intelligent graph traversal."""

import asyncio
import time
from typing import List, Dict, Any, Optional, NamedTuple, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from ..models import Entity, Relation
from ..operations.traversal import GraphTraversal, GraphPath

logger = structlog.get_logger(__name__)

# Optional LLM imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class PathRelevanceScore(NamedTuple):
    """Represents a path with its relevance score."""
    path: GraphPath
    relevance_score: float
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = {}


class TraversalStrategy(Enum):
    """Strategies for graph traversal."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    RELEVANCE_GUIDED = "relevance_guided"
    SHORTEST_PATH = "shortest_path"
    ADAPTIVE = "adaptive"


@dataclass
class QueryContext:
    """Context information for query-guided traversal."""
    query: str
    intent: str
    entities: List[str]
    keywords: List[str]
    complexity_score: float
    domain: Optional[str] = None
    metadata: Dict[str, Any] = None


class LLMPathSelector:
    """LLM-guided path selection for intelligent graph traversal."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LLM path selector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = get_settings()
        
        # LLM configuration
        self.llm_enabled = self.config.get('llm_enabled', True) and GEMINI_AVAILABLE
        self.model_name = self.config.get('model_name', 'gemini-1.5-flash')
        self.temperature = self.config.get('temperature', 0.1)
        self.max_tokens = self.config.get('max_tokens', 1000)
        
        # Path selection parameters
        self.max_paths_to_evaluate = self.config.get('max_paths_to_evaluate', 20)
        self.min_relevance_threshold = self.config.get('min_relevance_threshold', 0.3)
        self.max_path_length = self.config.get('max_path_length', 5)
        
        # Performance settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_size = self.config.get('cache_size', 1000)
        self.batch_size = self.config.get('batch_size', 5)
        
        # Initialize components
        self._llm_client = None
        self._cache = {} if self.enable_caching else None
        
        logger.info(
            "LLM path selector initialized",
            llm_enabled=self.llm_enabled,
            model_name=self.model_name,
            max_paths_to_evaluate=self.max_paths_to_evaluate,
            min_relevance_threshold=self.min_relevance_threshold
        )
    
    async def initialize(self) -> None:
        """Initialize the LLM client."""
        if not self.llm_enabled:
            logger.info("LLM path selection disabled")
            return
        
        if self._llm_client:
            return
        
        try:
            # Configure Gemini
            genai.configure(api_key=self.settings.gemini_api_key)
            self._llm_client = genai.GenerativeModel(self.model_name)
            
            logger.info("LLM client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_enabled = False
    
    async def select_paths(
        self,
        query: str,
        starting_entities: List[Entity],
        available_paths: List[GraphPath],
        query_context: Optional[QueryContext] = None,
        strategy: TraversalStrategy = TraversalStrategy.RELEVANCE_GUIDED
    ) -> List[PathRelevanceScore]:
        """Select and score paths based on relevance to the query.
        
        Args:
            query: User query
            starting_entities: Starting entities for traversal
            available_paths: Available paths to evaluate
            query_context: Optional query context
            strategy: Traversal strategy to use
            
        Returns:
            List of paths with relevance scores
        """
        if not available_paths:
            return []
        
        try:
            logger.debug(
                "Starting path selection",
                query=query,
                num_paths=len(available_paths),
                strategy=strategy.value
            )
            
            # Initialize LLM if needed
            await self.initialize()
            
            # Filter paths by basic criteria
            filtered_paths = self._filter_paths_basic(available_paths)
            
            # Limit number of paths to evaluate
            paths_to_evaluate = filtered_paths[:self.max_paths_to_evaluate]
            
            # Score paths
            if self.llm_enabled:
                scored_paths = await self._score_paths_with_llm(
                    query, paths_to_evaluate, query_context
                )
            else:
                scored_paths = await self._score_paths_heuristic(
                    query, paths_to_evaluate, query_context
                )
            
            # Filter by relevance threshold
            relevant_paths = [
                p for p in scored_paths 
                if p.relevance_score >= self.min_relevance_threshold
            ]
            
            # Sort by relevance score
            relevant_paths.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(
                "Path selection completed",
                total_paths=len(available_paths),
                evaluated_paths=len(paths_to_evaluate),
                relevant_paths=len(relevant_paths),
                strategy=strategy.value
            )
            
            return relevant_paths
            
        except Exception as e:
            logger.error(f"Path selection failed: {e}")
            # Fallback to heuristic scoring
            return await self._score_paths_heuristic(query, available_paths[:10], query_context)
    
    def _filter_paths_basic(self, paths: List[GraphPath]) -> List[GraphPath]:
        """Apply basic filtering to paths."""
        filtered = []
        
        for path in paths:
            # Skip empty paths
            if not path.entities or len(path.entities) < 2:
                continue
            
            # Skip overly long paths
            if len(path.entities) > self.max_path_length:
                continue
            
            # Skip circular paths (same start and end entity)
            if path.entities[0].id == path.entities[-1].id:
                continue
            
            filtered.append(path)
        
        return filtered
    
    async def _score_paths_with_llm(
        self,
        query: str,
        paths: List[GraphPath],
        query_context: Optional[QueryContext]
    ) -> List[PathRelevanceScore]:
        """Score paths using LLM evaluation."""
        scored_paths = []
        
        # Process paths in batches
        for i in range(0, len(paths), self.batch_size):
            batch = paths[i:i + self.batch_size]
            batch_scores = await self._evaluate_path_batch(query, batch, query_context)
            scored_paths.extend(batch_scores)
        
        return scored_paths
    
    async def _evaluate_path_batch(
        self,
        query: str,
        paths: List[GraphPath],
        query_context: Optional[QueryContext]
    ) -> List[PathRelevanceScore]:
        """Evaluate a batch of paths with LLM."""
        try:
            # Create evaluation prompt
            prompt = self._create_evaluation_prompt(query, paths, query_context)
            
            # Get LLM response
            response = await self._llm_client.generate_content_async(prompt)
            
            # Parse response
            return self._parse_llm_response(response.text, paths)
            
        except Exception as e:
            logger.warning(f"LLM evaluation failed for batch: {e}")
            # Fallback to heuristic scoring
            return await self._score_paths_heuristic(query, paths, query_context)
    
    def _create_evaluation_prompt(
        self,
        query: str,
        paths: List[GraphPath],
        query_context: Optional[QueryContext]
    ) -> str:
        """Create evaluation prompt for LLM."""
        prompt = f"""
You are an expert at evaluating graph paths for relevance to user queries.

Query: "{query}"

Please evaluate the following paths and score their relevance to the query on a scale of 0.0 to 1.0:

"""
        
        for i, path in enumerate(paths):
            path_description = self._describe_path(path)
            prompt += f"\nPath {i+1}: {path_description}\n"
        
        prompt += """
For each path, provide:
1. Relevance score (0.0-1.0)
2. Confidence (0.0-1.0)
3. Brief reasoning

Format your response as JSON:
{
  "evaluations": [
    {
      "path_id": 1,
      "relevance_score": 0.8,
      "confidence": 0.9,
      "reasoning": "Direct connection between query entities"
    },
    ...
  ]
}
"""
        
        return prompt
    
    def _describe_path(self, path: GraphPath) -> str:
        """Create a human-readable description of a path."""
        if not path.entities or not path.relations:
            return "Empty path"
        
        description_parts = []
        
        for i, entity in enumerate(path.entities):
            description_parts.append(f"{entity.name} ({entity.type})")
            
            if i < len(path.relations):
                relation = path.relations[i]
                description_parts.append(f" --[{relation.type}]--> ")
        
        return "".join(description_parts)
    
    def _parse_llm_response(
        self,
        response_text: str,
        paths: List[GraphPath]
    ) -> List[PathRelevanceScore]:
        """Parse LLM response into path scores."""
        try:
            import json
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[start_idx:end_idx]
            data = json.loads(json_text)
            
            scored_paths = []
            evaluations = data.get('evaluations', [])
            
            for eval_data in evaluations:
                path_id = eval_data.get('path_id', 1) - 1  # Convert to 0-based index
                
                if 0 <= path_id < len(paths):
                    path = paths[path_id]
                    score = PathRelevanceScore(
                        path=path,
                        relevance_score=float(eval_data.get('relevance_score', 0.0)),
                        confidence=float(eval_data.get('confidence', 0.0)),
                        reasoning=eval_data.get('reasoning', ''),
                        metadata={'evaluation_method': 'llm'}
                    )
                    scored_paths.append(score)
            
            return scored_paths
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback to default scores
            return [
                PathRelevanceScore(
                    path=path,
                    relevance_score=0.5,
                    confidence=0.3,
                    reasoning="LLM parsing failed, using default score",
                    metadata={'evaluation_method': 'fallback'}
                )
                for path in paths
            ]
    
    async def _score_paths_heuristic(
        self,
        query: str,
        paths: List[GraphPath],
        query_context: Optional[QueryContext]
    ) -> List[PathRelevanceScore]:
        """Score paths using heuristic methods."""
        scored_paths = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for path in paths:
            # Calculate relevance based on entity and relation names
            relevance_score = 0.0
            match_count = 0
            total_elements = len(path.entities) + len(path.relations)

            # Score entities
            for entity in path.entities:
                entity_words = set(entity.name.lower().split())

                # Exact name match
                if entity.name.lower() in query_lower:
                    relevance_score += 1.0
                    match_count += 1
                # Word overlap
                elif entity_words & query_words:
                    overlap_ratio = len(entity_words & query_words) / len(entity_words)
                    relevance_score += overlap_ratio * 0.8
                    match_count += 1
                # Partial match for entity type
                elif entity.type and any(word in entity.type.lower() for word in query_words):
                    relevance_score += 0.3
                    match_count += 1

            # Score relations
            for relation in path.relations:
                relation_words = set(relation.type.lower().split('_'))

                if relation_words & query_words:
                    overlap_ratio = len(relation_words & query_words) / len(relation_words)
                    relevance_score += overlap_ratio * 0.6
                    match_count += 1
                # Common relation types get small boost
                elif relation.type.lower() in ['developed', 'created', 'worked_at', 'related_to']:
                    relevance_score += 0.2

            # Normalize score - ensure we always get some score for valid paths
            if total_elements > 0:
                final_score = relevance_score / total_elements
                # Give minimum score if there are any matches
                if match_count > 0:
                    final_score = max(final_score, 0.3)
                else:
                    # Even with no matches, give small score for valid paths
                    final_score = max(final_score, 0.1)
            else:
                final_score = 0.0

            # Apply path length penalty (prefer shorter paths)
            length_penalty = 1.0 / (1.0 + len(path.entities) * 0.05)
            final_score *= length_penalty

            scored_path = PathRelevanceScore(
                path=path,
                relevance_score=min(final_score, 1.0),
                confidence=0.7,  # Moderate confidence for heuristic scoring
                reasoning=f"Heuristic scoring: {match_count} matches out of {total_elements} elements",
                metadata={'evaluation_method': 'heuristic', 'match_count': match_count}
            )
            scored_paths.append(scored_path)

        return scored_paths
