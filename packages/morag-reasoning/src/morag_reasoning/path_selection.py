"""Path selection agent for multi-hop reasoning."""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from morag_graph.operations import GraphPath
from morag_graph.models import Entity, Relation
from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class PathRelevanceScore:
    """Represents a path with its relevance score and reasoning."""
    path: GraphPath
    relevance_score: float
    confidence: float
    reasoning: str


@dataclass
class ReasoningStrategy:
    """Configuration for a reasoning strategy."""
    name: str
    description: str
    max_depth: int
    bidirectional: bool
    use_weights: bool


class PathSelectionAgent:
    """Agent for LLM-guided path selection in multi-hop reasoning."""
    
    def __init__(self, llm_client: LLMClient, max_paths: int = 10):
        """Initialize the path selection agent.
        
        Args:
            llm_client: LLM client for path selection
            max_paths: Maximum number of paths to select
        """
        self.llm_client = llm_client
        self.max_paths = max_paths
        self.logger = logging.getLogger(__name__)
        
        # Define reasoning strategies
        self.strategies = {
            "forward_chaining": ReasoningStrategy(
                name="forward_chaining",
                description="Start from query entities and explore forward",
                max_depth=4,
                bidirectional=False,
                use_weights=True
            ),
            "backward_chaining": ReasoningStrategy(
                name="backward_chaining", 
                description="Start from potential answers and work backward",
                max_depth=3,
                bidirectional=False,
                use_weights=True
            ),
            "bidirectional": ReasoningStrategy(
                name="bidirectional",
                description="Search from both ends and meet in the middle",
                max_depth=5,
                bidirectional=True,
                use_weights=True
            )
        }
    
    async def select_paths(
        self, 
        query: str, 
        available_paths: List[GraphPath],
        strategy: str = "forward_chaining"
    ) -> List[PathRelevanceScore]:
        """Use LLM to select most relevant paths for answering the query.
        
        Args:
            query: Search query
            available_paths: List of available graph paths
            strategy: Reasoning strategy to use
            
        Returns:
            List of selected paths with relevance scores
        """
        if not available_paths:
            return []
        
        try:
            # Create path selection prompt
            prompt = self._create_path_selection_prompt(query, available_paths)
            
            # Get LLM response
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent reasoning
            )
            
            # Parse LLM response
            selected_paths = self._parse_path_selection(response, available_paths)
            
            # Apply additional scoring
            scored_paths = await self._score_paths(query, selected_paths, strategy)
            
            # Sort by relevance and return top paths
            scored_paths.sort(key=lambda x: x.relevance_score, reverse=True)
            return scored_paths[:self.max_paths]
        
        except Exception as e:
            self.logger.error(f"Error in path selection: {str(e)}")
            # Fallback to simple scoring
            return self._fallback_path_selection(available_paths)
    
    def _create_path_selection_prompt(self, query: str, paths: List[GraphPath]) -> str:
        """Create a prompt for LLM to select relevant paths."""
        prompt = f"""Given the following query and available reasoning paths, select the most relevant paths that could help answer the question.

Query: {query}

Available Paths:
"""
        
        for i, path in enumerate(paths[:20]):  # Limit to avoid token overflow
            path_description = self._describe_path(path)
            prompt += f"{i+1}. {path_description}\n"
        
        prompt += """
For each path, provide:
1. Relevance score (0-10)
2. Confidence in the score (0-10)
3. Brief reasoning for the score

Format your response as JSON:
{
  "selected_paths": [
    {
      "path_id": 1,
      "relevance_score": 8.5,
      "confidence": 9.0,
      "reasoning": "This path directly connects the query entities through a relevant relationship."
    }
  ]
}"""
        
        return prompt
    
    def _describe_path(self, path: GraphPath) -> str:
        """Create a human-readable description of a graph path."""
        if len(path.entities) < 2:
            entity_name = path.entities[0].name if path.entities else 'Unknown'
            return f"Single entity: {entity_name}"
        
        description_parts = []
        for i in range(len(path.entities) - 1):
            entity1 = path.entities[i].name
            entity2 = path.entities[i + 1].name
            relation = path.relations[i].type if i < len(path.relations) else "connected_to"
            description_parts.append(f"{entity1} --[{relation}]--> {entity2}")
        
        return " -> ".join(description_parts)
    
    def _parse_path_selection(self, response: str, available_paths: List[GraphPath]) -> List[PathRelevanceScore]:
        """Parse LLM response to extract selected paths."""
        try:
            data = json.loads(response)
            selected_paths = []
            
            for selection in data.get("selected_paths", []):
                path_id = selection.get("path_id", 0) - 1  # Convert to 0-based index
                if 0 <= path_id < len(available_paths):
                    path_score = PathRelevanceScore(
                        path=available_paths[path_id],
                        relevance_score=float(selection.get("relevance_score", 0)),
                        confidence=float(selection.get("confidence", 0)),
                        reasoning=selection.get("reasoning", "")
                    )
                    selected_paths.append(path_score)
            
            return selected_paths
        
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            return self._fallback_path_selection(available_paths)
    
    async def _score_paths(
        self, 
        query: str, 
        paths: List[PathRelevanceScore], 
        strategy: str
    ) -> List[PathRelevanceScore]:
        """Apply additional scoring based on reasoning strategy."""
        strategy_config = self.strategies.get(strategy, self.strategies["forward_chaining"])
        
        for path_score in paths:
            # Apply strategy-specific adjustments
            if strategy_config.use_weights:
                # Boost score based on path weights (if available)
                if hasattr(path_score.path, 'total_weight'):
                    weight_bonus = min(path_score.path.total_weight / 10.0, 2.0)
                    path_score.relevance_score += weight_bonus
            
            # Penalize very long paths unless strategy allows it
            path_length = len(path_score.path.entities)
            if path_length > strategy_config.max_depth:
                penalty = (path_length - strategy_config.max_depth) * 0.5
                path_score.relevance_score = max(0, path_score.relevance_score - penalty)
            
            # Boost confidence for shorter, more direct paths
            if path_length <= 3:
                path_score.confidence += 1.0
        
        return paths
    
    def _fallback_path_selection(self, available_paths: List[GraphPath]) -> List[PathRelevanceScore]:
        """Fallback path selection when LLM fails."""
        fallback_paths = []
        for path in available_paths[:self.max_paths]:
            # Simple scoring based on path length and weight
            length_score = max(0, 10 - len(path.entities))
            weight_score = getattr(path, 'total_weight', 5.0)  # Default weight if not available
            weight_score = min(weight_score, 10)
            combined_score = (length_score + weight_score) / 2
            
            path_score = PathRelevanceScore(
                path=path,
                relevance_score=combined_score,
                confidence=5.0,  # Medium confidence for fallback
                reasoning="Fallback scoring based on path length and weight"
            )
            fallback_paths.append(path_score)
        
        return fallback_paths


class ReasoningPathFinder:
    """Finds and selects reasoning paths for multi-hop queries."""

    def __init__(self, graph_engine, path_selector: PathSelectionAgent):
        """Initialize the reasoning path finder.

        Args:
            graph_engine: Graph engine for path discovery
            path_selector: Path selection agent
        """
        self.graph_engine = graph_engine
        self.path_selector = path_selector
        self.logger = logging.getLogger(__name__)

    async def find_reasoning_paths(
        self,
        query: str,
        start_entities: List[str],
        target_entities: Optional[List[str]] = None,
        strategy: str = "forward_chaining",
        max_paths: int = 50
    ) -> List[PathRelevanceScore]:
        """Find and select reasoning paths for multi-hop queries.

        Args:
            query: Query to answer
            start_entities: Starting entities for path discovery
            target_entities: Optional target entities
            strategy: Reasoning strategy to use
            max_paths: Maximum number of paths to discover

        Returns:
            List of selected paths with relevance scores
        """
        try:
            # Get all possible paths
            all_paths = await self._discover_paths(
                start_entities, target_entities, strategy, max_paths
            )

            if not all_paths:
                self.logger.warning(f"No paths found for query: {query}")
                return []

            # Use LLM to select most relevant paths
            selected_paths = await self.path_selector.select_paths(
                query, all_paths, strategy
            )

            self.logger.info(f"Selected {len(selected_paths)} paths from {len(all_paths)} candidates")
            return selected_paths

        except Exception as e:
            self.logger.error(f"Error finding reasoning paths: {str(e)}")
            return []

    async def _discover_paths(
        self,
        start_entities: List[str],
        target_entities: Optional[List[str]],
        strategy: str,
        max_paths: int
    ) -> List[GraphPath]:
        """Discover all possible paths using the specified strategy."""
        strategy_config = self.path_selector.strategies[strategy]
        all_paths = []

        if strategy == "bidirectional" and target_entities:
            # Bidirectional search
            for start_entity in start_entities:
                for target_entity in target_entities:
                    if hasattr(self.graph_engine, 'find_bidirectional_paths'):
                        paths = await self.graph_engine.find_bidirectional_paths(
                            start_entity, target_entity, strategy_config.max_depth
                        )
                        all_paths.extend(paths)
                    else:
                        # Fallback to regular path finding
                        path = await self.graph_engine.find_shortest_path(start_entity, target_entity)
                        if path:
                            all_paths.append(path)

        elif strategy == "backward_chaining" and target_entities:
            # Backward chaining from target entities
            for target_entity in target_entities:
                if hasattr(self.graph_engine, 'traverse_backward'):
                    paths = await self.graph_engine.traverse_backward(
                        target_entity, strategy_config.max_depth
                    )
                    # Filter paths that connect to start entities
                    relevant_paths = [
                        path for path in paths
                        if any(entity.name in start_entities for entity in path.entities)
                    ]
                    all_paths.extend(relevant_paths)
                else:
                    # Fallback to forward search from start entities
                    for start_entity in start_entities:
                        path = await self.graph_engine.find_shortest_path(start_entity, target_entity)
                        if path:
                            all_paths.append(path)

        else:
            # Forward chaining (default)
            for start_entity in start_entities:
                if hasattr(self.graph_engine, 'traverse'):
                    traversal_result = await self.graph_engine.traverse(
                        start_entity,
                        algorithm="bfs",
                        max_depth=strategy_config.max_depth
                    )
                    all_paths.extend(traversal_result.get('paths', []))
                else:
                    # Fallback to finding neighbors
                    neighbors = await self.graph_engine.find_neighbors(
                        start_entity, max_distance=strategy_config.max_depth
                    )
                    # Create simple paths to neighbors
                    for neighbor in neighbors:
                        path = await self.graph_engine.find_shortest_path(start_entity, neighbor.id)
                        if path:
                            all_paths.append(path)

        # Remove duplicates and limit results
        unique_paths = self._deduplicate_paths(all_paths)
        return unique_paths[:max_paths]

    def _deduplicate_paths(self, paths: List[GraphPath]) -> List[GraphPath]:
        """Remove duplicate paths based on entity sequences."""
        seen_sequences = set()
        unique_paths = []

        for path in paths:
            sequence = tuple(entity.name for entity in path.entities)
            if sequence not in seen_sequences:
                seen_sequences.add(sequence)
                unique_paths.append(path)

        return unique_paths
