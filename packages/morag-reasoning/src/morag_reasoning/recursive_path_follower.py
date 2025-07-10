"""Recursive path following service for intelligent retrieval."""

import structlog
from typing import List, Optional, Dict, Any, Set, Tuple
from pydantic_ai import Agent
from pydantic import BaseModel, Field

from morag_reasoning.llm import LLMClient
from morag_reasoning.intelligent_retrieval_models import (
    EntityPath, PathDecision, RetrievalIteration, PathFollowingRequest
)
from morag_graph.storage.neo4j_storage import Neo4jStorage


class PathFollowingDecision(BaseModel):
    """LLM decision for path following."""
    continue_exploration: bool = Field(..., description="Whether to continue exploring")
    paths_to_follow: List[int] = Field(..., description="Indices of paths to follow")
    reasoning: str = Field(..., description="Reasoning for the decision")
    stop_reason: Optional[str] = Field(None, description="Reason to stop if continue_exploration is False")


class RecursivePathFollower:
    """Service for LLM-driven recursive graph path following."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        graph_storage: Neo4jStorage,
        max_iterations: int = 8,
        max_paths_per_entity: int = 10,
        max_depth: int = 3,
        min_relevance_threshold: float = 0.2
    ):
        """Initialize the recursive path follower.
        
        Args:
            llm_client: LLM client for decision making
            graph_storage: Neo4j storage for graph traversal
            max_iterations: Maximum recursive iterations
            max_paths_per_entity: Maximum paths to consider per entity
            max_depth: Maximum path depth
            min_relevance_threshold: Minimum relevance threshold
        """
        self.llm_client = llm_client
        self.graph_storage = graph_storage
        self.max_iterations = max_iterations
        self.max_paths_per_entity = max_paths_per_entity
        self.max_depth = max_depth
        self.min_relevance_threshold = min_relevance_threshold
        self.logger = structlog.get_logger(__name__)
        
        # Create PydanticAI agent for path following decisions
        self.agent = Agent(
            model=llm_client.get_model(),
            result_type=PathFollowingDecision,
            system_prompt=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for path following decisions."""
        return """You are an expert graph traversal decision system. Your task is to decide which paths to follow in a knowledge graph to gather comprehensive information for answering a user's query.

Guidelines:
1. Analyze the user's query to understand what information is needed
2. Evaluate each available path for its potential to provide relevant information
3. Consider the current exploration state and avoid redundant paths
4. Balance breadth and depth of exploration
5. Stop when sufficient information has been gathered or when paths become irrelevant
6. Prioritize paths that lead to entities directly related to the query

Decision Criteria:
- FOLLOW: Path likely contains relevant information for the query
- SKIP: Path is redundant or unlikely to provide new relevant information
- STOP: Sufficient information gathered or no more relevant paths available

Consider:
- Semantic relevance to the query
- Potential for new information discovery
- Relationship strength and type
- Current exploration coverage
- Iteration depth and efficiency

Be strategic and focused - aim for comprehensive but efficient exploration."""
    
    async def follow_paths_recursively(
        self,
        query: str,
        initial_entities: List[str],
        max_iterations: Optional[int] = None,
        max_entities_per_iteration: int = 10
    ) -> List[RetrievalIteration]:
        """Follow paths recursively through the graph.
        
        Args:
            query: Original user query
            initial_entities: Starting entities
            max_iterations: Maximum iterations (overrides instance default)
            max_entities_per_iteration: Max entities to explore per iteration
            
        Returns:
            List of retrieval iterations with path information
        """
        max_iter = max_iterations or self.max_iterations
        iterations = []
        explored_entities: Set[str] = set()
        current_entities = initial_entities.copy()
        
        self.logger.info(
            "Starting recursive path following",
            query=query,
            initial_entities=initial_entities,
            max_iterations=max_iter
        )
        
        for iteration in range(max_iter):
            self.logger.info(f"Starting iteration {iteration + 1}/{max_iter}")
            
            # Limit entities for this iteration
            entities_to_explore = current_entities[:max_entities_per_iteration]
            
            # Find paths from current entities
            paths_found = await self._discover_paths_from_entities(
                entities_to_explore, explored_entities
            )
            
            if not paths_found:
                self.logger.info("No new paths found, stopping exploration")
                break
            
            # Use LLM to decide which paths to follow
            decision = await self._make_path_decision(
                query, list(explored_entities), paths_found, iteration + 1, max_iter
            )
            
            # Process LLM decision
            paths_to_follow = []
            if decision.continue_exploration:
                for path_idx in decision.paths_to_follow:
                    if 0 <= path_idx < len(paths_found):
                        path = paths_found[path_idx]
                        path.llm_decision = PathDecision.FOLLOW
                        path.decision_reasoning = decision.reasoning
                        paths_to_follow.append(path)
                
                # Mark other paths as skipped
                for i, path in enumerate(paths_found):
                    if i not in decision.paths_to_follow:
                        path.llm_decision = PathDecision.SKIP
                        path.decision_reasoning = "Not selected by LLM for following"
            else:
                # Mark all paths as stopped
                for path in paths_found:
                    path.llm_decision = PathDecision.STOP
                    path.decision_reasoning = decision.stop_reason or "LLM decided to stop exploration"
            
            # Create iteration record
            iteration_record = RetrievalIteration(
                iteration=iteration + 1,
                entities_explored=entities_to_explore,
                paths_found=paths_found,
                paths_followed=paths_to_follow,
                chunks_retrieved=0,  # Will be filled by caller
                llm_stop_reason=decision.stop_reason if not decision.continue_exploration else None
            )
            iterations.append(iteration_record)
            
            # Update state for next iteration
            explored_entities.update(entities_to_explore)
            
            if not decision.continue_exploration:
                self.logger.info(f"LLM decided to stop: {decision.stop_reason}")
                break
            
            # Prepare entities for next iteration
            next_entities = []
            for path in paths_to_follow:
                # Add entities from followed paths that haven't been explored
                for entity in path.path_entities:
                    if entity not in explored_entities and entity not in next_entities:
                        next_entities.append(entity)
            
            current_entities = next_entities
            
            if not current_entities:
                self.logger.info("No new entities to explore, stopping")
                break
        
        self.logger.info(
            "Recursive path following completed",
            total_iterations=len(iterations),
            total_entities_explored=len(explored_entities)
        )
        
        return iterations
    
    async def _discover_paths_from_entities(
        self,
        entities: List[str],
        explored_entities: Set[str]
    ) -> List[EntityPath]:
        """Discover paths from a list of entities.
        
        Args:
            entities: Entities to explore from
            explored_entities: Already explored entities to avoid
            
        Returns:
            List of discovered paths
        """
        all_paths = []
        
        for entity_name in entities:
            try:
                # Find entity by name first
                entity_candidates = await self.graph_storage.search_entities(
                    entity_name, limit=1
                )
                
                if not entity_candidates:
                    continue
                
                entity = entity_candidates[0]
                
                # Get neighbors and relationships
                neighbors = await self.graph_storage.get_neighbors(
                    entity.id, max_depth=self.max_depth
                )
                
                # Convert neighbors to paths
                for neighbor in neighbors[:self.max_paths_per_entity]:
                    # Check if neighbor name is already explored (use names for consistency)
                    if neighbor.name in explored_entities:
                        continue

                    # Create a simple path (could be enhanced with actual path finding)
                    path = EntityPath(
                        entity_id=entity.id,
                        entity_name=entity.name,
                        path_entities=[entity.name, neighbor.name],  # Use names for consistency
                        path_relations=["RELATED_TO"],  # Simplified - could get actual relation types
                        depth=1,
                        relevance_score=0.7,  # Higher default relevance to encourage exploration
                        llm_decision=PathDecision.FOLLOW,  # Will be updated by LLM
                        decision_reasoning=""  # Will be updated by LLM
                    )
                    all_paths.append(path)
                    
            except Exception as e:
                self.logger.warning(
                    "Failed to discover paths from entity",
                    entity=entity_name,
                    error=str(e)
                )
        
        return all_paths
    
    async def _make_path_decision(
        self,
        query: str,
        explored_entities: List[str],
        available_paths: List[EntityPath],
        iteration: int,
        max_iterations: int
    ) -> PathFollowingDecision:
        """Use LLM to make path following decision.
        
        Args:
            query: Original user query
            explored_entities: Already explored entities
            available_paths: Available paths to consider
            iteration: Current iteration number
            max_iterations: Maximum iterations allowed
            
        Returns:
            LLM decision for path following
        """
        try:
            # Create decision prompt
            prompt = self._create_decision_prompt(
                query, explored_entities, available_paths, iteration, max_iterations
            )
            
            # Get LLM decision
            result = await self.agent.run(prompt)
            return result.data
            
        except Exception as e:
            self.logger.error(
                "Failed to make path decision",
                error=str(e),
                iteration=iteration
            )
            # Fallback decision
            return PathFollowingDecision(
                continue_exploration=False,
                paths_to_follow=[],
                reasoning="Error in LLM decision making",
                stop_reason="LLM decision error"
            )
    
    def _create_decision_prompt(
        self,
        query: str,
        explored_entities: List[str],
        available_paths: List[EntityPath],
        iteration: int,
        max_iterations: int
    ) -> str:
        """Create prompt for path following decision.
        
        Args:
            query: Original user query
            explored_entities: Already explored entities
            available_paths: Available paths to consider
            iteration: Current iteration number
            max_iterations: Maximum iterations allowed
            
        Returns:
            Decision prompt for LLM
        """
        paths_info = []
        for i, path in enumerate(available_paths):
            paths_info.append(
                f"Path {i}: {path.entity_name} -> {' -> '.join(path.path_entities[1:])} "
                f"(depth: {path.depth}, relevance: {path.relevance_score:.2f})"
            )
        
        return f"""Analyze the following graph exploration state and decide which paths to follow:

QUERY: "{query}"

CURRENT STATE:
- Iteration: {iteration}/{max_iterations}
- Already explored entities: {', '.join(explored_entities) if explored_entities else 'None'}

AVAILABLE PATHS:
{chr(10).join(paths_info) if paths_info else 'No paths available'}

DECISION REQUIRED:
1. Should exploration continue? Generally continue unless you have very comprehensive coverage or are at max iterations.
2. Which specific paths (by index) should be followed? Be generous - select multiple paths that could provide relevant information.
3. Provide clear reasoning for your decision.

Consider:
- Relevance to answering the query comprehensively
- Potential for discovering new information and perspectives
- Value of exploring related concepts and entities
- Aim for thorough coverage rather than minimal exploration
- Early iterations should be more exploratory, later iterations more selective

BIAS TOWARD EXPLORATION: Unless you have very strong reasons to stop, continue exploring to gather comprehensive information."""
