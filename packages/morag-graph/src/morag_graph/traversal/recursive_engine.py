"""Recursive graph traversal engine with intelligent path discovery."""

import asyncio
import time
from typing import List, Dict, Any, Optional, Set, Tuple, NamedTuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from ..models import Entity, Relation
from ..operations.traversal import GraphTraversal, GraphPath
from .path_selector import LLMPathSelector, QueryContext, TraversalStrategy, PathRelevanceScore

logger = structlog.get_logger(__name__)


@dataclass
class TraversalState:
    """Represents the current state of traversal."""
    current_entity: Entity
    visited_entities: Set[str]
    path_entities: List[Entity]
    path_relations: List[Relation]
    depth: int
    accumulated_score: float
    metadata: Dict[str, Any]


@dataclass
class TraversalResult:
    """Result of recursive graph traversal."""
    discovered_paths: List[PathRelevanceScore]
    total_entities_visited: int
    total_relations_traversed: int
    traversal_time: float
    strategy_used: TraversalStrategy
    query_context: QueryContext
    metadata: Dict[str, Any]


class CycleDetectionStrategy(Enum):
    """Strategies for detecting and handling cycles."""
    STRICT = "strict"  # Never revisit any entity
    RELAXED = "relaxed"  # Allow revisiting entities at different depths
    SMART = "smart"  # Use heuristics to determine when to allow revisits


class RecursiveTraversalEngine:
    """Intelligent recursive graph traversal engine."""
    
    def __init__(
        self,
        graph_traversal: GraphTraversal,
        path_selector: Optional[LLMPathSelector] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the recursive traversal engine.
        
        Args:
            graph_traversal: Graph traversal operations
            path_selector: Optional LLM path selector
            config: Optional configuration dictionary
        """
        self.graph_traversal = graph_traversal
        self.path_selector = path_selector or LLMPathSelector()
        self.config = config or {}
        self.settings = get_settings()
        
        # Traversal parameters
        self.max_depth = self.config.get('max_depth', 4)
        self.max_paths = self.config.get('max_paths', 100)
        self.max_entities_per_level = self.config.get('max_entities_per_level', 20)
        self.max_traversal_time = self.config.get('max_traversal_time', 30.0)  # seconds
        
        # Cycle detection
        self.cycle_detection = CycleDetectionStrategy(
            self.config.get('cycle_detection', 'smart')
        )
        self.allow_entity_revisit_depth = self.config.get('allow_entity_revisit_depth', 2)
        
        # Performance settings
        self.enable_parallel_traversal = self.config.get('enable_parallel_traversal', True)
        self.batch_size = self.config.get('batch_size', 10)
        self.enable_early_termination = self.config.get('enable_early_termination', True)
        
        # Scoring and filtering
        self.min_path_score = self.config.get('min_path_score', 0.2)
        self.score_decay_factor = self.config.get('score_decay_factor', 0.9)
        
        logger.info(
            "Recursive traversal engine initialized",
            max_depth=self.max_depth,
            max_paths=self.max_paths,
            cycle_detection=self.cycle_detection.value,
            enable_parallel_traversal=self.enable_parallel_traversal
        )
    
    async def traverse(
        self,
        starting_entities: List[Entity],
        query_context: QueryContext,
        strategy: TraversalStrategy = TraversalStrategy.ADAPTIVE,
        max_depth: Optional[int] = None
    ) -> TraversalResult:
        """Perform recursive graph traversal.
        
        Args:
            starting_entities: Starting entities for traversal
            query_context: Query context for guided traversal
            strategy: Traversal strategy to use
            max_depth: Optional override for maximum depth
            
        Returns:
            TraversalResult with discovered paths and metadata
        """
        if not starting_entities:
            return TraversalResult(
                discovered_paths=[],
                total_entities_visited=0,
                total_relations_traversed=0,
                traversal_time=0.0,
                strategy_used=strategy,
                query_context=query_context,
                metadata={'error': 'No starting entities provided'}
            )
        
        start_time = time.time()
        max_depth = max_depth or self.max_depth
        
        try:
            logger.info(
                "Starting recursive traversal",
                starting_entities=[e.name for e in starting_entities],
                query=query_context.query,
                strategy=strategy.value,
                max_depth=max_depth
            )
            
            # Initialize traversal state
            all_discovered_paths = []
            total_entities_visited = 0
            total_relations_traversed = 0
            
            # Choose traversal method based on strategy
            if strategy == TraversalStrategy.BREADTH_FIRST:
                paths, stats = await self._traverse_breadth_first(
                    starting_entities, query_context, max_depth
                )
            elif strategy == TraversalStrategy.DEPTH_FIRST:
                paths, stats = await self._traverse_depth_first(
                    starting_entities, query_context, max_depth
                )
            elif strategy == TraversalStrategy.RELEVANCE_GUIDED:
                paths, stats = await self._traverse_relevance_guided(
                    starting_entities, query_context, max_depth
                )
            else:  # ADAPTIVE or SHORTEST_PATH
                paths, stats = await self._traverse_adaptive(
                    starting_entities, query_context, max_depth
                )
            
            all_discovered_paths.extend(paths)
            total_entities_visited += stats['entities_visited']
            total_relations_traversed += stats['relations_traversed']
            
            # Score and filter paths
            if all_discovered_paths:
                scored_paths = await self.path_selector.select_paths(
                    query_context.query,
                    starting_entities,
                    [p.path for p in all_discovered_paths],
                    query_context,
                    strategy
                )
            else:
                scored_paths = []
            
            traversal_time = time.time() - start_time
            
            logger.info(
                "Recursive traversal completed",
                discovered_paths=len(all_discovered_paths),
                scored_paths=len(scored_paths),
                entities_visited=total_entities_visited,
                relations_traversed=total_relations_traversed,
                traversal_time=traversal_time
            )
            
            return TraversalResult(
                discovered_paths=scored_paths,
                total_entities_visited=total_entities_visited,
                total_relations_traversed=total_relations_traversed,
                traversal_time=traversal_time,
                strategy_used=strategy,
                query_context=query_context,
                metadata={
                    'max_depth_reached': max_depth,
                    'cycle_detection': self.cycle_detection.value,
                    'early_termination_used': self.enable_early_termination
                }
            )
            
        except Exception as e:
            logger.error(f"Recursive traversal failed: {e}")
            return TraversalResult(
                discovered_paths=[],
                total_entities_visited=0,
                total_relations_traversed=0,
                traversal_time=time.time() - start_time,
                strategy_used=strategy,
                query_context=query_context,
                metadata={'error': str(e)}
            )
    
    async def _traverse_breadth_first(
        self,
        starting_entities: List[Entity],
        query_context: QueryContext,
        max_depth: int
    ) -> Tuple[List[PathRelevanceScore], Dict[str, int]]:
        """Perform breadth-first traversal."""
        discovered_paths = []
        entities_visited = 0
        relations_traversed = 0
        
        # Initialize queue with starting states
        queue = deque()
        for entity in starting_entities:
            initial_state = TraversalState(
                current_entity=entity,
                visited_entities={entity.id},
                path_entities=[entity],
                path_relations=[],
                depth=0,
                accumulated_score=1.0,
                metadata={}
            )
            queue.append(initial_state)
        
        while queue and len(discovered_paths) < self.max_paths:
            # Check time limit
            if time.time() - time.time() > self.max_traversal_time:
                logger.warning("Traversal time limit exceeded")
                break
            
            current_state = queue.popleft()
            entities_visited += 1
            
            # Check depth limit
            if current_state.depth >= max_depth:
                # Create path from current state
                if len(current_state.path_entities) > 1:
                    path = GraphPath(
                        entities=current_state.path_entities,
                        relations=current_state.path_relations
                    )
                    path_score = PathRelevanceScore(
                        path=path,
                        relevance_score=current_state.accumulated_score,
                        confidence=0.8,
                        reasoning=f"Breadth-first path at depth {current_state.depth}",
                        metadata={'traversal_method': 'breadth_first'}
                    )
                    discovered_paths.append(path_score)
                continue
            
            # Get neighbors
            try:
                neighbors = await self.graph_traversal.find_neighbors(
                    current_state.current_entity.id,
                    max_distance=1
                )
                
                # Limit neighbors per level
                neighbors = neighbors[:self.max_entities_per_level]
                
                for neighbor in neighbors:
                    # Check cycle detection
                    if not self._should_visit_entity(neighbor, current_state):
                        continue
                    
                    # Get relation between current entity and neighbor
                    relations = await self._get_relations_between(
                        current_state.current_entity, neighbor
                    )
                    
                    for relation in relations:
                        relations_traversed += 1
                        
                        # Calculate new score
                        new_score = current_state.accumulated_score * self.score_decay_factor
                        
                        # Skip if score too low
                        if new_score < self.min_path_score:
                            continue
                        
                        # Create new state
                        new_visited = current_state.visited_entities.copy()
                        new_visited.add(neighbor.id)
                        
                        new_state = TraversalState(
                            current_entity=neighbor,
                            visited_entities=new_visited,
                            path_entities=current_state.path_entities + [neighbor],
                            path_relations=current_state.path_relations + [relation],
                            depth=current_state.depth + 1,
                            accumulated_score=new_score,
                            metadata={}
                        )
                        
                        queue.append(new_state)
                        
            except Exception as e:
                logger.warning(f"Failed to get neighbors for {current_state.current_entity.id}: {e}")
                continue
        
        return discovered_paths, {
            'entities_visited': entities_visited,
            'relations_traversed': relations_traversed
        }
    
    async def _traverse_depth_first(
        self,
        starting_entities: List[Entity],
        query_context: QueryContext,
        max_depth: int
    ) -> Tuple[List[PathRelevanceScore], Dict[str, int]]:
        """Perform depth-first traversal."""
        discovered_paths = []
        entities_visited = 0
        relations_traversed = 0
        
        async def dfs_recursive(
            current_entity: Entity,
            visited: Set[str],
            path_entities: List[Entity],
            path_relations: List[Relation],
            depth: int,
            accumulated_score: float
        ):
            nonlocal entities_visited, relations_traversed
            
            entities_visited += 1
            
            # Check limits
            if (depth >= max_depth or 
                len(discovered_paths) >= self.max_paths or
                accumulated_score < self.min_path_score):
                
                # Create path if valid
                if len(path_entities) > 1:
                    path = GraphPath(entities=path_entities, relations=path_relations)
                    path_score = PathRelevanceScore(
                        path=path,
                        relevance_score=accumulated_score,
                        confidence=0.8,
                        reasoning=f"Depth-first path at depth {depth}",
                        metadata={'traversal_method': 'depth_first'}
                    )
                    discovered_paths.append(path_score)
                return
            
            # Get neighbors
            try:
                neighbors = await self.graph_traversal.find_neighbors(
                    current_entity.id, max_distance=1
                )
                
                for neighbor in neighbors[:self.max_entities_per_level]:
                    if neighbor.id in visited:
                        continue
                    
                    # Get relations
                    relations = await self._get_relations_between(current_entity, neighbor)
                    
                    for relation in relations:
                        relations_traversed += 1
                        
                        new_score = accumulated_score * self.score_decay_factor
                        new_visited = visited.copy()
                        new_visited.add(neighbor.id)
                        
                        await dfs_recursive(
                            neighbor,
                            new_visited,
                            path_entities + [neighbor],
                            path_relations + [relation],
                            depth + 1,
                            new_score
                        )
                        
            except Exception as e:
                logger.warning(f"DFS failed for {current_entity.id}: {e}")
        
        # Start DFS from each starting entity
        for entity in starting_entities:
            await dfs_recursive(
                entity, {entity.id}, [entity], [], 0, 1.0
            )
        
        return discovered_paths, {
            'entities_visited': entities_visited,
            'relations_traversed': relations_traversed
        }
    
    async def _traverse_relevance_guided(
        self,
        starting_entities: List[Entity],
        query_context: QueryContext,
        max_depth: int
    ) -> Tuple[List[PathRelevanceScore], Dict[str, int]]:
        """Perform relevance-guided traversal using LLM scoring."""
        # For now, use breadth-first with enhanced scoring
        return await self._traverse_breadth_first(starting_entities, query_context, max_depth)
    
    async def _traverse_adaptive(
        self,
        starting_entities: List[Entity],
        query_context: QueryContext,
        max_depth: int
    ) -> Tuple[List[PathRelevanceScore], Dict[str, int]]:
        """Perform adaptive traversal based on query characteristics."""
        # Choose strategy based on query complexity
        if query_context.complexity_score > 0.7:
            return await self._traverse_breadth_first(starting_entities, query_context, max_depth)
        else:
            return await self._traverse_depth_first(starting_entities, query_context, max_depth)
    
    def _should_visit_entity(self, entity: Entity, current_state: TraversalState) -> bool:
        """Determine if an entity should be visited based on cycle detection strategy."""
        if self.cycle_detection == CycleDetectionStrategy.STRICT:
            return entity.id not in current_state.visited_entities
        
        elif self.cycle_detection == CycleDetectionStrategy.RELAXED:
            # Allow revisiting if we're at a different depth
            return (entity.id not in current_state.visited_entities or 
                    current_state.depth >= self.allow_entity_revisit_depth)
        
        else:  # SMART
            # Use heuristics: allow revisiting important entities
            if entity.id not in current_state.visited_entities:
                return True
            
            # Allow revisiting if entity appears in query
            query_lower = current_state.metadata.get('query', '').lower()
            return entity.name.lower() in query_lower
    
    async def _get_relations_between(
        self,
        entity1: Entity,
        entity2: Entity
    ) -> List[Relation]:
        """Get relations between two entities."""
        try:
            # This is a simplified implementation
            # In practice, you'd query the graph storage for actual relations
            return [Relation(
                id=f"rel_{entity1.id}_{entity2.id}",
                type="RELATED_TO",
                source_entity_id=entity1.id,
                target_entity_id=entity2.id,
                confidence=0.8
            )]
        except Exception as e:
            logger.warning(f"Failed to get relations between {entity1.id} and {entity2.id}: {e}")
            return []
