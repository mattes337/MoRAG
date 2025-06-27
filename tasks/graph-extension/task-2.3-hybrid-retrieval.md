# Task 2.3: Hybrid Retrieval System

**Phase**: 2 - Core Graph Features  
**Priority**: High  
**Estimated Time**: 10-12 days total  
**Dependencies**: Task 2.2 (Graph Traversal), Task 1.3 (LLM-Based Entity and Relation Extraction)

## Overview

This task implements the core hybrid retrieval system that combines traditional vector-based retrieval with graph-guided context expansion. The system extracts entities from user queries, uses graph traversal to find related context, and fuses results from multiple retrieval methods to provide comprehensive and contextually rich responses.

## Subtasks

### 3.1.1: Query Entity Recognition
**Estimated Time**: 3-4 days  
**Priority**: High

#### Implementation Steps

1. **Query Entity Extractor**
   ```python
   # src/morag_retrieval/query/entity_extractor.py
   from typing import List, Dict, Optional, Tuple
   from dataclasses import dataclass
   from morag_nlp.extractors import BaseEntityExtractor
   from morag_graph.storage.base import BaseGraphStorage
   from morag_graph.models import Entity
   
   @dataclass
   class QueryEntity:
       text: str
       entity_type: str
       confidence: float
       start_pos: int
       end_pos: int
       linked_entity_id: Optional[str] = None
       linked_entity: Optional[Entity] = None
   
   @dataclass
   class QueryAnalysis:
       original_query: str
       entities: List[QueryEntity]
       intent: str
       query_type: str  # "factual", "exploratory", "comparative", etc.
       complexity_score: float
   
   class QueryEntityExtractor:
       def __init__(
           self, 
           entity_extractor: BaseEntityExtractor,
           graph_storage: BaseGraphStorage,
           similarity_threshold: float = 0.8
       ):
           self.entity_extractor = entity_extractor
           self.graph_storage = graph_storage
           self.similarity_threshold = similarity_threshold
           self.logger = logging.getLogger(__name__)
       
       async def extract_and_link_entities(self, query: str) -> QueryAnalysis:
           """Extract entities from query and link to knowledge graph."""
           try:
               # Extract entities using NLP pipeline
               extraction_result = await self.entity_extractor.extract_entities(query)
               
               # Convert to QueryEntity objects and link to graph
               query_entities = []
               for entity in extraction_result.entities:
                   query_entity = QueryEntity(
                       text=entity.text,
                       entity_type=entity.entity_type,
                       confidence=entity.confidence,
                       start_pos=entity.start_pos,
                       end_pos=entity.end_pos
                   )
                   
                   # Attempt to link to existing entities in graph
                   linked_entity = await self._link_to_graph_entity(query_entity)
                   if linked_entity:
                       query_entity.linked_entity_id = linked_entity.id
                       query_entity.linked_entity = linked_entity
                   
                   query_entities.append(query_entity)
               
               # Analyze query intent and type
               intent = self._analyze_query_intent(query, query_entities)
               query_type = self._classify_query_type(query, query_entities)
               complexity = self._calculate_complexity_score(query, query_entities)
               
               return QueryAnalysis(
                   original_query=query,
                   entities=query_entities,
                   intent=intent,
                   query_type=query_type,
                   complexity_score=complexity
               )
           
           except Exception as e:
               self.logger.error(f"Error extracting entities from query '{query}': {str(e)}")
               raise QueryProcessingError(f"Failed to extract entities: {str(e)}")
       
       async def _link_to_graph_entity(self, query_entity: QueryEntity) -> Optional[Entity]:
           """Link query entity to existing entity in knowledge graph."""
           # Search for entities with similar names
           candidates = await self.graph_storage.search_entities_by_name(
               query_entity.text, 
               entity_type=query_entity.entity_type
           )
           
           if not candidates:
               return None
           
           # Find best match using similarity scoring
           best_match = None
           best_score = 0.0
           
           for candidate in candidates:
               similarity = self._calculate_entity_similarity(query_entity, candidate)
               if similarity > best_score and similarity >= self.similarity_threshold:
                   best_score = similarity
                   best_match = candidate
           
           return best_match
       
       def _calculate_entity_similarity(self, query_entity: QueryEntity, graph_entity: Entity) -> float:
           """Calculate similarity between query entity and graph entity."""
           # Exact name match
           if query_entity.text.lower() == graph_entity.name.lower():
               return 1.0
           
           # Type compatibility
           type_match = query_entity.entity_type == graph_entity.type
           type_score = 1.0 if type_match else 0.5
           
           # Text similarity (using simple approach, could be enhanced with embeddings)
           text_similarity = self._calculate_text_similarity(
               query_entity.text.lower(), 
               graph_entity.name.lower()
           )
           
           # Alias matching
           alias_score = 0.0
           if hasattr(graph_entity, 'aliases') and graph_entity.aliases:
               for alias in graph_entity.aliases:
                   alias_sim = self._calculate_text_similarity(
                       query_entity.text.lower(), 
                       alias.lower()
                   )
                   alias_score = max(alias_score, alias_sim)
           
           # Combine scores
           final_score = (
               0.4 * text_similarity + 
               0.3 * type_score + 
               0.3 * alias_score
           )
           
           return final_score
       
       def _calculate_text_similarity(self, text1: str, text2: str) -> float:
           """Calculate text similarity using Levenshtein distance."""
           import difflib
           return difflib.SequenceMatcher(None, text1, text2).ratio()
       
       def _analyze_query_intent(self, query: str, entities: List[QueryEntity]) -> str:
           """Analyze the intent of the query."""
           query_lower = query.lower()
           
           # Intent keywords
           if any(word in query_lower for word in ['what', 'who', 'where', 'when', 'which']):
               return 'factual'
           elif any(word in query_lower for word in ['how', 'why', 'explain']):
               return 'explanatory'
           elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
               return 'comparative'
           elif any(word in query_lower for word in ['find', 'search', 'show', 'list']):
               return 'exploratory'
           else:
               return 'general'
       
       def _classify_query_type(self, query: str, entities: List[QueryEntity]) -> str:
           """Classify the type of query based on structure and entities."""
           if len(entities) == 0:
               return 'general'
           elif len(entities) == 1:
               return 'single_entity'
           elif len(entities) == 2:
               return 'entity_relationship'
           else:
               return 'multi_entity'
       
       def _calculate_complexity_score(self, query: str, entities: List[QueryEntity]) -> float:
           """Calculate query complexity score."""
           # Base complexity from query length
           length_score = min(len(query.split()) / 20.0, 1.0)
           
           # Entity complexity
           entity_score = min(len(entities) / 5.0, 1.0)
           
           # Linked entities bonus
           linked_entities = sum(1 for e in entities if e.linked_entity_id)
           linked_score = min(linked_entities / len(entities) if entities else 0, 1.0)
           
           return (length_score + entity_score + linked_score) / 3.0
   ```

2. **Query Intent Analysis**
   ```python
   # src/morag_retrieval/query/intent_analyzer.py
   class QueryIntentAnalyzer:
       def __init__(self):
           self.intent_patterns = {
               'factual': [
                   r'\bwhat is\b', r'\bwho is\b', r'\bwhere is\b', r'\bwhen did\b',
                   r'\bdefine\b', r'\btell me about\b'
               ],
               'comparative': [
                   r'\bcompare\b', r'\bdifference between\b', r'\bversus\b', r'\bvs\b',
                   r'\bbetter than\b', r'\bsimilar to\b'
               ],
               'procedural': [
                   r'\bhow to\b', r'\bsteps to\b', r'\bprocess of\b', r'\bway to\b'
               ],
               'causal': [
                   r'\bwhy does\b', r'\bcause of\b', r'\breason for\b', r'\bdue to\b'
               ]
           }
       
       def analyze_intent(self, query: str, entities: List[QueryEntity]) -> Dict[str, float]:
           """Analyze query intent with confidence scores."""
           intent_scores = {}
           query_lower = query.lower()
           
           for intent, patterns in self.intent_patterns.items():
               score = 0.0
               for pattern in patterns:
                   if re.search(pattern, query_lower):
                       score = max(score, 0.8)
               
               # Adjust based on entity types
               if entities:
                   entity_types = [e.entity_type for e in entities]
                   if intent == 'factual' and 'PERSON' in entity_types:
                       score += 0.1
                   elif intent == 'comparative' and len(entities) >= 2:
                       score += 0.2
               
               intent_scores[intent] = min(score, 1.0)
           
           return intent_scores
   ```

#### Deliverables
- Query entity extraction and linking system
- Intent analysis and query classification
- Entity similarity matching algorithms
- Comprehensive error handling and logging

### 3.1.2: Graph-Guided Context Expansion
**Estimated Time**: 4-5 days  
**Priority**: High

#### Implementation Steps

1. **Context Expansion Engine**
   ```python
   # src/morag_retrieval/context/expansion_engine.py
   from typing import List, Dict, Set, Optional
   from dataclasses import dataclass
   from morag_graph.query import GraphQueryEngine
   from morag_retrieval.query.entity_extractor import QueryAnalysis
   
   @dataclass
   class ContextExpansionConfig:
       max_expansion_depth: int = 2
       max_entities_per_hop: int = 10
       relation_type_weights: Dict[str, float] = None
       entity_type_priorities: Dict[str, float] = None
       expansion_strategies: List[str] = None
   
   @dataclass
   class ExpandedContext:
       original_entities: List[str]
       expanded_entities: List[Entity]
       expansion_paths: List[GraphPath]
       context_score: float
       expansion_reasoning: str
   
   class ContextExpansionEngine:
       def __init__(
           self, 
           graph_query_engine: GraphQueryEngine,
           config: ContextExpansionConfig = None
       ):
           self.graph_query_engine = graph_query_engine
           self.config = config or ContextExpansionConfig()
           self.logger = logging.getLogger(__name__)
       
       async def expand_context(
           self, 
           query_analysis: QueryAnalysis
       ) -> ExpandedContext:
           """Expand context based on query entities and intent."""
           try:
               # Get linked entities from query
               linked_entities = [
                   e.linked_entity_id for e in query_analysis.entities 
                   if e.linked_entity_id
               ]
               
               if not linked_entities:
                   return ExpandedContext(
                       original_entities=[],
                       expanded_entities=[],
                       expansion_paths=[],
                       context_score=0.0,
                       expansion_reasoning="No linked entities found in query"
                   )
               
               # Choose expansion strategy based on query type and intent
               strategy = self._select_expansion_strategy(query_analysis)
               
               # Perform context expansion
               expanded_entities, expansion_paths = await self._execute_expansion_strategy(
                   linked_entities, strategy, query_analysis
               )
               
               # Calculate context relevance score
               context_score = self._calculate_context_score(
                   query_analysis, expanded_entities, expansion_paths
               )
               
               return ExpandedContext(
                   original_entities=linked_entities,
                   expanded_entities=expanded_entities,
                   expansion_paths=expansion_paths,
                   context_score=context_score,
                   expansion_reasoning=f"Used {strategy} strategy based on {query_analysis.query_type} query"
               )
           
           except Exception as e:
               self.logger.error(f"Error expanding context: {str(e)}")
               raise ContextExpansionError(f"Failed to expand context: {str(e)}")
       
       def _select_expansion_strategy(self, query_analysis: QueryAnalysis) -> str:
           """Select appropriate expansion strategy based on query characteristics."""
           if query_analysis.query_type == "single_entity":
               if query_analysis.intent == "factual":
                   return "direct_neighbors"
               elif query_analysis.intent == "exploratory":
                   return "breadth_first"
           elif query_analysis.query_type == "entity_relationship":
               return "shortest_path"
           elif query_analysis.query_type == "multi_entity":
               return "subgraph_extraction"
           else:
               return "adaptive"
       
       async def _execute_expansion_strategy(
           self, 
           entity_ids: List[str], 
           strategy: str, 
           query_analysis: QueryAnalysis
       ) -> Tuple[List[Entity], List[GraphPath]]:
           """Execute the selected expansion strategy."""
           expanded_entities = []
           expansion_paths = []
           
           if strategy == "direct_neighbors":
               expanded_entities, expansion_paths = await self._expand_direct_neighbors(entity_ids)
           elif strategy == "breadth_first":
               expanded_entities, expansion_paths = await self._expand_breadth_first(entity_ids)
           elif strategy == "shortest_path":
               expanded_entities, expansion_paths = await self._expand_shortest_paths(entity_ids)
           elif strategy == "subgraph_extraction":
               expanded_entities, expansion_paths = await self._extract_subgraph(entity_ids)
           elif strategy == "adaptive":
               expanded_entities, expansion_paths = await self._adaptive_expansion(
                   entity_ids, query_analysis
               )
           
           return expanded_entities, expansion_paths
       
       async def _expand_direct_neighbors(self, entity_ids: List[str]) -> Tuple[List[Entity], List[GraphPath]]:
           """Expand to direct neighbors of given entities."""
           all_entities = []
           all_paths = []
           
           for entity_id in entity_ids:
               paths = await self.graph_query_engine.find_related_entities(
                   entity_id, 
                   max_depth=1, 
                   max_results=self.config.max_entities_per_hop
               )
               
               for path in paths:
                   if path.entities:
                       all_entities.extend(path.entities)
                       all_paths.append(path)
           
           # Remove duplicates while preserving order
           unique_entities = []
           seen_ids = set()
           for entity in all_entities:
               if entity.id not in seen_ids:
                   unique_entities.append(entity)
                   seen_ids.add(entity.id)
           
           return unique_entities, all_paths
       
       async def _expand_breadth_first(self, entity_ids: List[str]) -> Tuple[List[Entity], List[GraphPath]]:
           """Expand using breadth-first traversal."""
           all_entities = []
           all_paths = []
           
           for entity_id in entity_ids:
               paths = await self.graph_query_engine.find_related_entities(
                   entity_id,
                   max_depth=self.config.max_expansion_depth,
                   max_results=self.config.max_entities_per_hop * 2
               )
               
               # Sort by path length and relevance
               sorted_paths = sorted(paths, key=lambda p: (len(p.entities), -p.total_weight))
               
               for path in sorted_paths[:self.config.max_entities_per_hop]:
                   all_entities.extend(path.entities)
                   all_paths.append(path)
           
           return self._deduplicate_entities(all_entities), all_paths
       
       async def _expand_shortest_paths(self, entity_ids: List[str]) -> Tuple[List[Entity], List[GraphPath]]:
           """Find shortest paths between all pairs of entities."""
           all_entities = []
           all_paths = []
           
           # Find paths between all pairs
           for i, entity1 in enumerate(entity_ids):
               for entity2 in entity_ids[i+1:]:
                   path = await self.graph_query_engine.find_shortest_path(
                       entity1, entity2, max_depth=self.config.max_expansion_depth
                   )
                   
                   if path:
                       all_entities.extend(path.entities)
                       all_paths.append(path)
           
           return self._deduplicate_entities(all_entities), all_paths
       
       def _calculate_context_score(self, query_analysis: QueryAnalysis, entities: List[Entity], paths: List[GraphPath]) -> float:
           """Calculate relevance score for expanded context."""
           if not entities:
               return 0.0
           
           # Base score from number of entities
           entity_score = min(len(entities) / 20.0, 1.0)
           
           # Path quality score
           if paths:
               avg_path_weight = sum(p.total_weight for p in paths) / len(paths)
               path_score = 1.0 / (1.0 + avg_path_weight)
           else:
               path_score = 0.0
           
           # Entity type diversity
           entity_types = set(e.type for e in entities)
           diversity_score = min(len(entity_types) / 5.0, 1.0)
           
           # Query complexity bonus
           complexity_bonus = query_analysis.complexity_score * 0.2
           
           return min((entity_score + path_score + diversity_score + complexity_bonus) / 3.0, 1.0)
       
       def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
           """Remove duplicate entities while preserving order."""
           unique_entities = []
           seen_ids = set()
           
           for entity in entities:
               if entity.id not in seen_ids:
                   unique_entities.append(entity)
                   seen_ids.add(entity.id)
           
           return unique_entities
   ```

2. **Adaptive Expansion Strategies**
   ```python
   # src/morag_retrieval/context/adaptive_strategies.py
   class AdaptiveExpansionStrategies:
       def __init__(self, graph_query_engine: GraphQueryEngine):
           self.graph_query_engine = graph_query_engine
       
       async def adaptive_expansion(
           self, 
           entity_ids: List[str], 
           query_analysis: QueryAnalysis
       ) -> Tuple[List[Entity], List[GraphPath]]:
           """Adaptively choose expansion based on graph structure and query."""
           # Analyze local graph density around entities
           density_scores = await self._analyze_local_density(entity_ids)
           
           # Choose strategy based on density and query characteristics
           if max(density_scores.values()) > 0.7:  # High density
               if query_analysis.intent == "comparative":
                   return await self._focused_comparison_expansion(entity_ids)
               else:
                   return await self._selective_expansion(entity_ids)
           else:  # Low density
               return await self._broad_exploration_expansion(entity_ids)
       
       async def _analyze_local_density(self, entity_ids: List[str]) -> Dict[str, float]:
           """Analyze local graph density around entities."""
           density_scores = {}
           
           for entity_id in entity_ids:
               neighbors = await self.graph_query_engine.find_related_entities(
                   entity_id, max_depth=1, max_results=50
               )
               
               # Calculate density as ratio of actual connections to possible connections
               num_neighbors = len(neighbors)
               if num_neighbors > 1:
                   # Simplified density calculation
                   density = min(num_neighbors / 20.0, 1.0)
               else:
                   density = 0.0
               
               density_scores[entity_id] = density
           
           return density_scores
   ```

#### Deliverables
- Context expansion engine with multiple strategies
- Adaptive expansion based on graph structure
- Context relevance scoring
- Performance optimization for large graphs

### 3.1.3: Result Fusion System
**Estimated Time**: 3-4 days  
**Priority**: High

#### Implementation Steps

1. **Hybrid Retrieval Coordinator**
   ```python
   # src/morag_retrieval/fusion/hybrid_coordinator.py
   from typing import List, Dict, Optional, Tuple
   from dataclasses import dataclass
   from morag_retrieval.vector import VectorRetriever
   from morag_retrieval.context import ContextExpansionEngine
   from morag_retrieval.query import QueryEntityExtractor
   
   @dataclass
   class RetrievalResult:
       content: str
       source: str  # "vector", "graph", "hybrid"
       score: float
       metadata: Dict[str, Any]
       entities: List[str] = None
       reasoning: str = ""
   
   @dataclass
   class HybridRetrievalConfig:
       vector_weight: float = 0.6
       graph_weight: float = 0.4
       max_vector_results: int = 20
       max_graph_results: int = 15
       fusion_strategy: str = "weighted_combination"  # "weighted_combination", "rank_fusion", "adaptive"
       min_confidence_threshold: float = 0.3
   
   class HybridRetrievalCoordinator:
       def __init__(
           self,
           vector_retriever: VectorRetriever,
           context_expansion_engine: ContextExpansionEngine,
           query_entity_extractor: QueryEntityExtractor,
           config: HybridRetrievalConfig = None
       ):
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
           """Perform hybrid retrieval combining vector and graph methods."""
           try:
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
               
               # Fuse results
               fused_results = await self._fuse_results(
                   vector_results, graph_results, query_analysis
               )
               
               # Rank and filter final results
               final_results = self._rank_and_filter_results(
                   fused_results, max_results
               )
               
               return final_results
           
           except Exception as e:
               self.logger.error(f"Error in hybrid retrieval: {str(e)}")
               # Fallback to vector retrieval only
               try:
                   vector_results = await self._vector_retrieval(query)
                   return vector_results[:max_results]
               except Exception as fallback_error:
                   self.logger.error(f"Fallback retrieval also failed: {fallback_error}")
                   raise RetrievalError(f"All retrieval methods failed: {str(e)}")
       
       async def _vector_retrieval(self, query: str) -> List[RetrievalResult]:
           """Perform traditional vector-based retrieval."""
           vector_docs = await self.vector_retriever.search(
               query, limit=self.config.max_vector_results
           )
           
           results = []
           for doc in vector_docs:
               result = RetrievalResult(
                   content=doc.content,
                   source="vector",
                   score=doc.score,
                   metadata=doc.metadata,
                   reasoning="Retrieved via semantic similarity"
               )
               results.append(result)
           
           return results
       
       async def _graph_retrieval(self, query_analysis: QueryAnalysis) -> List[RetrievalResult]:
           """Perform graph-guided retrieval."""
           if not query_analysis.entities:
               return []
           
           # Expand context using graph
           expanded_context = await self.context_expansion_engine.expand_context(query_analysis)
           
           if not expanded_context.expanded_entities:
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
                           "entity_type": entity.type,
                           "expansion_reasoning": expanded_context.expansion_reasoning
                       },
                       entities=[entity.id],
                       reasoning=f"Retrieved via graph expansion from {expanded_context.expansion_reasoning}"
                   )
                   results.append(result)
           
           return results[:self.config.max_graph_results]
       
       async def _fuse_results(
           self,
           vector_results: List[RetrievalResult],
           graph_results: List[RetrievalResult],
           query_analysis: QueryAnalysis
       ) -> List[RetrievalResult]:
           """Fuse vector and graph retrieval results."""
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
           """Fuse results using weighted combination of scores."""
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
       
       def _calculate_graph_relevance_score(self, entity: Entity, context: ExpandedContext) -> float:
           """Calculate relevance score for graph-retrieved content."""
           base_score = 0.5
           
           # Boost for entities in original query
           if entity.id in context.original_entities:
               base_score += 0.3
           
           # Boost based on entity importance (if available)
           if hasattr(entity, 'importance_score'):
               base_score += entity.importance_score * 0.2
           
           # Context quality bonus
           base_score += context.context_score * 0.2
           
           return min(base_score, 1.0)
       
       def _rank_and_filter_results(
           self, 
           results: List[RetrievalResult], 
           max_results: int
       ) -> List[RetrievalResult]:
           """Rank and filter final results."""
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
   ```

2. **Advanced Fusion Strategies**
   ```python
   # src/morag_retrieval/fusion/fusion_strategies.py
   class AdvancedFusionStrategies:
       @staticmethod
       def reciprocal_rank_fusion(
           vector_results: List[RetrievalResult],
           graph_results: List[RetrievalResult],
           k: int = 60
       ) -> List[RetrievalResult]:
           """Implement Reciprocal Rank Fusion (RRF)."""
           # Create content to result mapping
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
           
           return sorted(fused_results, key=lambda r: r.score, reverse=True)
   ```

#### Deliverables
- Hybrid retrieval coordinator
- Multiple fusion strategies (weighted, rank fusion, adaptive)
- Result ranking and filtering
- Performance monitoring and fallback mechanisms

## Testing Requirements

### Unit Tests
```python
# tests/test_hybrid_retrieval.py
import pytest
from morag_retrieval.fusion import HybridRetrievalCoordinator
from morag_retrieval.query import QueryEntityExtractor

class TestHybridRetrieval:
    @pytest.mark.asyncio
    async def test_entity_extraction_and_linking(self, mock_extractors):
        extractor = QueryEntityExtractor(
            mock_extractors['entity'], 
            mock_extractors['graph_storage']
        )
        
        analysis = await extractor.extract_and_link_entities(
            "What is the relationship between Einstein and quantum physics?"
        )
        
        assert len(analysis.entities) > 0
        assert analysis.intent in ['factual', 'exploratory']
        assert analysis.query_type == 'entity_relationship'
    
    @pytest.mark.asyncio
    async def test_context_expansion(self, mock_expansion_engine):
        query_analysis = QueryAnalysis(
            original_query="test query",
            entities=[mock_query_entity],
            intent="factual",
            query_type="single_entity",
            complexity_score=0.5
        )
        
        context = await mock_expansion_engine.expand_context(query_analysis)
        
        assert isinstance(context, ExpandedContext)
        assert len(context.expanded_entities) > 0
        assert context.context_score > 0
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_integration(self, mock_hybrid_coordinator):
        results = await mock_hybrid_coordinator.retrieve(
            "Explain machine learning algorithms", 
            max_results=10
        )
        
        assert len(results) <= 10
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.score > 0 for r in results)
```

### Integration Tests
```python
# tests/integration/test_end_to_end_retrieval.py
class TestEndToEndRetrieval:
    @pytest.mark.asyncio
    async def test_complete_retrieval_pipeline(self, full_system_setup):
        """Test complete retrieval pipeline from query to results."""
        coordinator = full_system_setup['coordinator']
        
        test_queries = [
            "What are the applications of neural networks?",
            "Compare supervised and unsupervised learning",
            "How does backpropagation work?"
        ]
        
        for query in test_queries:
            results = await coordinator.retrieve(query, max_results=5)
            
            assert len(results) > 0
            assert all(r.score > 0.3 for r in results)  # Minimum quality threshold
            assert results == sorted(results, key=lambda r: r.score, reverse=True)
```

## Success Criteria

- [ ] Query entity extraction achieves >85% accuracy
- [ ] Entity linking to graph achieves >80% precision
- [ ] Context expansion provides relevant entities
- [ ] Hybrid retrieval outperforms vector-only baseline
- [ ] Result fusion improves relevance scores
- [ ] System handles edge cases gracefully
- [ ] Performance targets met (< 3 seconds end-to-end)
- [ ] Unit test coverage > 90%
- [ ] Integration tests pass

## Performance Targets

- **Query Processing**: < 500ms for entity extraction and linking
- **Context Expansion**: < 1 second for typical queries
- **Result Fusion**: < 500ms for combining results
- **End-to-End Retrieval**: < 3 seconds total
- **Memory Usage**: < 1GB during peak operation

## Next Steps

After completing this task:
1. Proceed to **Task 3.2**: Sparse Vector Integration
2. Implement enhanced query endpoints
3. Add performance monitoring and optimization

## Dependencies

**Requires**:
- Task 2.3: Graph Traversal
- Task 1.3: NLP Pipeline Foundation
- Task 2.1: Relation Extraction

**Enables**:
- Task 3.2: Sparse Vector Integration
- Task 3.3: Enhanced Query Endpoints
- Task 4.1: Multi-Hop Reasoning