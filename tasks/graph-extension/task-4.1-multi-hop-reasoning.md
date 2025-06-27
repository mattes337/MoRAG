# Task 4.1: Multi-Hop Reasoning

**Phase**: 4 - Advanced Features  
**Priority**: High  
**Estimated Time**: 9-11 days total  
**Dependencies**: Task 2.3 (Hybrid Retrieval System), Task 3.3 (Enhanced Query Endpoints)

## Overview

This task implements advanced multi-hop reasoning capabilities that enable the system to perform complex reasoning across multiple entities and relationships in the knowledge graph. The system uses LLM-guided path selection and iterative context refinement to answer complex queries that require connecting information from multiple sources.

## Subtasks

### 4.1.1: LLM-Guided Path Selection
**Estimated Time**: 5-6 days  
**Priority**: High

#### Implementation Steps

1. **Path Selection Agent**
   ```python
   # src/morag_reasoning/path_selection.py
   from typing import List, Dict, Any, Optional, Tuple
   from dataclasses import dataclass
   import asyncio
   import logging
   from morag_graph.models import GraphPath, Entity, Relation
   from morag_llm.client import LLMClient
   
   @dataclass
   class PathRelevanceScore:
       path: GraphPath
       relevance_score: float
       confidence: float
       reasoning: str
   
   @dataclass
   class ReasoningStrategy:
       name: str
       description: str
       max_depth: int
       bidirectional: bool
       use_weights: bool
   
   class PathSelectionAgent:
       def __init__(self, llm_client: LLMClient, max_paths: int = 10):
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
           """Use LLM to select most relevant paths for answering the query."""
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
               return f"Single entity: {path.entities[0] if path.entities else 'Unknown'}"
           
           description_parts = []
           for i in range(len(path.entities) - 1):
               entity1 = path.entities[i]
               entity2 = path.entities[i + 1]
               relation = path.relations[i] if i < len(path.relations) else "connected_to"
               description_parts.append(f"{entity1} --[{relation}]--> {entity2}")
           
           return " -> ".join(description_parts)
       
       def _parse_path_selection(self, response: str, available_paths: List[GraphPath]) -> List[PathRelevanceScore]:
           """Parse LLM response to extract selected paths."""
           try:
               import json
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
                   # Boost score based on path weights
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
               weight_score = min(path.total_weight, 10)
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
       def __init__(self, graph_engine, path_selector: PathSelectionAgent):
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
           """Find and select reasoning paths for multi-hop queries."""
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
                       paths = await self.graph_engine.find_bidirectional_paths(
                           start_entity, target_entity, strategy_config.max_depth
                       )
                       all_paths.extend(paths)
           
           elif strategy == "backward_chaining" and target_entities:
               # Backward chaining from target entities
               for target_entity in target_entities:
                   paths = await self.graph_engine.traverse_backward(
                       target_entity, strategy_config.max_depth
                   )
                   # Filter paths that connect to start entities
                   relevant_paths = [
                       path for path in paths 
                       if any(entity in start_entities for entity in path.entities)
                   ]
                   all_paths.extend(relevant_paths)
           
           else:
               # Forward chaining (default)
               for start_entity in start_entities:
                   paths = await self.graph_engine.traverse(
                       start_entity, 
                       algorithm="bfs",
                       max_depth=strategy_config.max_depth
                   )
                   all_paths.extend(paths.get('paths', []))
           
           # Remove duplicates and limit results
           unique_paths = self._deduplicate_paths(all_paths)
           return unique_paths[:max_paths]
       
       def _deduplicate_paths(self, paths: List[GraphPath]) -> List[GraphPath]:
           """Remove duplicate paths based on entity sequences."""
           seen_sequences = set()
           unique_paths = []
           
           for path in paths:
               sequence = tuple(path.entities)
               if sequence not in seen_sequences:
                   seen_sequences.add(sequence)
                   unique_paths.append(path)
           
           return unique_paths
   ```

#### Deliverables
- LLM-guided path selection agent
- Multiple reasoning strategies (forward, backward, bidirectional)
- Path ranking and scoring system
- Fallback mechanisms for robust operation

### 4.1.2: Iterative Context Refinement
**Estimated Time**: 4-5 days  
**Priority**: High

#### Implementation Steps

1. **Iterative Retrieval System**
   ```python
   # src/morag_reasoning/iterative_retrieval.py
   from typing import List, Dict, Any, Optional, Set
   from dataclasses import dataclass, field
   import asyncio
   import logging
   
   @dataclass
   class ContextGap:
       gap_type: str  # "missing_entity", "missing_relation", "insufficient_detail"
       description: str
       entities_needed: List[str] = field(default_factory=list)
       relations_needed: List[str] = field(default_factory=list)
       priority: float = 1.0
   
   @dataclass
   class ContextAnalysis:
       is_sufficient: bool
       confidence: float
       gaps: List[ContextGap]
       reasoning: str
       suggested_queries: List[str] = field(default_factory=list)
   
   @dataclass
   class RetrievalContext:
       entities: Dict[str, Any] = field(default_factory=dict)
       relations: List[Dict[str, Any]] = field(default_factory=list)
       documents: List[Dict[str, Any]] = field(default_factory=list)
       paths: List[GraphPath] = field(default_factory=list)
       metadata: Dict[str, Any] = field(default_factory=dict)
   
   class IterativeRetriever:
       def __init__(
           self, 
           llm_client: LLMClient,
           graph_engine,
           vector_retriever,
           max_iterations: int = 5,
           sufficiency_threshold: float = 0.8
       ):
           self.llm_client = llm_client
           self.graph_engine = graph_engine
           self.vector_retriever = vector_retriever
           self.max_iterations = max_iterations
           self.sufficiency_threshold = sufficiency_threshold
           self.logger = logging.getLogger(__name__)
       
       async def refine_context(
           self, 
           query: str, 
           initial_context: RetrievalContext
       ) -> RetrievalContext:
           """Iteratively refine context until sufficient for answering the query."""
           current_context = initial_context
           iteration_count = 0
           
           self.logger.info(f"Starting iterative context refinement for query: {query}")
           
           while iteration_count < self.max_iterations:
               iteration_count += 1
               self.logger.info(f"Iteration {iteration_count}/{self.max_iterations}")
               
               # Analyze current context
               analysis = await self._analyze_context(query, current_context)
               
               self.logger.info(
                   f"Context analysis - Sufficient: {analysis.is_sufficient}, "
                   f"Confidence: {analysis.confidence:.2f}, Gaps: {len(analysis.gaps)}"
               )
               
               # Check if context is sufficient
               if analysis.is_sufficient and analysis.confidence >= self.sufficiency_threshold:
                   self.logger.info("Context deemed sufficient, stopping refinement")
                   break
               
               # Retrieve additional information based on gaps
               additional_context = await self._retrieve_additional(
                   query, analysis.gaps, current_context
               )
               
               # Merge contexts
               current_context = self._merge_context(current_context, additional_context)
               
               # Log progress
               self.logger.info(
                   f"Added {len(additional_context.entities)} entities, "
                   f"{len(additional_context.relations)} relations, "
                   f"{len(additional_context.documents)} documents"
               )
           
           # Final analysis
           final_analysis = await self._analyze_context(query, current_context)
           current_context.metadata['final_analysis'] = final_analysis
           current_context.metadata['iterations_used'] = iteration_count
           
           self.logger.info(
               f"Context refinement completed after {iteration_count} iterations. "
               f"Final confidence: {final_analysis.confidence:.2f}"
           )
           
           return current_context
       
       async def _analyze_context(self, query: str, context: RetrievalContext) -> ContextAnalysis:
           """Analyze current context to determine if it's sufficient."""
           try:
               # Create context analysis prompt
               prompt = self._create_analysis_prompt(query, context)
               
               # Get LLM analysis
               response = await self.llm_client.generate(
                   prompt=prompt,
                   max_tokens=800,
                   temperature=0.1
               )
               
               # Parse analysis
               analysis = self._parse_context_analysis(response)
               return analysis
           
           except Exception as e:
               self.logger.error(f"Error in context analysis: {str(e)}")
               # Fallback analysis
               return ContextAnalysis(
                   is_sufficient=len(context.entities) > 0 and len(context.documents) > 0,
                   confidence=0.5,
                   gaps=[],
                   reasoning="Fallback analysis due to LLM error"
               )
       
       def _create_analysis_prompt(self, query: str, context: RetrievalContext) -> str:
           """Create prompt for context analysis."""
           prompt = f"""Analyze whether the provided context is sufficient to answer the given query.

Query: {query}

Current Context:

Entities ({len(context.entities)}):
"""
           
           # Add entity information
           for entity_id, entity_data in list(context.entities.items())[:10]:  # Limit for token efficiency
               prompt += f"- {entity_id}: {entity_data.get('type', 'Unknown')}\n"
           
           prompt += f"\nRelations ({len(context.relations)}):\n"
           
           # Add relation information
           for relation in context.relations[:10]:
               prompt += f"- {relation.get('subject', '?')} --[{relation.get('predicate', '?')}]--> {relation.get('object', '?')}\n"
           
           prompt += f"\nDocuments ({len(context.documents)}):\n"
           
           # Add document information
           for doc in context.documents[:5]:
               content_preview = doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', '')
               prompt += f"- {doc.get('id', 'Unknown')}: {content_preview}\n"
           
           prompt += """
Analyze this context and provide:
1. Is the context sufficient to answer the query? (true/false)
2. Confidence level (0-10)
3. What gaps exist in the context?
4. Suggested additional queries to fill gaps

Format as JSON:
{
  "is_sufficient": false,
  "confidence": 6.5,
  "reasoning": "Context provides basic information but lacks specific details about...",
  "gaps": [
    {
      "gap_type": "missing_entity",
      "description": "Need more information about entity X",
      "entities_needed": ["entity_name"],
      "priority": 0.8
    }
  ],
  "suggested_queries": ["What is the relationship between X and Y?"]
}"""
           
           return prompt
       
       def _parse_context_analysis(self, response: str) -> ContextAnalysis:
           """Parse LLM response into ContextAnalysis object."""
           try:
               import json
               data = json.loads(response)
               
               gaps = []
               for gap_data in data.get("gaps", []):
                   gap = ContextGap(
                       gap_type=gap_data.get("gap_type", "unknown"),
                       description=gap_data.get("description", ""),
                       entities_needed=gap_data.get("entities_needed", []),
                       relations_needed=gap_data.get("relations_needed", []),
                       priority=float(gap_data.get("priority", 1.0))
                   )
                   gaps.append(gap)
               
               return ContextAnalysis(
                   is_sufficient=bool(data.get("is_sufficient", False)),
                   confidence=float(data.get("confidence", 0)) / 10.0,  # Normalize to 0-1
                   gaps=gaps,
                   reasoning=data.get("reasoning", ""),
                   suggested_queries=data.get("suggested_queries", [])
               )
           
           except Exception as e:
               self.logger.error(f"Error parsing context analysis: {str(e)}")
               return ContextAnalysis(
                   is_sufficient=False,
                   confidence=0.3,
                   gaps=[],
                   reasoning="Failed to parse LLM analysis"
               )
       
       async def _retrieve_additional(
           self, 
           query: str, 
           gaps: List[ContextGap], 
           current_context: RetrievalContext
       ) -> RetrievalContext:
           """Retrieve additional information to fill context gaps."""
           additional_context = RetrievalContext()
           
           # Sort gaps by priority
           sorted_gaps = sorted(gaps, key=lambda g: g.priority, reverse=True)
           
           for gap in sorted_gaps[:3]:  # Process top 3 gaps to avoid overwhelming
               try:
                   if gap.gap_type == "missing_entity":
                       # Retrieve entity information
                       for entity_name in gap.entities_needed:
                           entity_info = await self.graph_engine.get_entity_details(entity_name)
                           if entity_info:
                               additional_context.entities[entity_name] = entity_info
                   
                   elif gap.gap_type == "missing_relation":
                       # Retrieve relation information
                       for relation_name in gap.relations_needed:
                           relations = await self.graph_engine.get_relations_by_type(relation_name)
                           additional_context.relations.extend(relations)
                   
                   elif gap.gap_type == "insufficient_detail":
                       # Perform additional vector search
                       search_query = f"{query} {gap.description}"
                       vector_results = await self.vector_retriever.search(
                           search_query, limit=5
                       )
                       additional_context.documents.extend(vector_results)
               
               except Exception as e:
                   self.logger.error(f"Error retrieving additional info for gap {gap.gap_type}: {str(e)}")
                   continue
           
           return additional_context
       
       def _merge_context(
           self, 
           current_context: RetrievalContext, 
           additional_context: RetrievalContext
       ) -> RetrievalContext:
           """Merge additional context into current context."""
           # Merge entities (avoid duplicates)
           for entity_id, entity_data in additional_context.entities.items():
               if entity_id not in current_context.entities:
                   current_context.entities[entity_id] = entity_data
           
           # Merge relations (avoid duplicates)
           existing_relations = set(
               (r.get('subject'), r.get('predicate'), r.get('object')) 
               for r in current_context.relations
           )
           
           for relation in additional_context.relations:
               relation_tuple = (relation.get('subject'), relation.get('predicate'), relation.get('object'))
               if relation_tuple not in existing_relations:
                   current_context.relations.append(relation)
                   existing_relations.add(relation_tuple)
           
           # Merge documents (avoid duplicates)
           existing_doc_ids = set(doc.get('id') for doc in current_context.documents)
           for doc in additional_context.documents:
               if doc.get('id') not in existing_doc_ids:
                   current_context.documents.append(doc)
                   existing_doc_ids.add(doc.get('id'))
           
           # Merge paths
           current_context.paths.extend(additional_context.paths)
           
           # Update metadata
           current_context.metadata.update(additional_context.metadata)
           
           return current_context
   ```

#### Deliverables
- Iterative context refinement system
- Context gap analysis and identification
- Automatic additional information retrieval
- Stopping criteria based on sufficiency assessment

## Testing Requirements

### Unit Tests
```python
# tests/test_multi_hop_reasoning.py
import pytest
from morag_reasoning.path_selection import PathSelectionAgent, ReasoningPathFinder
from morag_reasoning.iterative_retrieval import IterativeRetriever, ContextAnalysis

class TestPathSelectionAgent:
    @pytest.mark.asyncio
    async def test_path_selection(self, mock_llm_client, sample_graph_paths):
        agent = PathSelectionAgent(mock_llm_client)
        
        selected_paths = await agent.select_paths(
            "What is the relationship between Apple and Microsoft?",
            sample_graph_paths
        )
        
        assert len(selected_paths) > 0
        assert all(hasattr(path, 'relevance_score') for path in selected_paths)
        assert all(path.relevance_score >= 0 for path in selected_paths)
        
        # Check that paths are sorted by relevance
        scores = [path.relevance_score for path in selected_paths]
        assert scores == sorted(scores, reverse=True)

class TestIterativeRetriever:
    @pytest.mark.asyncio
    async def test_context_refinement(self, mock_llm_client, mock_graph_engine, mock_vector_retriever):
        retriever = IterativeRetriever(
            llm_client=mock_llm_client,
            graph_engine=mock_graph_engine,
            vector_retriever=mock_vector_retriever,
            max_iterations=3
        )
        
        initial_context = RetrievalContext(
            entities={"Apple": {"type": "ORG"}},
            documents=[{"id": "doc1", "content": "Apple is a technology company"}]
        )
        
        refined_context = await retriever.refine_context(
            "What products does Apple make?",
            initial_context
        )
        
        # Should have more information after refinement
        assert len(refined_context.entities) >= len(initial_context.entities)
        assert len(refined_context.documents) >= len(initial_context.documents)
        assert 'final_analysis' in refined_context.metadata
        assert 'iterations_used' in refined_context.metadata
```

### Integration Tests
```python
# tests/integration/test_multi_hop_integration.py
class TestMultiHopIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_multi_hop_reasoning(self, test_graph, test_corpus):
        """Test complete multi-hop reasoning pipeline."""
        # Set up components
        path_selector = PathSelectionAgent(llm_client)
        path_finder = ReasoningPathFinder(graph_engine, path_selector)
        iterative_retriever = IterativeRetriever(
            llm_client, graph_engine, vector_retriever
        )
        
        # Test complex multi-hop query
        query = "How are Apple's AI research efforts related to their partnership with universities?"
        
        # Find reasoning paths
        start_entities = ["Apple Inc.", "AI research"]
        reasoning_paths = await path_finder.find_reasoning_paths(
            query, start_entities, strategy="bidirectional"
        )
        
        assert len(reasoning_paths) > 0
        
        # Create initial context from paths
        initial_context = RetrievalContext(
            paths=[path.path for path in reasoning_paths[:5]]
        )
        
        # Refine context iteratively
        refined_context = await iterative_retriever.refine_context(
            query, initial_context
        )
        
        # Verify refined context has sufficient information
        final_analysis = refined_context.metadata.get('final_analysis')
        assert final_analysis is not None
        assert final_analysis.confidence > 0.5
```

## Success Criteria

- [ ] LLM-guided path selection correctly identifies relevant reasoning paths
- [ ] Multiple reasoning strategies (forward, backward, bidirectional) work effectively
- [ ] Iterative context refinement improves answer quality
- [ ] Context gap analysis accurately identifies missing information
- [ ] System handles complex multi-hop queries requiring 3+ reasoning steps
- [ ] Performance targets met (< 10 seconds for complex reasoning)
- [ ] Unit test coverage > 90%
- [ ] Integration tests pass
- [ ] Quality metrics show improvement over baseline

## Performance Targets

- **Path Selection**: < 3 seconds for 50 candidate paths
- **Context Analysis**: < 2 seconds per iteration
- **Multi-hop Reasoning**: < 10 seconds for complex queries
- **Memory Usage**: < 2GB for large reasoning tasks

## Next Steps

After completing this task:
1. Proceed to **Task 4.2**: Performance Optimization
2. Implement comprehensive evaluation metrics
3. Fine-tune reasoning strategies based on performance data

## Dependencies

**Requires**:
- Task 3.1: Hybrid Retrieval System
- LLM integration for reasoning
- Graph traversal capabilities

**Enables**:
- Task 4.2: Performance Optimization
- Task 4.3: Monitoring & Analytics
- Advanced query answering capabilities