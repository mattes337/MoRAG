# Quick Win 1: Basic Query Classification

## Overview

**Priority**: 🔥 **Immediate** (1 week, High Impact, Very High ROI)
**Source**: LightRAG dual-level retrieval and GraphRAG global vs local classification
**Expected Impact**: 25-30% improvement in query response relevance

## Problem Statement

Currently, MoRAG uses a single recursive fact retrieval approach for all query types, regardless of whether the user is asking for:
- Specific factual information (who, what, when, where)
- Analytical insights (why, how, compare)
- Summary/overview information (summarize, overview)

This one-size-fits-all approach leads to suboptimal retrieval strategies and response quality.

## Solution Overview

Implement a simple heuristic-based query classification system that routes different query types to appropriate retrieval strategies, providing immediate performance improvements while laying groundwork for more sophisticated adaptive systems.

## Technical Implementation

### 1. Query Classification Engine

Create a new module `packages/morag-reasoning/src/morag_reasoning/query_classifier.py`:

```python
from enum import Enum
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass

class QueryType(Enum):
    FACTUAL = "factual"           # Direct fact lookup
    ANALYTICAL = "analytical"     # Multi-hop reasoning
    SUMMARY = "summary"          # Broad context retrieval
    COMPARATIVE = "comparative"   # Multi-entity comparison
    TEMPORAL = "temporal"        # Time-based queries

@dataclass
class QueryClassification:
    query_type: QueryType
    confidence: float
    reasoning: str
    suggested_strategy: str

class QueryClassifier:
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        # Cache for classification results to avoid repeated LLM calls
        self.classification_cache = {}

    async def classify(self, query: str, language: str = None) -> QueryClassification:
        """Classify query type using LLM-based analysis."""

        # Check cache first
        cache_key = f"{query}:{language or 'auto'}"
        if cache_key in self.classification_cache:
            cached_result = self.classification_cache[cache_key]
            return QueryClassification(
                query_type=QueryType(cached_result['type']),
                confidence=cached_result['confidence'],
                reasoning=cached_result['reasoning'],
                suggested_strategy=cached_result['strategy']
            )

        # Use LLM for classification
        if self.llm_service:
            classification = await self._llm_classify_query(query, language)
        else:
            # Fallback to simple heuristics
            classification = self._fallback_classify(query)

        # Cache result
        self.classification_cache[cache_key] = {
            'type': classification.query_type.value,
            'confidence': classification.confidence,
            'reasoning': classification.reasoning,
            'strategy': classification.suggested_strategy
        }

        return classification

    async def _llm_classify_query(self, query: str, language: str = None) -> QueryClassification:
        """Use LLM to classify query type."""

        prompt = f"""
        Classify the following query into one of these categories:

        1. FACTUAL: Direct fact lookup (who, what, when, where questions)
        2. ANALYTICAL: Multi-hop reasoning (why, how, explain questions)
        3. SUMMARY: Broad context retrieval (summarize, overview questions)
        4. COMPARATIVE: Multi-entity comparison (vs, compare, contrast questions)
        5. TEMPORAL: Time-based queries (timeline, sequence, history questions)

        Query: "{query}"
        Language context: {language or "auto-detect"}

        Consider the query intent, complexity, and information need.

        Respond with JSON:
        {{
            "type": "FACTUAL|ANALYTICAL|SUMMARY|COMPARATIVE|TEMPORAL",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation of classification",
            "strategy": "suggested retrieval strategy"
        }}
        """

        try:
            response = await self.llm_service.generate(prompt, max_tokens=200)
            result = json.loads(response)

            query_type = QueryType(result.get('type', 'ANALYTICAL').lower())
            confidence = float(result.get('confidence', 0.7))
            reasoning = result.get('reasoning', 'LLM classification')
            strategy = result.get('strategy', self._get_strategy(query_type))

            return QueryClassification(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                suggested_strategy=strategy
            )

        except Exception as e:
            # Fallback on LLM failure
            return self._fallback_classify(query)

    def _fallback_classify(self, query: str) -> QueryClassification:
        """Fallback classification when LLM is unavailable."""
        # Simple heuristics based on query length and structure
        query_lower = query.lower()

        # Very simple pattern matching for common cases
        if '?' in query and len(query.split()) <= 6:
            query_type = QueryType.FACTUAL
            confidence = 0.6
        elif len(query.split()) > 15:
            query_type = QueryType.SUMMARY
            confidence = 0.5
        else:
            query_type = QueryType.ANALYTICAL
            confidence = 0.4

        return QueryClassification(
            query_type=query_type,
            confidence=confidence,
            reasoning="Fallback heuristic classification",
            suggested_strategy=self._get_strategy(query_type)
        )

    def _get_strategy(self, query_type: QueryType) -> str:
        """Map query type to retrieval strategy."""
        strategy_map = {
            QueryType.FACTUAL: "direct_entity_lookup",
            QueryType.ANALYTICAL: "multi_hop_traversal",
            QueryType.SUMMARY: "broad_context_retrieval",
            QueryType.COMPARATIVE: "multi_entity_comparison",
            QueryType.TEMPORAL: "temporal_graph_traversal"
        }
        return strategy_map[query_type]
```

### 2. Strategy Router

Create `packages/morag-reasoning/src/morag_reasoning/strategy_router.py`:

```python
from typing import Dict, List, Any
from .query_classifier import QueryClassifier, QueryType, QueryClassification
from ..graph_traversal_agent import GraphTraversalAgent

class StrategyRouter:
    def __init__(self, graph_agent: GraphTraversalAgent, llm_service=None):
        self.classifier = QueryClassifier(llm_service)
        self.graph_agent = graph_agent

    async def route_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Route query to appropriate retrieval strategy."""
        language = context.get('language')
        classification = await self.classifier.classify(query, language)

        # Log classification for monitoring
        context['query_classification'] = {
            'type': classification.query_type.value,
            'confidence': classification.confidence,
            'reasoning': classification.reasoning,
            'strategy': classification.suggested_strategy
        }

        # Route to appropriate strategy
        if classification.query_type == QueryType.FACTUAL:
            return await self._factual_strategy(query, context)
        elif classification.query_type == QueryType.ANALYTICAL:
            return await self._analytical_strategy(query, context)
        elif classification.query_type == QueryType.SUMMARY:
            return await self._summary_strategy(query, context)
        elif classification.query_type == QueryType.COMPARATIVE:
            return await self._comparative_strategy(query, context)
        elif classification.query_type == QueryType.TEMPORAL:
            return await self._temporal_strategy(query, context)
        else:
            # Fallback to current approach
            return await self._analytical_strategy(query, context)

    async def _factual_strategy(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Direct entity lookup for factual queries."""
        # Extract entities and do direct lookup
        # Limit to 1-hop traversal for speed
        context['max_hops'] = 1
        context['focus_entities'] = True
        return await self.graph_agent.process_query(query, context)

    async def _analytical_strategy(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-hop traversal for analytical queries."""
        # Use existing recursive fact retrieval
        context['max_hops'] = 3
        context['enable_reasoning'] = True
        return await self.graph_agent.process_query(query, context)

    async def _summary_strategy(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Broad context retrieval for summary queries."""
        # Retrieve more chunks with lower similarity threshold
        context['max_chunks'] = 15
        context['similarity_threshold'] = 0.6
        context['enable_summarization'] = True
        return await self.graph_agent.process_query(query, context)

    async def _comparative_strategy(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-entity comparison strategy."""
        # Extract multiple entities and compare
        context['max_hops'] = 2
        context['enable_comparison'] = True
        context['extract_multiple_entities'] = True
        return await self.graph_agent.process_query(query, context)

    async def _temporal_strategy(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Temporal graph traversal strategy."""
        # Focus on temporal relationships
        context['temporal_focus'] = True
        context['max_hops'] = 2
        context['sort_by_time'] = True
        return await self.graph_agent.process_query(query, context)
```

### 3. Integration Points

#### 3.1 Update GraphTraversalAgent

Modify `packages/morag-reasoning/src/morag_reasoning/graph_traversal_agent.py`:

```python
# Add to imports
from .strategy_router import StrategyRouter

class GraphTraversalAgent:
    def __init__(self, llm_service=None, ...):
        # ... existing initialization
        self.strategy_router = StrategyRouter(self, llm_service)

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced query processing with strategy routing."""
        if context is None:
            context = {}

        # Use strategy router if enabled
        if context.get('use_strategy_routing', True):
            return await self.strategy_router.route_query(query, context)
        else:
            # Fallback to existing logic
            return await self._original_process_query(query, context)
```

#### 3.2 Update API Endpoints

Modify query endpoints to use strategy routing:

```python
# In packages/morag-services/src/morag_services/api/query_endpoints.py

@router.post("/query")
async def enhanced_query(request: QueryRequest):
    """Enhanced query with automatic strategy selection."""
    context = {
        'use_strategy_routing': True,
        'collection_name': request.collection_name,
        'max_results': request.max_results or 10
    }

    result = await graph_agent.process_query(request.query, context)

    # Include classification info in response
    return {
        'answer': result['answer'],
        'sources': result['sources'],
        'query_classification': result.get('query_classification'),
        'processing_time': result.get('processing_time')
    }
```

## Testing Strategy

### 1. Unit Tests

Create `tests/unit/test_query_classification.py`:

```python
import pytest
from morag_reasoning.query_classifier import QueryClassifier, QueryType

class TestQueryClassifier:
    def setup_method(self):
        self.classifier = QueryClassifier()

    def test_factual_queries(self):
        queries = [
            "Who is the CEO of Tesla?",
            "What is machine learning?",
            "When was Python created?",
            "Where is Google headquarters?"
        ]
        for query in queries:
            result = self.classifier.classify(query)
            assert result.query_type == QueryType.FACTUAL

    def test_analytical_queries(self):
        queries = [
            "Why did the stock market crash?",
            "How does neural network training work?",
            "Explain the relationship between AI and ML"
        ]
        for query in queries:
            result = self.classifier.classify(query)
            assert result.query_type == QueryType.ANALYTICAL

    def test_summary_queries(self):
        queries = [
            "Summarize the key points about climate change",
            "Give me an overview of quantum computing",
            "Tell me about artificial intelligence"
        ]
        for query in queries:
            result = self.classifier.classify(query)
            assert result.query_type == QueryType.SUMMARY
```

### 2. Integration Tests

Create `tests/integration/test_strategy_routing.py`:

```python
import pytest
from morag_reasoning.strategy_router import StrategyRouter

class TestStrategyRouting:
    @pytest.mark.asyncio
    async def test_factual_query_routing(self):
        # Test that factual queries use direct lookup
        pass

    @pytest.mark.asyncio
    async def test_analytical_query_routing(self):
        # Test that analytical queries use multi-hop traversal
        pass
```

## Configuration

Add configuration options to `configs/`:

```yaml
# query_classification.yml
query_classification:
  enabled: true
  confidence_threshold: 0.3
  default_strategy: "analytical"

  strategies:
    factual:
      max_hops: 1
      max_chunks: 5
      similarity_threshold: 0.8

    analytical:
      max_hops: 3
      max_chunks: 10
      similarity_threshold: 0.7

    summary:
      max_hops: 2
      max_chunks: 15
      similarity_threshold: 0.6

    comparative:
      max_hops: 2
      max_chunks: 12
      similarity_threshold: 0.7

    temporal:
      max_hops: 2
      max_chunks: 8
      similarity_threshold: 0.7
      sort_by_time: true
```

## Monitoring and Metrics

### 1. Classification Metrics

Track classification accuracy and distribution:

```python
# Add to monitoring
classification_metrics = {
    'total_queries': 0,
    'classification_distribution': {
        'factual': 0,
        'analytical': 0,
        'summary': 0,
        'comparative': 0,
        'temporal': 0
    },
    'average_confidence': 0.0,
    'strategy_performance': {
        'factual': {'avg_response_time': 0, 'success_rate': 0},
        'analytical': {'avg_response_time': 0, 'success_rate': 0},
        # ... etc
    }
}
```

### 2. Performance Tracking

Monitor response quality improvements:

```python
# Track before/after metrics
performance_tracking = {
    'response_relevance_score': 0.0,
    'user_satisfaction': 0.0,
    'query_processing_time': 0.0,
    'strategy_effectiveness': {}
}
```

## Rollout Plan

### Phase 1: Development (Week 1)
- Day 1-2: Implement QueryClassifier
- Day 3-4: Implement StrategyRouter
- Day 5: Integration with GraphTraversalAgent

### Phase 2: Testing (Week 1)
- Day 6: Unit tests and basic integration tests
- Day 7: End-to-end testing and bug fixes

### Phase 3: Deployment
- Deploy with feature flag for gradual rollout
- Monitor classification accuracy and performance
- Collect user feedback on response quality

## Success Metrics

- **Classification Accuracy**: >80% correct classification on test queries
- **Response Relevance**: 25-30% improvement in user satisfaction scores
- **Performance**: Factual queries 40% faster, summary queries handle 50% more context
- **Adoption**: 90% of queries successfully classified and routed

## Future Enhancements

This quick win lays the foundation for:
1. **LLM-based Classification**: Replace heuristics with fine-tuned models
2. **Adaptive Learning**: Learn from user feedback to improve classification
3. **Complex Query Handling**: Support for multi-part and hybrid queries
4. **Domain-Specific Patterns**: Specialized classification for different domains

## Dependencies

- No new external dependencies required
- Uses existing MoRAG components
- Compatible with current API structure

## Risk Mitigation

- **Fallback Strategy**: Always fall back to existing analytical approach
- **Feature Flag**: Can be disabled if issues arise
- **Gradual Rollout**: Test with subset of queries first
- **Monitoring**: Comprehensive metrics to detect regressions
