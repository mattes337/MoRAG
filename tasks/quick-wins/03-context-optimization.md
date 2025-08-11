# Quick Win 3: Context Window Optimization

## Overview

**Priority**: ⚡ **Next Sprint** (1-2 weeks, Medium Impact, High ROI)  
**Source**: KG2RAG context organization insights  
**Expected Impact**: 15-20% improvement in response quality, better token efficiency

## Problem Statement

MoRAG currently uses simple concatenation of retrieved chunks without considering:
- Relevance-based prioritization of chunks
- Redundant information across chunks
- Relationship indicators between chunks
- Optimal context organization for LLM processing
- Token budget optimization

This leads to suboptimal context windows that may include irrelevant information while missing important connections.

## Solution Overview

Implement intelligent context organization that prioritizes chunks by relevance, removes redundancy, adds relationship indicators, and optimizes for token efficiency while preserving semantic coherence.

## Technical Implementation

### 1. Context Optimizer

Create `packages/morag-reasoning/src/morag_reasoning/context_optimizer.py`:

```python
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import re
from collections import defaultdict
from difflib import SequenceMatcher

@dataclass
class ContextChunk:
    id: str
    content: str
    relevance_score: float
    source_document: str
    entities: List[str]
    relationships: List[str]
    timestamp: Optional[str] = None
    chunk_type: str = "text"

@dataclass
class ChunkRelationship:
    chunk1_id: str
    chunk2_id: str
    relationship_type: str
    strength: float
    description: str

@dataclass
class OptimizedContext:
    chunks: List[ContextChunk]
    relationships: List[ChunkRelationship]
    total_tokens: int
    organization_strategy: str
    removed_chunks: List[str]
    deduplication_stats: Dict[str, int]

class ContextOptimizer:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.similarity_threshold = 0.7
        self.min_chunk_relevance = 0.3
        
    async def optimize_context(self, 
                             chunks: List[ContextChunk], 
                             query: str,
                             strategy: str = "auto") -> OptimizedContext:
        """Optimize context for LLM processing."""
        
        # 1. Filter by minimum relevance
        filtered_chunks = [c for c in chunks if c.relevance_score >= self.min_chunk_relevance]
        
        # 2. Remove redundant chunks
        deduplicated_chunks, dedup_stats = await self._remove_redundancy(filtered_chunks)
        
        # 3. Identify relationships between chunks
        relationships = await self._identify_relationships(deduplicated_chunks)
        
        # 4. Prioritize chunks based on relevance and relationships
        prioritized_chunks = await self._prioritize_chunks(deduplicated_chunks, relationships, query)
        
        # 5. Fit within token budget
        final_chunks, removed = await self._fit_token_budget(prioritized_chunks)
        
        # 6. Organize chunks optimally
        organized_chunks = await self._organize_chunks(final_chunks, relationships, strategy)
        
        total_tokens = sum(self._estimate_tokens(chunk.content) for chunk in organized_chunks)
        
        return OptimizedContext(
            chunks=organized_chunks,
            relationships=relationships,
            total_tokens=total_tokens,
            organization_strategy=strategy,
            removed_chunks=removed,
            deduplication_stats=dedup_stats
        )

    async def _remove_redundancy(self, chunks: List[ContextChunk]) -> Tuple[List[ContextChunk], Dict[str, int]]:
        """Remove redundant chunks based on content similarity."""
        unique_chunks = []
        removed_count = 0
        similarity_groups = defaultdict(list)
        
        # Group similar chunks
        for i, chunk in enumerate(chunks):
            assigned = False
            for group_id, group in similarity_groups.items():
                if self._chunks_similar(chunk, group[0], self.similarity_threshold):
                    group.append(chunk)
                    assigned = True
                    break
            
            if not assigned:
                similarity_groups[i] = [chunk]
        
        # Keep best chunk from each group
        for group in similarity_groups.values():
            if len(group) == 1:
                unique_chunks.append(group[0])
            else:
                # Keep chunk with highest relevance score
                best_chunk = max(group, key=lambda c: c.relevance_score)
                unique_chunks.append(best_chunk)
                removed_count += len(group) - 1
        
        stats = {
            'original_count': len(chunks),
            'unique_count': len(unique_chunks),
            'removed_count': removed_count,
            'similarity_groups': len(similarity_groups)
        }
        
        return unique_chunks, stats

    def _chunks_similar(self, chunk1: ContextChunk, chunk2: ContextChunk, threshold: float) -> bool:
        """Check if two chunks are similar enough to be considered redundant."""
        # Content similarity
        content_sim = SequenceMatcher(None, chunk1.content, chunk2.content).ratio()
        
        # Entity overlap
        entities1 = set(chunk1.entities)
        entities2 = set(chunk2.entities)
        entity_overlap = len(entities1 & entities2) / max(len(entities1 | entities2), 1)
        
        # Combined similarity
        combined_sim = 0.7 * content_sim + 0.3 * entity_overlap
        
        return combined_sim >= threshold

    async def _identify_relationships(self, chunks: List[ContextChunk]) -> List[ChunkRelationship]:
        """Identify relationships between chunks."""
        relationships = []
        
        for i, chunk1 in enumerate(chunks):
            for chunk2 in chunks[i+1:]:
                relationship = await self._analyze_chunk_relationship(chunk1, chunk2)
                if relationship:
                    relationships.append(relationship)
        
        return relationships

    async def _analyze_chunk_relationship(self, chunk1: ContextChunk, chunk2: ContextChunk) -> Optional[ChunkRelationship]:
        """Analyze relationship between two chunks."""
        # Entity overlap
        entities1 = set(chunk1.entities)
        entities2 = set(chunk2.entities)
        shared_entities = entities1 & entities2
        
        if not shared_entities:
            return None
        
        # Determine relationship type and strength
        overlap_ratio = len(shared_entities) / len(entities1 | entities2)
        
        if overlap_ratio > 0.5:
            rel_type = "strong_overlap"
            strength = overlap_ratio
        elif overlap_ratio > 0.2:
            rel_type = "moderate_overlap"
            strength = overlap_ratio
        else:
            rel_type = "weak_overlap"
            strength = overlap_ratio
        
        # Check for temporal relationships
        if chunk1.timestamp and chunk2.timestamp:
            if chunk1.timestamp < chunk2.timestamp:
                rel_type = "temporal_sequence"
                strength = max(strength, 0.6)
        
        # Check for same document
        if chunk1.source_document == chunk2.source_document:
            rel_type = "same_document"
            strength = max(strength, 0.4)
        
        description = f"Shared entities: {', '.join(list(shared_entities)[:3])}"
        
        return ChunkRelationship(
            chunk1_id=chunk1.id,
            chunk2_id=chunk2.id,
            relationship_type=rel_type,
            strength=strength,
            description=description
        )

    async def _prioritize_chunks(self, chunks: List[ContextChunk], 
                               relationships: List[ChunkRelationship], 
                               query: str) -> List[ContextChunk]:
        """Prioritize chunks based on relevance and relationships."""
        
        # Calculate relationship scores for each chunk
        relationship_scores = defaultdict(float)
        for rel in relationships:
            relationship_scores[rel.chunk1_id] += rel.strength
            relationship_scores[rel.chunk2_id] += rel.strength
        
        # Calculate query relevance boost
        query_terms = set(query.lower().split())
        
        # Calculate final priority scores
        for chunk in chunks:
            base_score = chunk.relevance_score
            relationship_boost = relationship_scores.get(chunk.id, 0) * 0.2
            
            # Query term overlap boost
            chunk_terms = set(chunk.content.lower().split())
            query_overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            query_boost = query_overlap * 0.3
            
            # Recency boost (if timestamp available)
            recency_boost = 0
            if chunk.timestamp:
                # Simple recency boost - more sophisticated logic could be added
                recency_boost = 0.1
            
            chunk.relevance_score = base_score + relationship_boost + query_boost + recency_boost
        
        # Sort by priority score
        return sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

    async def _fit_token_budget(self, chunks: List[ContextChunk]) -> Tuple[List[ContextChunk], List[str]]:
        """Fit chunks within token budget."""
        selected_chunks = []
        removed_chunks = []
        current_tokens = 0
        
        # Reserve tokens for system prompt and response
        available_tokens = self.max_tokens - 500
        
        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk.content)
            
            if current_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                removed_chunks.append(chunk.id)
        
        return selected_chunks, removed_chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    async def _organize_chunks(self, chunks: List[ContextChunk], 
                             relationships: List[ChunkRelationship], 
                             strategy: str) -> List[ContextChunk]:
        """Organize chunks for optimal LLM processing."""
        
        if strategy == "auto":
            strategy = self._determine_best_strategy(chunks, relationships)
        
        if strategy == "relevance":
            return chunks  # Already sorted by relevance
        elif strategy == "temporal":
            return self._organize_temporally(chunks)
        elif strategy == "topical":
            return self._organize_topically(chunks, relationships)
        elif strategy == "hierarchical":
            return self._organize_hierarchically(chunks, relationships)
        else:
            return chunks

    def _determine_best_strategy(self, chunks: List[ContextChunk], 
                               relationships: List[ChunkRelationship]) -> str:
        """Determine the best organization strategy."""
        
        # Check if temporal information is available
        temporal_chunks = sum(1 for c in chunks if c.timestamp)
        if temporal_chunks > len(chunks) * 0.5:
            return "temporal"
        
        # Check relationship density
        relationship_density = len(relationships) / max(len(chunks) * (len(chunks) - 1) / 2, 1)
        if relationship_density > 0.3:
            return "topical"
        
        # Default to relevance-based
        return "relevance"

    def _organize_temporally(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Organize chunks by timestamp."""
        timestamped = [c for c in chunks if c.timestamp]
        non_timestamped = [c for c in chunks if not c.timestamp]
        
        timestamped.sort(key=lambda c: c.timestamp)
        
        return timestamped + non_timestamped

    def _organize_topically(self, chunks: List[ContextChunk], 
                          relationships: List[ChunkRelationship]) -> List[ContextChunk]:
        """Organize chunks by topic clusters."""
        # Build adjacency graph
        adjacency = defaultdict(list)
        for rel in relationships:
            if rel.strength > 0.3:  # Only strong relationships
                adjacency[rel.chunk1_id].append(rel.chunk2_id)
                adjacency[rel.chunk2_id].append(rel.chunk1_id)
        
        # Find connected components (topic clusters)
        visited = set()
        clusters = []
        
        chunk_map = {c.id: c for c in chunks}
        
        for chunk in chunks:
            if chunk.id not in visited:
                cluster = self._dfs_cluster(chunk.id, adjacency, visited)
                clusters.append([chunk_map[cid] for cid in cluster])
        
        # Sort clusters by average relevance
        clusters.sort(key=lambda cluster: sum(c.relevance_score for c in cluster) / len(cluster), reverse=True)
        
        # Flatten clusters
        organized = []
        for cluster in clusters:
            cluster.sort(key=lambda c: c.relevance_score, reverse=True)
            organized.extend(cluster)
        
        return organized

    def _dfs_cluster(self, chunk_id: str, adjacency: Dict[str, List[str]], visited: Set[str]) -> List[str]:
        """DFS to find connected component."""
        if chunk_id in visited:
            return []
        
        visited.add(chunk_id)
        cluster = [chunk_id]
        
        for neighbor in adjacency.get(chunk_id, []):
            cluster.extend(self._dfs_cluster(neighbor, adjacency, visited))
        
        return cluster

    def _organize_hierarchically(self, chunks: List[ContextChunk], 
                               relationships: List[ChunkRelationship]) -> List[ContextChunk]:
        """Organize chunks hierarchically (overview first, then details)."""
        # Classify chunks by type
        overview_chunks = []
        detail_chunks = []
        
        for chunk in chunks:
            if self._is_overview_chunk(chunk):
                overview_chunks.append(chunk)
            else:
                detail_chunks.append(chunk)
        
        # Sort each category by relevance
        overview_chunks.sort(key=lambda c: c.relevance_score, reverse=True)
        detail_chunks.sort(key=lambda c: c.relevance_score, reverse=True)
        
        return overview_chunks + detail_chunks

    def _is_overview_chunk(self, chunk: ContextChunk) -> bool:
        """Determine if chunk contains overview/summary information using heuristics."""
        content = chunk.content

        # Use structural and positional heuristics instead of hardcoded terms
        # Check if chunk is at beginning of document (likely introduction/overview)
        if hasattr(chunk, 'position') and chunk.position < 0.2:  # First 20% of document
            return True

        # Check for structural indicators (headings, bullet points, lists)
        if content.count('\n') > 5 and ('•' in content or '-' in content or content.count(':') > 2):
            return True

        # Check for high entity density (overview chunks often mention many entities)
        if len(chunk.entities) > len(content.split()) * 0.1:  # >10% entity density
            return True

        # Check for shorter chunks (summaries are often concise)
        if len(content.split()) < 100 and len(chunk.entities) > 3:
            return True

        return False

    def format_context(self, optimized_context: OptimizedContext, query: str) -> str:
        """Format optimized context for LLM consumption."""
        formatted_parts = []
        
        # Add context header
        formatted_parts.append(f"# Context for Query: {query}\n")
        formatted_parts.append(f"Organization Strategy: {optimized_context.organization_strategy}")
        formatted_parts.append(f"Total Chunks: {len(optimized_context.chunks)}")
        formatted_parts.append(f"Estimated Tokens: {optimized_context.total_tokens}\n")
        
        # Add chunks with relationship indicators
        for i, chunk in enumerate(optimized_context.chunks):
            # Add chunk header
            formatted_parts.append(f"## Source {i+1}: {chunk.source_document}")
            formatted_parts.append(f"Relevance: {chunk.relevance_score:.2f}")
            
            # Add relationship indicators
            related_chunks = self._find_related_chunks(chunk, optimized_context.relationships)
            if related_chunks:
                formatted_parts.append(f"Related to: {', '.join(related_chunks)}")
            
            # Add chunk content
            formatted_parts.append(f"\n{chunk.content}\n")
            
            # Add separator
            if i < len(optimized_context.chunks) - 1:
                formatted_parts.append("---\n")
        
        return "\n".join(formatted_parts)

    def _find_related_chunks(self, chunk: ContextChunk, 
                           relationships: List[ChunkRelationship]) -> List[str]:
        """Find chunks related to the given chunk."""
        related = []
        
        for rel in relationships:
            if rel.chunk1_id == chunk.id:
                related.append(f"Source {rel.chunk2_id} ({rel.relationship_type})")
            elif rel.chunk2_id == chunk.id:
                related.append(f"Source {rel.chunk1_id} ({rel.relationship_type})")
        
        return related[:3]  # Limit to top 3 relationships
```

### 2. Integration with Retrieval Pipeline

Update `packages/morag-reasoning/src/morag_reasoning/graph_traversal_agent.py`:

```python
from .context_optimizer import ContextOptimizer, ContextChunk

class GraphTraversalAgent:
    def __init__(self, ...):
        # ... existing initialization
        self.context_optimizer = ContextOptimizer()
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced query processing with context optimization."""
        
        # ... existing retrieval logic
        
        # Convert retrieved chunks to ContextChunk objects
        context_chunks = []
        for chunk_data in retrieved_chunks:
            context_chunk = ContextChunk(
                id=chunk_data['id'],
                content=chunk_data['content'],
                relevance_score=chunk_data.get('score', 0.5),
                source_document=chunk_data.get('source', 'unknown'),
                entities=chunk_data.get('entities', []),
                relationships=chunk_data.get('relationships', []),
                timestamp=chunk_data.get('timestamp'),
                chunk_type=chunk_data.get('type', 'text')
            )
            context_chunks.append(context_chunk)
        
        # Optimize context
        max_tokens = context.get('max_context_tokens', 4000)
        self.context_optimizer.max_tokens = max_tokens
        
        optimized_context = await self.context_optimizer.optimize_context(
            context_chunks, 
            query,
            strategy=context.get('organization_strategy', 'auto')
        )
        
        # Format for LLM
        formatted_context = self.context_optimizer.format_context(optimized_context, query)
        
        # ... continue with LLM processing
        
        return {
            'answer': answer,
            'sources': sources,
            'context_optimization': {
                'original_chunks': len(context_chunks),
                'optimized_chunks': len(optimized_context.chunks),
                'total_tokens': optimized_context.total_tokens,
                'organization_strategy': optimized_context.organization_strategy,
                'deduplication_stats': optimized_context.deduplication_stats
            }
        }
```

## Configuration

Add context optimization settings:

```yaml
# context_optimization.yml
context_optimization:
  enabled: true
  max_tokens: 4000
  similarity_threshold: 0.7
  min_chunk_relevance: 0.3
  
  organization_strategies:
    auto: true
    relevance: true
    temporal: true
    topical: true
    hierarchical: true
    
  token_allocation:
    system_prompt: 200
    response_buffer: 300
    context_content: 3500
    
  deduplication:
    enabled: true
    content_similarity_threshold: 0.7
    entity_overlap_weight: 0.3
    
  relationship_detection:
    enabled: true
    min_entity_overlap: 0.2
    temporal_boost: 0.6
    same_document_boost: 0.4
```

## Testing Strategy

### 1. Unit Tests

```python
# tests/unit/test_context_optimization.py
import pytest
from morag_reasoning.context_optimizer import ContextOptimizer, ContextChunk

class TestContextOptimization:
    def setup_method(self):
        self.optimizer = ContextOptimizer(max_tokens=1000)

    def test_redundancy_removal(self):
        # Test with similar chunks
        chunks = [
            ContextChunk("1", "AI is transforming healthcare", 0.9, "doc1", ["AI", "healthcare"], []),
            ContextChunk("2", "Artificial intelligence is changing healthcare", 0.8, "doc2", ["AI", "healthcare"], []),
            ContextChunk("3", "Machine learning in finance", 0.7, "doc3", ["ML", "finance"], [])
        ]
        
        unique_chunks, stats = await self.optimizer._remove_redundancy(chunks)
        assert len(unique_chunks) == 2  # First two should be merged
        assert stats['removed_count'] == 1

    def test_token_budget_fitting(self):
        # Test token budget constraints
        pass

    def test_relationship_identification(self):
        # Test chunk relationship detection
        pass
```

### 2. Integration Tests

```python
# tests/integration/test_context_optimization_integration.py
class TestContextOptimizationIntegration:
    @pytest.mark.asyncio
    async def test_full_optimization_pipeline(self):
        # Test complete optimization workflow
        pass
```

## Monitoring

Track context optimization effectiveness:

```python
context_metrics = {
    'chunks_processed': 0,
    'chunks_removed_redundancy': 0,
    'chunks_removed_budget': 0,
    'average_relevance_score': 0.0,
    'token_utilization': 0.0,
    'organization_strategies_used': {
        'relevance': 0,
        'temporal': 0,
        'topical': 0,
        'hierarchical': 0
    },
    'relationship_detection_rate': 0.0
}
```

## Success Metrics

- **Response Quality**: 15-20% improvement in relevance and coherence
- **Token Efficiency**: 25-30% better token utilization
- **Redundancy Reduction**: 40-50% reduction in duplicate information
- **Processing Speed**: Faster LLM processing due to better context organization

## Future Enhancements

1. **ML-based Optimization**: Learn optimal organization patterns
2. **Dynamic Token Allocation**: Adjust based on query complexity
3. **Multi-modal Context**: Handle images, tables, and other content types
4. **User Preference Learning**: Adapt to user feedback on context quality
