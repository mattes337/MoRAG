# Quick Win 5: Improved Chunk-Entity Association

## Overview

**Priority**: 📋 **Planned** (1 week, Medium Impact, Medium ROI)
**Source**: KG2RAG chunk-KG association patterns
**Expected Impact**: Better retrieval precision, more reliable entity-based queries

## Problem Statement

MoRAG currently has basic entity-chunk linking but lacks:
- Confidence scores for entity extractions
- Context snippets showing where entities were found
- Entity frequency tracking across chunks for importance weighting
- Relationship context between entities within chunks
- Quality metrics for entity extraction accuracy

This limits the precision of entity-based retrieval and makes it difficult to assess extraction quality.

## Solution Overview

Enhance the chunk-entity association system to store confidence scores, context snippets, frequency information, and relationship context, enabling more precise retrieval and better quality assessment.

## Technical Implementation

### 1. Enhanced Entity Association Model

Create `packages/morag-graph/src/morag_graph/entity_association.py`:

```python
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
from collections import defaultdict

@dataclass
class EntityMention:
    entity_name: str
    start_position: int
    end_position: int
    context_snippet: str
    confidence: float
    extraction_method: str
    normalized_form: str

@dataclass
class EntityAssociation:
    chunk_id: str
    entity_name: str
    mentions: List[EntityMention]
    frequency: int
    importance_score: float
    confidence_avg: float
    confidence_min: float
    confidence_max: float
    context_snippets: List[str]
    co_occurring_entities: Set[str]
    extraction_metadata: Dict[str, any] = field(default_factory=dict)

@dataclass
class ChunkEntityProfile:
    chunk_id: str
    total_entities: int
    unique_entities: int
    entity_density: float  # entities per 100 words
    extraction_quality_score: float
    dominant_entities: List[str]  # Top entities by importance
    entity_relationships: List[Tuple[str, str, str]]  # (entity1, relation, entity2)
    extraction_timestamp: datetime

class EntityAssociationManager:
    def __init__(self):
        self.confidence_threshold = 0.5
        self.context_window = 50  # characters around entity mention

    def create_entity_associations(self,
                                 chunk_id: str,
                                 chunk_text: str,
                                 extracted_entities: List[Dict[str, any]]) -> List[EntityAssociation]:
        """Create enhanced entity associations for a chunk."""

        associations = []
        entity_mentions = defaultdict(list)

        # Group mentions by entity
        for entity_data in extracted_entities:
            entity_name = entity_data['name']
            normalized_name = entity_data.get('normalized_name', entity_name)

            # Find all mentions of this entity in the text
            mentions = self._find_entity_mentions(
                chunk_text,
                entity_name,
                entity_data.get('confidence', 0.5),
                entity_data.get('extraction_method', 'unknown'),
                normalized_name
            )

            entity_mentions[normalized_name].extend(mentions)

        # Create associations
        for entity_name, mentions in entity_mentions.items():
            if not mentions:
                continue

            # Calculate statistics
            frequency = len(mentions)
            confidences = [m.confidence for m in mentions]
            confidence_avg = sum(confidences) / len(confidences)
            confidence_min = min(confidences)
            confidence_max = max(confidences)

            # Calculate importance score
            importance_score = self._calculate_importance_score(
                mentions, chunk_text, frequency
            )

            # Get context snippets
            context_snippets = [m.context_snippet for m in mentions]

            # Find co-occurring entities
            co_occurring = self._find_co_occurring_entities(
                entity_name, entity_mentions, chunk_text
            )

            association = EntityAssociation(
                chunk_id=chunk_id,
                entity_name=entity_name,
                mentions=mentions,
                frequency=frequency,
                importance_score=importance_score,
                confidence_avg=confidence_avg,
                confidence_min=confidence_min,
                confidence_max=confidence_max,
                context_snippets=context_snippets,
                co_occurring_entities=co_occurring,
                extraction_metadata={
                    'chunk_length': len(chunk_text),
                    'entity_positions': [(m.start_position, m.end_position) for m in mentions]
                }
            )

            associations.append(association)

        return associations

    def _find_entity_mentions(self,
                            text: str,
                            entity_name: str,
                            confidence: float,
                            extraction_method: str,
                            normalized_name: str) -> List[EntityMention]:
        """Find all mentions of an entity in text."""
        mentions = []

        # Create pattern for entity (case-insensitive, word boundaries)
        pattern = rf'\b{re.escape(entity_name)}\b'

        for match in re.finditer(pattern, text, re.IGNORECASE):
            start_pos = match.start()
            end_pos = match.end()

            # Extract context snippet
            context_start = max(0, start_pos - self.context_window)
            context_end = min(len(text), end_pos + self.context_window)
            context_snippet = text[context_start:context_end].strip()

            mention = EntityMention(
                entity_name=entity_name,
                start_position=start_pos,
                end_position=end_pos,
                context_snippet=context_snippet,
                confidence=confidence,
                extraction_method=extraction_method,
                normalized_form=normalized_name
            )

            mentions.append(mention)

        return mentions

    def _calculate_importance_score(self,
                                  mentions: List[EntityMention],
                                  chunk_text: str,
                                  frequency: int) -> float:
        """Calculate importance score for entity in chunk."""

        # Base score from frequency
        chunk_words = len(chunk_text.split())
        frequency_score = min(frequency / max(chunk_words / 100, 1), 1.0)

        # Confidence score
        avg_confidence = sum(m.confidence for m in mentions) / len(mentions)

        # Position score (entities mentioned early are often more important)
        position_scores = []
        for mention in mentions:
            relative_position = mention.start_position / len(chunk_text)
            # Higher score for earlier positions
            position_score = 1.0 - (relative_position * 0.5)
            position_scores.append(position_score)

        avg_position_score = sum(position_scores) / len(position_scores)

        # Combined importance score
        importance = (
            0.4 * frequency_score +
            0.4 * avg_confidence +
            0.2 * avg_position_score
        )

        return min(importance, 1.0)

    def _find_co_occurring_entities(self,
                                  target_entity: str,
                                  all_entity_mentions: Dict[str, List[EntityMention]],
                                  chunk_text: str) -> Set[str]:
        """Find entities that co-occur with target entity."""
        co_occurring = set()

        target_mentions = all_entity_mentions.get(target_entity, [])

        for target_mention in target_mentions:
            # Define co-occurrence window around target mention
            window_start = max(0, target_mention.start_position - 200)
            window_end = min(len(chunk_text), target_mention.end_position + 200)

            # Check other entities in this window
            for other_entity, other_mentions in all_entity_mentions.items():
                if other_entity == target_entity:
                    continue

                for other_mention in other_mentions:
                    if (window_start <= other_mention.start_position <= window_end or
                        window_start <= other_mention.end_position <= window_end):
                        co_occurring.add(other_entity)

        return co_occurring

    def create_chunk_profile(self,
                           chunk_id: str,
                           chunk_text: str,
                           associations: List[EntityAssociation]) -> ChunkEntityProfile:
        """Create comprehensive profile for chunk's entity content."""

        if not associations:
            return ChunkEntityProfile(
                chunk_id=chunk_id,
                total_entities=0,
                unique_entities=0,
                entity_density=0.0,
                extraction_quality_score=0.0,
                dominant_entities=[],
                entity_relationships=[],
                extraction_timestamp=datetime.now()
            )

        # Calculate statistics
        total_entities = sum(assoc.frequency for assoc in associations)
        unique_entities = len(associations)

        chunk_words = len(chunk_text.split())
        entity_density = (total_entities / chunk_words) * 100 if chunk_words > 0 else 0

        # Calculate extraction quality score
        avg_confidence = sum(assoc.confidence_avg for assoc in associations) / len(associations)
        confidence_variance = self._calculate_confidence_variance(associations)
        quality_score = avg_confidence * (1 - confidence_variance)  # Penalize high variance

        # Find dominant entities
        dominant_entities = sorted(
            associations,
            key=lambda a: a.importance_score,
            reverse=True
        )[:5]
        dominant_entity_names = [assoc.entity_name for assoc in dominant_entities]

        # Extract entity relationships (simple co-occurrence based)
        entity_relationships = self._extract_entity_relationships(associations, chunk_text)

        return ChunkEntityProfile(
            chunk_id=chunk_id,
            total_entities=total_entities,
            unique_entities=unique_entities,
            entity_density=entity_density,
            extraction_quality_score=quality_score,
            dominant_entities=dominant_entity_names,
            entity_relationships=entity_relationships,
            extraction_timestamp=datetime.now()
        )

    def _calculate_confidence_variance(self, associations: List[EntityAssociation]) -> float:
        """Calculate variance in confidence scores."""
        if len(associations) <= 1:
            return 0.0

        confidences = [assoc.confidence_avg for assoc in associations]
        mean_confidence = sum(confidences) / len(confidences)

        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        return min(variance, 1.0)  # Normalize to 0-1

    def _extract_entity_relationships(self,
                                    associations: List[EntityAssociation],
                                    chunk_text: str) -> List[Tuple[str, str, str]]:
        """Extract simple entity relationships from co-occurrence patterns."""
        relationships = []

        # Simple pattern-based relationship extraction
        relationship_patterns = [
            (r'(\w+)\s+(?:is|was)\s+(?:a|an|the)?\s*(\w+)', 'is_a'),
            (r'(\w+)\s+(?:works for|employed by)\s+(\w+)', 'works_for'),
            (r'(\w+)\s+(?:founded|created|established)\s+(\w+)', 'founded'),
            (r'(\w+)\s+(?:owns|acquired)\s+(\w+)', 'owns'),
            (r'(\w+)\s+(?:and|with)\s+(\w+)', 'associated_with')
        ]

        entity_names = {assoc.entity_name for assoc in associations}

        for pattern, relation_type in relationship_patterns:
            for match in re.finditer(pattern, chunk_text, re.IGNORECASE):
                entity1, entity2 = match.groups()

                # Check if both entities are in our extracted entities
                if entity1 in entity_names and entity2 in entity_names:
                    relationships.append((entity1, relation_type, entity2))

        return relationships[:10]  # Limit to top 10 relationships

    def filter_associations_by_quality(self,
                                     associations: List[EntityAssociation],
                                     min_confidence: float = 0.5,
                                     min_importance: float = 0.3) -> List[EntityAssociation]:
        """Filter associations by quality thresholds."""
        return [
            assoc for assoc in associations
            if assoc.confidence_avg >= min_confidence and assoc.importance_score >= min_importance
        ]

    def get_top_entities_by_importance(self,
                                     associations: List[EntityAssociation],
                                     top_k: int = 10) -> List[EntityAssociation]:
        """Get top entities by importance score."""
        return sorted(
            associations,
            key=lambda a: a.importance_score,
            reverse=True
        )[:top_k]
```

### 2. Enhanced Storage Schema

Update Neo4j schema to store enhanced associations:

```cypher
// Enhanced entity-chunk relationship
CREATE CONSTRAINT entity_chunk_association IF NOT EXISTS
FOR ()-[r:MENTIONED_IN]-() REQUIRE r.chunk_id IS NOT NULL;

// Store enhanced association data
MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk)
SET r.frequency = $frequency,
    r.importance_score = $importance_score,
    r.confidence_avg = $confidence_avg,
    r.confidence_min = $confidence_min,
    r.confidence_max = $confidence_max,
    r.context_snippets = $context_snippets,
    r.co_occurring_entities = $co_occurring_entities,
    r.extraction_metadata = $extraction_metadata;

// Add chunk entity profile
MATCH (c:Chunk)
SET c.total_entities = $total_entities,
    c.unique_entities = $unique_entities,
    c.entity_density = $entity_density,
    c.extraction_quality_score = $extraction_quality_score,
    c.dominant_entities = $dominant_entities;
```

### 3. Integration with Extraction Pipeline

Update `packages/morag-graph/src/morag_graph/entity_extractor.py`:

```python
from .entity_association import EntityAssociationManager

class EntityExtractor:
    def __init__(self):
        # ... existing initialization
        self.association_manager = EntityAssociationManager()

    async def extract_and_associate_entities(self,
                                           chunk_id: str,
                                           chunk_text: str) -> Dict[str, any]:
        """Extract entities and create enhanced associations."""

        # Extract entities using existing methods
        extracted_entities = await self._extract_entities_raw(chunk_text)

        # Create enhanced associations
        associations = self.association_manager.create_entity_associations(
            chunk_id, chunk_text, extracted_entities
        )

        # Create chunk profile
        chunk_profile = self.association_manager.create_chunk_profile(
            chunk_id, chunk_text, associations
        )

        # Filter by quality
        high_quality_associations = self.association_manager.filter_associations_by_quality(
            associations, min_confidence=0.6, min_importance=0.4
        )

        return {
            'all_associations': associations,
            'high_quality_associations': high_quality_associations,
            'chunk_profile': chunk_profile,
            'extraction_stats': {
                'total_entities': len(associations),
                'high_quality_entities': len(high_quality_associations),
                'avg_confidence': chunk_profile.extraction_quality_score,
                'entity_density': chunk_profile.entity_density
            }
        }
```

### 4. Enhanced Retrieval with Association Data

Update retrieval to use association information:

```python
# packages/morag-reasoning/src/morag_reasoning/entity_retrieval.py

class EntityBasedRetriever:
    def __init__(self, neo4j_service, association_manager):
        self.neo4j = neo4j_service
        self.association_manager = association_manager

    async def retrieve_by_entity_importance(self,
                                          entities: List[str],
                                          collection: str,
                                          top_k: int = 10) -> List[Dict[str, any]]:
        """Retrieve chunks based on entity importance scores."""

        query = """
        MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk)
        WHERE e.name IN $entities
          AND c.collection_name = $collection
        RETURN c.id as chunk_id,
               c.content as content,
               e.name as entity,
               r.importance_score as importance,
               r.confidence_avg as confidence,
               r.frequency as frequency,
               r.context_snippets as context_snippets
        ORDER BY r.importance_score DESC, r.confidence_avg DESC
        LIMIT $top_k
        """

        results = await self.neo4j.execute_query(query, {
            'entities': entities,
            'collection': collection,
            'top_k': top_k
        })

        return results

    async def retrieve_by_entity_co_occurrence(self,
                                             entities: List[str],
                                             collection: str) -> List[Dict[str, any]]:
        """Retrieve chunks where multiple entities co-occur."""

        query = """
        MATCH (e1:Entity)-[r1:MENTIONED_IN]->(c:Chunk)<-[r2:MENTIONED_IN]-(e2:Entity)
        WHERE e1.name IN $entities
          AND e2.name IN $entities
          AND e1 <> e2
          AND c.collection_name = $collection
        RETURN c.id as chunk_id,
               c.content as content,
               collect(DISTINCT e1.name) + collect(DISTINCT e2.name) as entities,
               avg(r1.importance_score + r2.importance_score) as combined_importance,
               avg(r1.confidence_avg + r2.confidence_avg) as combined_confidence
        ORDER BY combined_importance DESC
        """

        results = await self.neo4j.execute_query(query, {
            'entities': entities,
            'collection': collection
        })

        return results
```

## Configuration

```yaml
# entity_association.yml
entity_association:
  enabled: true

  confidence_thresholds:
    minimum: 0.5
    high_quality: 0.7

  importance_scoring:
    frequency_weight: 0.4
    confidence_weight: 0.4
    position_weight: 0.2

  context_extraction:
    window_size: 50  # characters around mention
    max_snippets_per_entity: 5

  co_occurrence:
    window_size: 200  # characters for co-occurrence detection

  quality_filtering:
    min_confidence: 0.6
    min_importance: 0.4
    max_entities_per_chunk: 50

  relationship_extraction:
    enabled: true
    max_relationships_per_chunk: 10
    pattern_based: true
```

## Testing Strategy

```python
# tests/unit/test_entity_association.py
import pytest
from morag_graph.entity_association import EntityAssociationManager

class TestEntityAssociation:
    def setup_method(self):
        self.manager = EntityAssociationManager()

    def test_entity_mention_finding(self):
        text = "Tesla was founded by Elon Musk. Tesla is an electric vehicle company."
        mentions = self.manager._find_entity_mentions(
            text, "Tesla", 0.9, "spacy", "Tesla"
        )
        assert len(mentions) == 2
        assert all(m.entity_name == "Tesla" for m in mentions)

    def test_importance_scoring(self):
        # Test importance score calculation
        pass

    def test_co_occurrence_detection(self):
        # Test entity co-occurrence detection
        pass

    def test_chunk_profile_creation(self):
        # Test chunk profile generation
        pass
```

## Monitoring

```python
association_metrics = {
    'extraction_quality': {
        'avg_confidence': 0.0,
        'confidence_variance': 0.0,
        'high_quality_ratio': 0.0
    },
    'entity_coverage': {
        'avg_entities_per_chunk': 0.0,
        'avg_entity_density': 0.0,
        'chunks_with_entities': 0.0
    },
    'association_quality': {
        'avg_importance_score': 0.0,
        'co_occurrence_rate': 0.0,
        'context_snippet_coverage': 0.0
    }
}
```

## Success Metrics

- **Retrieval Precision**: 20-25% improvement in entity-based retrieval accuracy
- **Extraction Quality**: >80% of entities with confidence >0.7
- **Context Coverage**: 95% of entity mentions have context snippets
- **Co-occurrence Detection**: 60% of related entities properly identified

## Future Enhancements

1. **ML-based Importance Scoring**: Learn importance patterns from user feedback
2. **Advanced Relationship Extraction**: Use dependency parsing for better relationships
3. **Cross-Document Entity Linking**: Link entities across document boundaries
4. **Temporal Entity Tracking**: Track entity mentions over time
