# Quick Win 2: Enhanced Entity Normalization

## Overview

**Priority**: 🔥 **Immediate** (1-2 weeks, High Impact, High ROI)  
**Source**: LightRAG systematic deduplication insights  
**Expected Impact**: 15-20% reduction in duplicate entities, cleaner knowledge graphs

## Problem Statement

MoRAG currently has basic LLM-based entity normalization, but lacks systematic post-processing to handle common entity variations. This leads to:
- Duplicate entities for the same concept (e.g., "AI", "A.I.", "artificial intelligence")
- Inconsistent person name formats ("John Smith" vs "Smith, John")
- Technical term variations ("ML" vs "Machine Learning")
- Organization name inconsistencies ("Google Inc." vs "Google" vs "Alphabet")

These duplicates fragment the knowledge graph, reducing retrieval effectiveness and query performance.

## Solution Overview

Implement a comprehensive entity normalization pipeline with post-processing rules, fuzzy matching, and domain-specific patterns to automatically detect and merge duplicate entities while preserving semantic distinctions.

## Technical Implementation

### 1. Enhanced Entity Normalizer

Create `packages/morag-graph/src/morag_graph/entity_normalizer.py`:

```python
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
import unicodedata
from collections import defaultdict

@dataclass
class EntityVariation:
    original: str
    normalized: str
    confidence: float
    rule_applied: str

@dataclass
class EntityMergeCandidate:
    entities: List[str]
    canonical_form: str
    confidence: float
    merge_reason: str

class EnhancedEntityNormalizer:
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        # Cache for LLM-based normalization results to avoid repeated calls
        self.normalization_cache = {}
        self.entity_type_cache = {}
        self.similarity_cache = {}

    async def normalize_entity(self, entity: str, language: str = None) -> EntityVariation:
        """Normalize a single entity using LLM-based analysis."""
        original = entity

        # Check cache first
        cache_key = f"{entity}:{language or 'auto'}"
        if cache_key in self.normalization_cache:
            cached_result = self.normalization_cache[cache_key]
            return EntityVariation(
                original=original,
                normalized=cached_result['normalized'],
                confidence=cached_result['confidence'],
                rule_applied=cached_result['rule_applied']
            )

        # Use LLM for normalization
        normalized, confidence, rule_applied = await self._llm_normalize_entity(entity, language)

        # Cache result
        self.normalization_cache[cache_key] = {
            'normalized': normalized,
            'confidence': confidence,
            'rule_applied': rule_applied
        }

        return EntityVariation(
            original=original,
            normalized=normalized,
            confidence=confidence,
            rule_applied=rule_applied
        )

    async def find_merge_candidates(self, entities: List[str], language: str = None) -> List[EntityMergeCandidate]:
        """Find entities that should be merged using LLM-based similarity analysis."""
        candidates = []

        # Use LLM to analyze entity similarities in batches
        batch_size = 20  # Process entities in batches to avoid token limits

        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batch_candidates = await self._llm_find_merge_candidates(batch, language)
            candidates.extend(batch_candidates)

        return candidates

    async def _llm_normalize_entity(self, entity: str, language: str = None) -> Tuple[str, float, str]:
        """Use LLM to normalize entity."""
        if not self.llm_service:
            # Fallback to basic normalization
            return entity.strip(), 0.5, "basic_cleanup"

        prompt = f"""
        Normalize the following entity name to its canonical form. Consider:
        - Remove unnecessary punctuation and formatting
        - Standardize capitalization appropriately
        - Expand common abbreviations if beneficial
        - Maintain the core meaning and identity

        Entity: "{entity}"
        Language context: {language or "auto-detect"}

        Respond with JSON:
        {{
            "normalized": "canonical form",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation of changes made"
        }}
        """

        try:
            response = await self.llm_service.generate(prompt, max_tokens=200)
            result = json.loads(response)

            return (
                result.get('normalized', entity.strip()),
                float(result.get('confidence', 0.5)),
                result.get('reasoning', 'llm_normalization')
            )
        except Exception as e:
            # Fallback on LLM failure
            return entity.strip(), 0.3, f"llm_error: {str(e)}"

    async def _llm_find_merge_candidates(self, entities: List[str], language: str = None) -> List[EntityMergeCandidate]:
        """Use LLM to find entities that should be merged."""
        if not self.llm_service or len(entities) < 2:
            return []

        entities_text = "\n".join([f"{i+1}. {entity}" for i, entity in enumerate(entities)])

        prompt = f"""
        Analyze the following entities and identify which ones refer to the same real-world entity and should be merged.
        Consider variations in:
        - Spelling and formatting
        - Abbreviations vs full forms
        - Different name formats (e.g., "John Smith" vs "Smith, John")
        - Language variations
        - Punctuation differences

        Entities:
        {entities_text}

        Language context: {language or "auto-detect"}

        Respond with JSON array of merge groups:
        [
            {{
                "entities": ["entity1", "entity2"],
                "canonical_form": "preferred form",
                "confidence": 0.0-1.0,
                "reason": "explanation"
            }}
        ]

        Only include groups with confidence > 0.7.
        """

        try:
            response = await self.llm_service.generate(prompt, max_tokens=1000)
            merge_groups = json.loads(response)

            candidates = []
            for group in merge_groups:
                if (isinstance(group, dict) and
                    'entities' in group and
                    len(group['entities']) >= 2 and
                    group.get('confidence', 0) > 0.7):

                    candidates.append(EntityMergeCandidate(
                        entities=group['entities'],
                        canonical_form=group.get('canonical_form', group['entities'][0]),
                        confidence=float(group.get('confidence', 0.8)),
                        merge_reason=group.get('reason', 'llm_similarity_analysis')
                    ))

            return candidates

        except Exception as e:
            # Fallback to simple string similarity
            return self._fallback_similarity_matching(entities)

    def _fallback_similarity_matching(self, entities: List[str]) -> List[EntityMergeCandidate]:
        """Fallback similarity matching when LLM is unavailable."""
        candidates = []

        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Simple string similarity
                similarity = SequenceMatcher(None, entity1.lower(), entity2.lower()).ratio()

                if similarity > 0.85:  # High similarity threshold
                    canonical = entity1 if len(entity1) >= len(entity2) else entity2
                    candidates.append(EntityMergeCandidate(
                        entities=[entity1, entity2],
                        canonical_form=canonical,
                        confidence=similarity,
                        merge_reason="string_similarity_fallback"
                    ))

        return candidates


```

### 2. Entity Deduplication Service

Create `packages/morag-graph/src/morag_graph/entity_deduplicator.py`:

```python
from typing import Dict, List, Set, Tuple
from .entity_normalizer import EnhancedEntityNormalizer, EntityMergeCandidate
from .neo4j_service import Neo4jService

class EntityDeduplicator:
    def __init__(self, neo4j_service: Neo4jService, llm_service=None):
        self.neo4j = neo4j_service
        self.normalizer = EnhancedEntityNormalizer(llm_service)

    async def deduplicate_entities(self, collection_name: str, language: str = None) -> Dict[str, any]:
        """Deduplicate entities in a collection."""
        # 1. Get all entities
        entities = await self._get_all_entities(collection_name)

        # 2. Find merge candidates using LLM-based analysis
        candidates = await self.normalizer.find_merge_candidates(entities, language)

        # 3. Apply merges
        merge_results = []
        for candidate in candidates:
            if candidate.confidence > 0.8:  # High confidence threshold
                result = await self._merge_entities(candidate, collection_name)
                merge_results.append(result)

        return {
            'total_entities_before': len(entities),
            'merge_candidates_found': len(candidates),
            'merges_applied': len(merge_results),
            'merge_results': merge_results
        }

    async def _get_all_entities(self, collection_name: str) -> List[str]:
        """Get all entity names from the graph."""
        query = """
        MATCH (e:Entity)
        WHERE e.collection_name = $collection_name
        RETURN DISTINCT e.name as entity_name
        """
        
        result = await self.neo4j.execute_query(query, {'collection_name': collection_name})
        return [record['entity_name'] for record in result]

    async def _merge_entities(self, candidate: EntityMergeCandidate, collection_name: str) -> Dict[str, any]:
        """Merge entities in the graph."""
        entities_to_merge = candidate.entities
        canonical_form = candidate.canonical_form
        
        # Find the canonical entity or create it
        canonical_entity = None
        for entity in entities_to_merge:
            if entity == canonical_form:
                canonical_entity = entity
                break
                
        if not canonical_entity:
            canonical_entity = canonical_form
            
        # Merge all relationships to canonical entity
        merge_query = """
        MATCH (old:Entity)
        WHERE old.name IN $entities_to_merge 
          AND old.collection_name = $collection_name
          AND old.name <> $canonical_form
        
        MATCH (canonical:Entity {name: $canonical_form, collection_name: $collection_name})
        
        // Transfer all relationships
        OPTIONAL MATCH (old)-[r1]->(other)
        WHERE NOT (canonical)-[:SAME_TYPE]->(other)
        CREATE (canonical)-[r2:SAME_TYPE]->(other)
        SET r2 = properties(r1)
        
        OPTIONAL MATCH (other)-[r3]->(old)
        WHERE NOT (other)-[:SAME_TYPE]->(canonical)
        CREATE (other)-[r4:SAME_TYPE]->(canonical)
        SET r4 = properties(r3)
        
        // Delete old entities and relationships
        DETACH DELETE old
        
        RETURN count(old) as merged_count
        """
        
        result = await self.neo4j.execute_query(merge_query, {
            'entities_to_merge': entities_to_merge,
            'canonical_form': canonical_form,
            'collection_name': collection_name
        })
        
        return {
            'merged_entities': entities_to_merge,
            'canonical_form': canonical_form,
            'confidence': candidate.confidence,
            'reason': candidate.merge_reason,
            'merged_count': result[0]['merged_count'] if result else 0
        }
```

## Integration Points

### 1. Update Entity Extraction Pipeline

Modify `packages/morag-graph/src/morag_graph/entity_extractor.py`:

```python
from .entity_normalizer import EnhancedEntityNormalizer

class EntityExtractor:
    def __init__(self, llm_service=None):
        # ... existing initialization
        self.normalizer = EnhancedEntityNormalizer(llm_service)

    async def extract_entities(self, text: str, language: str = None) -> List[Dict[str, any]]:
        """Extract and normalize entities."""
        # ... existing extraction logic

        # Normalize extracted entities using LLM-based approach
        normalized_entities = []
        for entity in raw_entities:
            variation = await self.normalizer.normalize_entity(entity['name'], language)
            entity['normalized_name'] = variation.normalized
            entity['normalization_confidence'] = variation.confidence
            entity['normalization_rule'] = variation.rule_applied
            normalized_entities.append(entity)

        return normalized_entities
```

### 2. Add Deduplication CLI Command

Create `cli/deduplicate-entities.py`:

```python
import asyncio
import argparse
from morag_graph.entity_deduplicator import EntityDeduplicator
from morag_graph.neo4j_service import Neo4jService

async def main():
    parser = argparse.ArgumentParser(description='Deduplicate entities in collection')
    parser.add_argument('collection_name', help='Collection to deduplicate')
    parser.add_argument('--language', help='Language context for normalization')
    parser.add_argument('--dry-run', action='store_true', help='Show candidates without merging')

    args = parser.parse_args()

    neo4j_service = Neo4jService()
    llm_service = LLMService()  # Initialize LLM service
    deduplicator = EntityDeduplicator(neo4j_service, llm_service)

    if args.dry_run:
        # Show candidates only
        entities = await deduplicator._get_all_entities(args.collection_name)
        candidates = await deduplicator.normalizer.find_merge_candidates(entities, args.language)

        print(f"Found {len(candidates)} merge candidates:")
        for candidate in candidates:
            print(f"  {candidate.entities} -> {candidate.canonical_form} "
                  f"(confidence: {candidate.confidence:.2f}, reason: {candidate.merge_reason})")
    else:
        # Perform deduplication
        result = await deduplicator.deduplicate_entities(args.collection_name, args.language)
        print(f"Deduplication complete:")
        print(f"  Entities before: {result['total_entities_before']}")
        print(f"  Candidates found: {result['merge_candidates_found']}")
        print(f"  Merges applied: {result['merges_applied']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing Strategy

### 1. Unit Tests

Create comprehensive tests for normalization rules:

```python
# tests/unit/test_entity_normalization.py
import pytest
from morag_graph.entity_normalizer import EnhancedEntityNormalizer

class TestEntityNormalization:
    def setup_method(self):
        self.normalizer = EnhancedEntityNormalizer()

    def test_acronym_normalization(self):
        test_cases = [
            ("A.I.", "AI"),
            ("M.L.", "ML"),
            ("ai", "artificial intelligence"),
            ("ml", "machine learning")
        ]
        
        for input_entity, expected in test_cases:
            result = self.normalizer.normalize_entity(input_entity)
            assert result.normalized == expected

    def test_person_name_normalization(self):
        test_cases = [
            ("Smith, John", "John Smith"),
            ("Dr. Jane Doe", "Jane Doe"),
            ("john smith", "John Smith")
        ]
        
        for input_name, expected in test_cases:
            result = self.normalizer.normalize_entity(input_name)
            assert result.normalized == expected

    def test_organization_normalization(self):
        test_cases = [
            ("Google Inc.", "Google Inc"),
            ("Microsoft Corporation", "Microsoft Corp"),
            ("Apple Computer Company", "Apple Computer Co")
        ]
        
        for input_org, expected in test_cases:
            result = self.normalizer.normalize_entity(input_org)
            assert result.normalized == expected
```

### 2. Integration Tests

Test the complete deduplication pipeline:

```python
# tests/integration/test_entity_deduplication.py
import pytest
from morag_graph.entity_deduplicator import EntityDeduplicator

class TestEntityDeduplication:
    @pytest.mark.asyncio
    async def test_full_deduplication_pipeline(self):
        # Test with sample entities that should be merged
        pass
```

## Configuration

Add normalization settings:

```yaml
# entity_normalization.yml
entity_normalization:
  enabled: true
  confidence_threshold: 0.8
  
  rules:
    acronym_expansion: true
    person_name_standardization: true
    organization_cleanup: true
    general_lowercasing: true
    
  merge_thresholds:
    acronym: 0.9
    person: 0.8
    organization: 0.85
    general: 0.95
    
  custom_acronyms:
    # Add domain-specific acronyms
    "gpt": "generative pre-trained transformer"
    "bert": "bidirectional encoder representations from transformers"
```

## Monitoring

Track normalization effectiveness:

```python
normalization_metrics = {
    'entities_processed': 0,
    'entities_normalized': 0,
    'merge_candidates_found': 0,
    'merges_applied': 0,
    'normalization_rules_used': {
        'acronym': 0,
        'person': 0,
        'organization': 0,
        'general': 0
    },
    'average_confidence': 0.0
}
```

## Success Metrics

- **Duplicate Reduction**: 15-20% reduction in duplicate entities
- **Graph Quality**: Improved connectivity and relationship density
- **Query Performance**: Faster entity lookup and traversal
- **Accuracy**: >95% correct normalization decisions

## Future Enhancements

1. **Machine Learning**: Train models on normalization patterns
2. **Domain Adaptation**: Specialized rules for different domains
3. **User Feedback**: Learn from manual corrections
4. **Real-time Processing**: Normalize entities during ingestion
