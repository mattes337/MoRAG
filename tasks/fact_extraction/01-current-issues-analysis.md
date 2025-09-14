# Task 1: Current Issues Analysis

## Objective
Thoroughly analyze the current entity extraction approach to document specific issues and limitations that make the knowledge graph too broad and generic.

## Current Issues Identified

### 1. Language Word Removal Logic
**File**: `packages/morag-graph/src/morag_graph/extraction/relation_extractor.py`
**Lines**: 502-505

**Problem**: Hardcoded language-specific word removal
```python
words_to_remove = [
    'sich', 'zu', 'der', 'die', 'das', 'den', 'dem', 'des',
    'the', 'a', 'an', 'to', 'of', 'for', 'with', 'by'
]
```

**Issues**:
- Too aggressive removal of meaningful words
- Language-specific hardcoding (German/English only)
- No context awareness
- Breaks compound entity names
- Removes prepositions that might be part of proper names

**Impact**: Entity names like "University of California" become "University California", causing matching failures.

### 2. Entity Type Assignment Problems
**Files**: 
- `packages/morag-graph/src/morag_graph/models/entity.py`
- `packages/morag-graph/src/morag_graph/storage/neo4j_operations/entity_operations.py`

**Problem**: All entities getting "ORGANIZATION" label

**Root Causes**:
1. **Default Type Assignment**: Auto-created entities default to "CUSTOM" or "ORGANIZATION"
2. **LLM Prompt Issues**: Entity extraction prompts don't provide sufficient type guidance
3. **Type Normalization**: Generic fallback types when specific types can't be determined

**Code Locations**:
- Line 152 in entity_operations.py: `entity_type="CUSTOM"` default
- Line 32 in entity.py: `type: str = "CUSTOM"` default
- Entity type normalization logic that converts everything to uppercase

### 3. Graph Size and Genericity Issues

**Problems**:
1. **Generic Entity Types**: Too many "CUSTOM", "ENTITY", "THING" types
2. **Overly Broad Extraction**: Extracts every possible noun as an entity
3. **Weak Relationships**: Generic "RELATES_TO" relationships without semantic meaning
4. **No Domain Focus**: Doesn't adapt to specific domains or use cases
5. **Poor Signal-to-Noise Ratio**: Important entities lost among generic ones

**Evidence**:
- Entity normalizer creates generic labels for unclear types
- Relation extractor creates broad, non-specific relationships
- No filtering for entity importance or relevance

### 4. Performance Impact

**Issues**:
1. **Large Graph Size**: Too many nodes and relationships
2. **Poor Query Performance**: Generic types make targeted queries difficult
3. **Memory Usage**: Storing too much irrelevant information
4. **Retrieval Quality**: Important information buried in noise

## Analysis Tasks

### Task 1.1: Code Review and Documentation
- [ ] Review all entity extraction code paths
- [ ] Document current entity type assignment logic
- [ ] Identify all hardcoded language-specific rules
- [ ] Map the complete entity lifecycle from extraction to storage

### Task 1.2: Data Analysis
- [ ] Analyze current graph structure and entity type distribution
- [ ] Measure graph size and complexity metrics
- [ ] Identify most common entity types and relationships
- [ ] Assess retrieval performance with current approach

### Task 1.3: Issue Prioritization
- [ ] Rank issues by impact on system performance
- [ ] Identify quick wins vs. architectural changes needed
- [ ] Determine which issues the fact-based approach will solve

### Task 1.4: Requirements for New Approach
- [ ] Define requirements for fact-based extraction
- [ ] Specify what constitutes a "fact" vs. generic entity
- [ ] Design fact schema (Subject, Object, Approach, Solution, Remarks)
- [ ] Plan migration strategy from current to new approach

## Deliverables

1. **Issue Analysis Report**: Detailed documentation of all current problems
2. **Performance Baseline**: Metrics on current graph size and query performance
3. **Requirements Document**: Specifications for the new fact-based approach
4. **Migration Plan**: Strategy for transitioning to fact-based extraction

## Success Criteria

- All major issues with current approach are documented
- Performance impact is quantified
- Clear requirements for fact-based approach are defined
- Migration strategy is feasible and low-risk
