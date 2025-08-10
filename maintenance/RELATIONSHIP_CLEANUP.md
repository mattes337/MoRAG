# Relationship Cleanup Maintenance Job

## Overview

The `RELATIONSHIP_CLEANUP` maintenance job performs comprehensive cleanup of problematic relationships in the Neo4j knowledge graph. It identifies and removes relationships that are duplicates, semantically meaningless, or logically invalid.

## What it does

### 1. LLM-Powered Assessment
- **Intelligent Analysis**: Uses LLM to assess relationship types and individual relationships for meaningfulness, validity, and duplication
- **Context-Aware Decisions**: Considers entity types, relationship semantics, and graph context
- **Performance Optimization**: Analyzes relationship types first for bulk operations, then handles individual cases
- **Fallback Rules**: Uses rule-based detection when LLM is unavailable

### 2. Duplicate Relationship Detection
- **Exact Duplicates**: Identifies relationships with identical source, target, and type
- **Semantic Duplicates**: Uses LLM to find relationships with different types but equivalent meaning (e.g., "WORKS_AT" vs "EMPLOYED_BY")
- **Bidirectional Duplicates**: Detects redundant A→B and B→A relationships where only one direction is meaningful

### 3. Meaningless Relationship Removal
- **"UNRELATED" Relationships**: Removes relationships explicitly marked as "UNRELATED" (if entities are unrelated, they shouldn't be connected)
- **Generic/Vague Types**: LLM identifies overly generic relationship types like "RELATED_TO", "ASSOCIATED_WITH" when more specific relationships exist
- **Self-Referential Relationships**: Removes relationships where source and target are the same entity (unless explicitly valid like "PART_OF" for hierarchical structures)

### 4. Invalid Relationship Detection
- **Orphaned Relationships**: Finds relationships pointing to non-existent entities
- **Low Confidence Relationships**: Identifies relationships with confidence scores below threshold that may be extraction errors
- **Type Incompatibility**: LLM detects relationships that don't make semantic sense between entity types (e.g., "BORN_IN" between two organizations)

### 5. Relationship Consolidation
- **Merge Similar Relationships**: Combines multiple weak relationships into stronger consolidated ones
- **Confidence Aggregation**: Merges confidence scores when consolidating relationships
- **Metadata Preservation**: Maintains source attribution and context when merging

## What it doesn't do

- It does not remove relationships that are semantically valid but low confidence (configurable threshold)
- It does not modify entity nodes themselves (see KEYWORD_DEDUPLICATION.md for entity cleanup)
- It does not create new relationship types (only cleans existing ones)

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MORAG_REL_CLEANUP_DRY_RUN` | `false` | Apply changes by default (set to `true` for preview) |
| `MORAG_REL_CLEANUP_BATCH_SIZE` | `100` | Number of relationships to process per batch |
| `MORAG_REL_CLEANUP_LIMIT_RELATIONS` | `1000` | Maximum relationships to process per run |
| `MORAG_REL_CLEANUP_MIN_CONFIDENCE` | `0.3` | Minimum confidence threshold for relationships |
| `MORAG_REL_CLEANUP_REMOVE_UNRELATED` | `true` | Remove "UNRELATED" type relationships |
| `MORAG_REL_CLEANUP_REMOVE_GENERIC` | `true` | Remove overly generic relationship types |
| `MORAG_REL_CLEANUP_CONSOLIDATE_SIMILAR` | `true` | Merge semantically similar relationships |
| `MORAG_REL_CLEANUP_SIMILARITY_THRESHOLD` | `0.85` | Threshold for semantic similarity merging |
| `MORAG_REL_CLEANUP_JOB_TAG` | `""` | Job tag for tracking and idempotency |
| `MORAG_REL_CLEANUP_ENABLE_ROTATION` | `false` | Enable rotation to prevent processing same relationships |

## Performance & Safety

- **Apply changes by default**: Use `MORAG_REL_CLEANUP_DRY_RUN=true` for preview mode
- **Optimized Performance**: Uses type-based bulk operations for maximum efficiency
- **LLM-powered assessment**: Uses intelligent analysis to make cleanup decisions
- **Type-based optimization**: Analyzes relationship types first, then performs bulk operations for maximum efficiency
- **Batch processing**: Processes relationships in small batches to avoid long transactions
- **Job tagging**: Includes job_tag on modified relationships for auditing
- **Confidence preservation**: Maintains highest confidence when merging relationships
- **Source attribution**: Preserves metadata about relationship origins

## Performance Optimization

The job uses an optimized type-based approach for maximum efficiency:

### Type-Based Approach (Always Used)
1. **Analyze Types**: Get summary of all relationship types and their counts
2. **LLM Assessment**: Use LLM to identify problematic types and merge candidates
3. **Bulk Operations**: Remove entire relationship types (e.g., all "UNRELATED" relationships)
4. **Type Merging**: Convert relationship types in bulk (e.g., "EMPLOYED_BY" → "WORKS_AT")
5. **Individual Cleanup**: Handle remaining issues (orphaned, low confidence)

This approach provides optimal performance by analyzing types first, then performing bulk operations rather than assessing individual relationships.

## Running the Job

### Using the Maintenance Runner (recommended)

```bash
# Run all maintenance jobs (includes relationship_cleanup)
python scripts/maintenance_runner.py

# Run only relationship cleanup
MORAG_MAINT_JOBS="relationship_cleanup" python scripts/maintenance_runner.py

# Apply changes (default behavior)
MORAG_MAINT_JOBS="relationship_cleanup" \
python scripts/maintenance_runner.py

# Preview changes only (dry-run)
MORAG_MAINT_JOBS="relationship_cleanup" \
MORAG_REL_CLEANUP_DRY_RUN="true" \
python scripts/maintenance_runner.py

# Custom configuration
MORAG_MAINT_JOBS="relationship_cleanup" \
MORAG_REL_CLEANUP_DRY_RUN="false" \
MORAG_REL_CLEANUP_LIMIT_RELATIONS="500" \
MORAG_REL_CLEANUP_MIN_CONFIDENCE="0.4" \
python scripts/maintenance_runner.py
```

### Using Docker

```bash
# Dry run
docker run --rm \
  -e MORAG_MAINT_JOBS="relationship_cleanup" \
  -e MORAG_NEO4J_URI="bolt://neo4j:7687" \
  -e MORAG_NEO4J_USER="neo4j" \
  -e MORAG_NEO4J_PASSWORD="password" \
  morag-maintenance

# Apply changes
docker run --rm \
  -e MORAG_MAINT_JOBS="relationship_cleanup" \
  -e MORAG_REL_CLEANUP_DRY_RUN="false" \
  -e MORAG_NEO4J_URI="bolt://neo4j:7687" \
  -e MORAG_NEO4J_USER="neo4j" \
  -e MORAG_NEO4J_PASSWORD="password" \
  morag-maintenance
```

## Output Format

The job returns structured results:

```json
{
  "job": "relationship_cleanup",
  "result": {
    "relationships_processed": 1000,
    "duplicates_removed": 45,
    "meaningless_removed": 23,
    "invalid_removed": 12,
    "consolidated": 18,
    "total_removed": 80,
    "total_modified": 18,
    "execution_time_seconds": 45.2,
    "dry_run": true,
    "details": {
      "unrelated_removed": 15,
      "generic_removed": 8,
      "orphaned_removed": 12,
      "self_referential_removed": 3,
      "low_confidence_removed": 7,
      "semantic_duplicates_merged": 18
    }
  }
}
```

## Examples

### Problematic Relationships Cleaned

1. **UNRELATED relationships**:
   ```
   (Entity A)-[UNRELATED]->(Entity B)  → REMOVED
   ```

2. **Duplicate relationships**:
   ```
   (Person)-[WORKS_AT]->(Company)
   (Person)-[EMPLOYED_BY]->(Company)  → MERGED into WORKS_AT
   ```

3. **Self-referential invalid**:
   ```
   (Person)-[BORN_IN]->(Person)  → REMOVED
   ```

4. **Orphaned relationships**:
   ```
   (Entity A)-[RELATES_TO]->(Non-existent Entity)  → REMOVED
   ```

5. **Generic consolidation**:
   ```
   (A)-[RELATED_TO]->(B) + (A)-[WORKS_AT]->(B)  → Keep WORKS_AT, remove RELATED_TO
   ```

## Integration with Other Jobs

- **Run after**: `keyword_deduplication` (to avoid cleaning relationships to entities that will be merged)
- **Run before**: `keyword_linking` (to avoid creating links between entities with cleaned relationships)
- **Complements**: `relationship_merger` (focuses on semantic merging vs cleanup of invalid relationships)

## Performance Considerations

- Uses batched processing to avoid memory issues with large graphs
- Implements rotation to distribute load across multiple runs
- Configurable limits to control execution time
- Efficient Neo4j queries with proper indexing assumptions
