# MoRAG Graph Maintenance Jobs

This directory contains maintenance jobs for optimizing and cleaning up the Neo4j knowledge graph.

## Available Jobs

### 1. Keyword Deduplication (`keyword_deduplication`)
Performs intelligent deduplication of similar keywords/entities using LLM-based viability analysis.

**Features:**
- Combines semantically related entities while preserving meaningful distinctions
- Uses LLM to evaluate merge viability
- Supports rotation to prevent starvation
- Configurable similarity thresholds and batch sizes

### 2. Keyword Hierarchization (`keyword_hierarchization`)
Creates hierarchical relationships between keywords based on co-occurrence patterns.

**Features:**
- Analyzes fact co-occurrence to identify parent-child relationships
- Moves facts from child to parent entities when appropriate
- Configurable thresholds for relationship creation

### 3. Keyword Linking (`keyword_linking`)
Creates semantic relationships between keywords using LLM analysis.

**Features:**
- Uses LLM to determine relationship types between co-occurring entities
- Creates domain-specific relationship types (TREATS, CAUSES, INFLUENCES, etc.)
- Supports bidirectional relationship analysis

### 4. Relationship Merger (`relationship_merger`)
Merges redundant relationships to reduce graph overhead and improve performance.

**Features:**
- **Duplicate Detection**: Finds exact duplicate relationships (same source, target, type)
- **Semantic Analysis**: Uses LLM to identify semantically equivalent relationship types
- **Bidirectional Merging**: Consolidates A→B and B→A relationships when appropriate
- **Transitive Analysis**: Identifies and merges transitive redundancy (A→B→C where A→C exists)

### 5. Relationship Cleanup (`relationship_cleanup`)
Performs comprehensive cleanup of problematic relationships in the knowledge graph.

**Features:**
- **Meaningless Removal**: Removes "UNRELATED" and overly generic relationship types
- **Invalid Detection**: Finds orphaned, self-referential, and type-incompatible relationships
- **Duplicate Cleanup**: Identifies and removes exact and semantic duplicates
- **Consolidation**: Merges similar relationships with confidence aggregation

## Running Maintenance Jobs

### Using the Maintenance Runner

```bash
# Run all jobs (default order: deduplication, hierarchization, linking, relationship_merger, relationship_cleanup)
python scripts/maintenance_runner.py

# Run specific jobs
MORAG_MAINT_JOBS="keyword_deduplication,relationship_cleanup" python scripts/maintenance_runner.py

# Run single job
MORAG_MAINT_JOBS="relationship_merger" python scripts/maintenance_runner.py
```

### Configuration

All jobs support environment variable configuration:

#### Relationship Merger Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MORAG_REL_SIMILARITY_THRESHOLD` | 0.8 | Threshold for semantic similarity |
| `MORAG_REL_BATCH_SIZE` | 100 | Batch size for merge operations |
| `MORAG_REL_LIMIT_RELATIONS` | 1000 | Max relationships to process per run |
| `MORAG_REL_DRY_RUN` | false | Set to "true" for preview mode (applies changes by default) |
| `MORAG_REL_JOB_TAG` | auto | Job tag for rotation tracking |
| `MORAG_REL_ENABLE_ROTATION` | true | Enable rotation to prevent starvation |
| `MORAG_REL_MERGE_BIDIRECTIONAL` | true | Merge bidirectional relationships |
| `MORAG_REL_MERGE_TRANSITIVE` | false | Merge transitive relationships (conservative) |
| `MORAG_REL_MIN_CONFIDENCE` | 0.5 | Minimum confidence for relationships to consider |

#### Relationship Cleanup

| Variable | Default | Description |
|----------|---------|-------------|
| `MORAG_REL_CLEANUP_DRY_RUN` | `true` | Preview changes without applying them |
| `MORAG_REL_CLEANUP_BATCH_SIZE` | 100 | Number of relationships to process per batch |
| `MORAG_REL_CLEANUP_LIMIT_RELATIONS` | 1000 | Maximum relationships to process per run |
| `MORAG_REL_CLEANUP_MIN_CONFIDENCE` | 0.3 | Minimum confidence threshold for relationships |
| `MORAG_REL_CLEANUP_REMOVE_UNRELATED` | `true` | Remove "UNRELATED" type relationships |
| `MORAG_REL_CLEANUP_REMOVE_GENERIC` | `true` | Remove overly generic relationship types |
| `MORAG_REL_CLEANUP_CONSOLIDATE_SIMILAR` | `true` | Merge semantically similar relationships |
| `MORAG_REL_CLEANUP_SIMILARITY_THRESHOLD` | 0.85 | Threshold for semantic similarity merging |
| `MORAG_REL_CLEANUP_JOB_TAG` | `""` | Job tag for tracking and idempotency |
| `MORAG_REL_CLEANUP_ENABLE_ROTATION` | `false` | Enable rotation to prevent processing same relationships |

#### Example: Relationship Merger

```bash
# Apply changes (default behavior)
MORAG_MAINT_JOBS="relationship_merger" \
MORAG_REL_LIMIT_RELATIONS="500" \
MORAG_REL_SIMILARITY_THRESHOLD="0.9" \
python scripts/maintenance_runner.py

# Dry run (preview only)
MORAG_MAINT_JOBS="relationship_merger" \
MORAG_REL_DRY_RUN="true" \
MORAG_REL_LIMIT_RELATIONS="500" \
python scripts/maintenance_runner.py
```

#### Example: Relationship Cleanup

```bash
# Apply changes (default behavior)
MORAG_MAINT_JOBS="relationship_cleanup" \
MORAG_REL_CLEANUP_DRY_RUN="false" \
MORAG_REL_CLEANUP_LIMIT_RELATIONS="500" \
MORAG_REL_CLEANUP_MIN_CONFIDENCE="0.4" \
python scripts/maintenance_runner.py

# Dry run (preview only)
MORAG_MAINT_JOBS="relationship_cleanup" \
MORAG_REL_CLEANUP_DRY_RUN="true" \
MORAG_REL_CLEANUP_REMOVE_UNRELATED="true" \
python scripts/maintenance_runner.py
```

## Job Output

Each job returns structured results:

### Relationship Merger Output
```json
{
  "total_relationships": 1000,
  "processed_relationships": 1000,
  "duplicate_merges": 12,
  "semantic_merges": 8,
  "bidirectional_merges": 3,
  "transitive_merges": 0,
  "total_merges": 23,
  "processing_time": 0.48,
  "dry_run": true,
  "job_tag": "rel_merge_20250810"
}
```

## Safety Features

- **Dry Run Mode**: Available as opt-in for preview (jobs apply changes by default)
- **Rotation**: Prevents starvation by processing different subsets on each run
- **Confidence Thresholds**: Only processes high-confidence relationships
- **Batch Processing**: Limits memory usage and allows for incremental progress
- **Detailed Logging**: Comprehensive logging for monitoring and debugging

## Performance Considerations

- **Relationship Merger**: Most effective on graphs with many duplicate or redundant relationships
- **Batch Sizes**: Adjust based on available memory and graph size
- **Rotation**: Enables processing of large graphs over multiple runs
- **LLM Calls**: Semantic analysis requires LLM access and may be rate-limited

## Best Practices

1. **Test First**: Use dry-run mode for testing before applying changes
2. **Monitor Performance**: Check processing times and adjust batch sizes
3. **Regular Execution**: Run maintenance jobs regularly to prevent accumulation
4. **Backup First**: Ensure database backups before applying changes
5. **Gradual Rollout**: Start with small batches and increase gradually

## Troubleshooting

- **LLM Errors**: Check API keys and rate limits for semantic analysis
- **Memory Issues**: Reduce batch sizes or limit processing counts
- **Connection Errors**: Verify Neo4j connection settings
- **Performance**: Use rotation and smaller batches for large graphs
