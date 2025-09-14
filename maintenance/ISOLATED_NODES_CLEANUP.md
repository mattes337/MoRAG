# Isolated Nodes Cleanup Maintenance Job

## Overview

The `ISOLATED_NODES_CLEANUP` maintenance job identifies and removes completely isolated nodes (entities with no relationships) from the Neo4j knowledge graph. These nodes are typically orphaned entities that were created but never connected to the knowledge graph through relationships.

## What it does

### 1. Isolated Node Detection
- **Complete Isolation**: Finds nodes that have no incoming or outgoing relationships
- **All Node Types**: Checks all node types (Entity, SUBJECT, OBJECT, etc.)
- **Comprehensive Scan**: Examines the entire graph to identify disconnected nodes

### 2. Batch Processing
- **Efficient Removal**: Processes isolated nodes in configurable batches
- **Memory Management**: Prevents memory issues with large numbers of isolated nodes
- **Progress Tracking**: Logs progress for each batch processed

### 3. Safety Checks
- **Double Verification**: Re-verifies isolation before deletion to prevent race conditions
- **Dry Run Support**: Preview mode to see what would be removed without making changes
- **Detailed Logging**: Comprehensive logging of all operations

## What it doesn't do

- **Connected Nodes**: Does not remove nodes that have any relationships
- **System Nodes**: Does not remove nodes that are part of the graph structure
- **Relationship Cleanup**: Does not handle relationship cleanup (use `relationship_cleanup` for that)

## Why isolated nodes occur

Isolated nodes can occur due to:

1. **Failed Relationship Creation**: Entities created but relationships failed to be established
2. **Relationship Cleanup**: Previous maintenance jobs removed all relationships to/from a node
3. **Import Errors**: Incomplete data imports that created entities without connections
4. **Processing Failures**: Interrupted processing that left orphaned entities

## Configuration

The job supports the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MORAG_ISOLATED_CLEANUP_DRY_RUN` | `true` | Preview changes without applying them |
| `MORAG_ISOLATED_CLEANUP_BATCH_SIZE` | `100` | Number of nodes to process in each batch |
| `MORAG_ISOLATED_CLEANUP_JOB_TAG` | `""` | Job tag for tracking and idempotency |

## Running the Job

### Using the Maintenance Runner (recommended)

```bash
# Run all maintenance jobs (includes isolated_nodes_cleanup)
python scripts/maintenance_runner.py

# Run only isolated nodes cleanup
MORAG_MAINT_JOBS="isolated_nodes_cleanup" python scripts/maintenance_runner.py

# Apply changes (disable dry-run)
MORAG_MAINT_JOBS="isolated_nodes_cleanup" \
MORAG_ISOLATED_CLEANUP_DRY_RUN="false" \
python scripts/maintenance_runner.py

# Preview changes only (dry-run)
MORAG_MAINT_JOBS="isolated_nodes_cleanup" \
MORAG_ISOLATED_CLEANUP_DRY_RUN="true" \
python scripts/maintenance_runner.py

# Custom batch size
MORAG_MAINT_JOBS="isolated_nodes_cleanup" \
MORAG_ISOLATED_CLEANUP_DRY_RUN="false" \
MORAG_ISOLATED_CLEANUP_BATCH_SIZE="50" \
python scripts/maintenance_runner.py
```

### Using Docker

```bash
# Dry run
docker run --rm \
  -e MORAG_MAINT_JOBS="isolated_nodes_cleanup" \
  -e MORAG_NEO4J_URI="bolt://neo4j:7687" \
  -e MORAG_NEO4J_USER="neo4j" \
  -e MORAG_NEO4J_PASSWORD="password" \
  morag-maintenance

# Apply changes
docker run --rm \
  -e MORAG_MAINT_JOBS="isolated_nodes_cleanup" \
  -e MORAG_ISOLATED_CLEANUP_DRY_RUN="false" \
  -e MORAG_NEO4J_URI="bolt://neo4j:7687" \
  -e MORAG_NEO4J_USER="neo4j" \
  -e MORAG_NEO4J_PASSWORD="password" \
  morag-maintenance
```

## Output

The job provides detailed output including:

```json
{
  "total_nodes_checked": 1500,
  "isolated_nodes_found": 3,
  "isolated_nodes_removed": 3,
  "execution_time_seconds": 2.5,
  "dry_run": false,
  "job_tag": "",
  "details": {
    "isolated_node_samples": [
      {
        "id": "entity_123",
        "name": "Orphaned Entity",
        "labels": ["Entity"]
      }
    ],
    "batch_size": 100
  }
}
```

### Key Metrics

- **total_nodes_checked**: Total number of nodes examined in the database
- **isolated_nodes_found**: Number of completely isolated nodes discovered
- **isolated_nodes_removed**: Number of isolated nodes actually removed (0 in dry-run mode)
- **execution_time_seconds**: Time taken to complete the cleanup
- **details.isolated_node_samples**: Sample of isolated nodes found (up to 10)

## Performance Considerations

- **Batch Processing**: Uses configurable batch sizes to manage memory usage
- **Efficient Queries**: Uses optimized Neo4j queries to identify isolated nodes
- **Minimal Impact**: Designed to have minimal impact on running applications
- **Quick Execution**: Typically completes quickly unless there are many isolated nodes

## Safety Features

- **Dry Run Default**: Defaults to dry-run mode to prevent accidental deletions
- **Double Verification**: Re-checks isolation status before deletion
- **Comprehensive Logging**: Detailed logging for audit trails
- **Batch Limits**: Processes nodes in batches to prevent overwhelming the database

## Integration with Other Jobs

- **Run after**: `relationship_cleanup` (to clean up nodes orphaned by relationship removal)
- **Run before**: Other jobs that might create new relationships
- **Independent**: Can be run independently without dependencies on other jobs

## Troubleshooting

### Common Issues

1. **No isolated nodes found**: This is normal if your graph is well-connected
2. **Large number of isolated nodes**: May indicate data quality issues in ingestion
3. **Slow execution**: Consider reducing batch size or running during off-peak hours

### Monitoring

Monitor the following:
- Number of isolated nodes over time
- Execution time trends
- Batch processing efficiency
- Error rates during cleanup

## Best Practices

1. **Regular Execution**: Run periodically to maintain graph cleanliness
2. **Dry Run First**: Always run in dry-run mode first to preview changes
3. **Monitor Results**: Track isolated node counts to identify data quality issues
4. **Coordinate with Ingestion**: Run after major data ingestion operations
5. **Backup Before**: Consider backing up before running in production

## Related Jobs

- `relationship_cleanup`: Cleans up problematic relationships that might orphan nodes
- `keyword_deduplication`: Merges duplicate entities that might leave orphans
- `keyword_hierarchization`: Organizes entities that might affect connectivity
