# Keyword Deduplication Maintenance Job

Performs intelligent deduplication of similar keywords/entities using LLM-based analysis to combine semantically related entities while preserving meaningful distinctions.

## Problem Statement

The current keyword linking system creates connections between keywords based on co-occurrence, but it doesn't address the fundamental issue of having multiple similar entities that should be merged:

- **Plurals vs Singulars**: "Omega 3" vs "Omega-3s" 
- **Formatting Variations**: "Omega 3" vs "Omega-3" vs "Omega-3 Wert"
- **Semantic Extensions**: "Omega-3", "Omega-3 Blutwert", "Omega-3 Mangel" (all referring to the same core concept)
- **Gender/Language Variations**: Male vs female forms, different language variants
- **Minor Additions**: Base concept + specific attributes that don't warrant separate entities

This fragmentation reduces retrieval effectiveness and creates unnecessary complexity in the knowledge graph.

## How Current Keyword Selection Works

Based on the current implementation:

1. **Keyword Selection**: Keywords are selected from Neo4j using fact count thresholds:
   - Only entities with ≥10 facts are considered as "parent" candidates
   - Limited to top N keywords by fact count (default: 10 parents per run)
   - All other entities can be "child" candidates for linking

2. **LLM Evaluation**: The LLM evaluates relationship types between keyword pairs:
   - Uses co-occurrence analysis (minimum 18% shared facts by default)
   - LLM infers specific relationship types (NARROWS_TO, CAUSES, TREATS, etc.)
   - Falls back to ASSOCIATED_WITH if inference fails

3. **Limited Connections**: Only creates 6 links per parent keyword maximum
   - This explains why "only a few connections are created"
   - Designed to prevent graph explosion while maintaining meaningful relationships

## What Keyword Deduplication Does

- **Similarity Detection**: Identifies entities that are semantically similar but distinct in the graph
- **LLM-Based Viability Analysis**: Uses LLM to determine if merging entities improves retrieval
- **Fact Count Consideration**: Weighs the number of facts on each entity in merge decisions
- **Retrieval Impact Assessment**: Evaluates whether merging improves or harms retrieval effectiveness
- **Intelligent Merging**: Combines entities while preserving the most informative name and highest confidence

## What It Doesn't Do

- It does not modify existing facts or relationships (those are updated via relationship rewiring)
- It does not create new entities; only merges existing ones
- It does not affect document chunks or their content
- It does not change entity types (merges only within same semantic domain)

## Deduplication Strategy

### Phase 1: Candidate Identification
1. **Similarity Scoring**: Use multiple similarity metrics:
   - Levenshtein distance for string similarity
   - Semantic similarity via embeddings
   - Pattern matching for common variations (plurals, punctuation)

2. **Clustering**: Group potentially similar entities:
   - Same entity type required
   - Minimum similarity threshold (configurable)
   - Maximum cluster size to prevent over-merging

### Phase 2: LLM Viability Analysis
For each cluster, the LLM evaluates:

```
Entities to evaluate: ["Omega 3", "Omega-3", "Omega-3 Wert", "Omega-3 Blutwert", "Omega-3 Mangel"]
Fact counts: [45, 23, 12, 8, 15]

Analysis criteria:
1. Semantic similarity: Do these refer to the same core concept?
2. Retrieval impact: Would merging improve search and retrieval?
3. Information preservation: What information would be lost/gained?
4. Fact distribution: How are facts distributed across entities?

Decision: MERGE to "Omega-3" (highest fact count + most canonical form)
Rationale: All refer to Omega-3 fatty acids; merging improves retrieval without losing semantic meaning.
```

### Phase 3: Merge Execution
1. **Primary Entity Selection**: Choose the entity with:
   - Highest fact count
   - Most canonical/standard name form
   - Highest confidence score

2. **Relationship Rewiring**: Update all relationships to point to primary entity
3. **Fact Consolidation**: Ensure all facts point to the merged entity
4. **Cleanup**: Remove duplicate entities while preserving all information

## Configuration Options

- `similarity_threshold`: Minimum similarity score for merge candidates (default: 0.75)
- `max_cluster_size`: Maximum entities per merge cluster (default: 8)
- `min_fact_threshold`: **REMOVED** - All entities are now considered regardless of fact count
- `preserve_high_confidence`: Don't merge entities with confidence > threshold (default: 0.95)

### Logic Change: Why All Entities Are Now Considered

**Previous Logic (Removed)**: Only entities with 3+ facts were considered for deduplication.
- **Problem**: This was backwards! Entities with fewer facts are MORE likely to be duplicates.
- **Example**: "ADHD" (50 facts) vs "Aufmerksamkeitsdefizit" (2 facts) - the latter should be merged into the former.

**New Logic**: All entities are considered, with priority given to entities with fewer facts.
- **Rationale**: Entities with few facts are often duplicates, typos, or variations that should be merged.
- **High-confidence filter remains**: Protects well-established entities from incorrect merges.
- **Ordering**: Entities with fewer facts are processed first (more likely to be duplicates).
- `semantic_similarity_weight`: Weight for embedding-based similarity vs string similarity (default: 0.6)
- `dry_run`: Preview merges without applying (default: true)
- `batch_size`: Number of merge operations per transaction (default: 50) - **Internal processing only**
- `limit_entities`: Maximum entities to process per run (default: 100) - **Controls rotation batch size**
- `enable_rotation`: Enable rotation to prevent entity starvation (default: true)
- `process_all_if_small`: Process all entities if total count < limit_entities (default: true)

### Important: Two Different "Batch Sizes"

⚠️ **Common Confusion**: There are two different "batch size" parameters:

1. **Merge Batch Size** (`batch_size`): Internal processing parameter (default: 50)
   - Environment variable: `MORAG_KWD_BATCH_SIZE`
   - Controls how many merge operations are processed in one transaction
   - Does NOT affect rotation or the "batch X/Y" shown in logs

2. **Rotation Batch Size** (`limit_entities`): Controls entity selection per run (default: 100)
   - Environment variable: `MORAG_KWD_LIMIT_ENTITIES`
   - This determines how many entities are processed per run
   - **This is what controls the rotation behavior and batch count in logs**
   - To process 1000 entities per run: set `MORAG_KWD_LIMIT_ENTITIES=1000`

## Deployment Considerations

### Rotation State Management

The rotation mechanism uses deterministic hashing based on `job_tag` to select batches. For predictable rotation across deployments:

1. **Daily Rotation** (default): Uses date-based job_tag (`kw_dedup_YYYYMMDD`)
   - Same batch processed for all runs on the same day
   - Different batch each day automatically
   - Good for daily maintenance schedules

2. **Custom Rotation**: Set `MORAG_KWD_JOB_TAG` environment variable
   - Use sequential tags like `batch_1`, `batch_2`, etc. for manual control
   - Use timestamp-based tags for more frequent rotation
   - Use fixed tags for testing specific batches

3. **Container Orchestration**:
   - Set consistent `MORAG_KWD_JOB_TAG` across container runs
   - Use external scheduling to vary job_tag over time
   - Monitor rotation coverage using entity timestamps

## Rotation Mechanism (Prevents Starvation)

**Problem**: Processing only the top 100 entities by fact count means entities with fewer facts never get processed.

**Solution**: Intelligent rotation system that ensures all entities get processed over time:

1. **Deterministic Rotation**: Uses job_tag hash to select different batches each run
2. **Complete Coverage**: Cycles through ALL entities (not just eligible ones) across multiple runs
3. **Batch Calculation**: Number of batches = ceil(total_entities_all / limit_entities)
   - Example: 576 total entities ÷ 100 per batch = 6 batches
   - Each batch processes up to 100 entities from different offsets
4. **Smart Defaults**:
   - If total entities ≤ 100: processes all entities (no rotation needed)
   - If total entities > 100: rotates through batches of 100
5. **Job Tag Strategy**:
   - Default: Date-based (`kw_dedup_YYYYMMDD`) for consistent daily rotation
   - Custom: Set `MORAG_KWD_JOB_TAG` for specific rotation patterns
   - Deployment: Use same job_tag across container runs for predictable rotation
6. **Tracking**: Marks processed entities with timestamps (in non-dry-run mode)

**Example**: With 500 eligible entities and limit_entities=100:
- Run 1: Processes entities 1-100
- Run 2: Processes entities 101-200
- Run 3: Processes entities 201-300
- Run 4: Processes entities 301-400
- Run 5: Processes entities 401-500
- Run 6: Cycles back to entities 1-100

This ensures **no entity starvation** while maintaining reasonable processing limits per run.

## Safety Measures

- **Dry-run by default**: All operations are previewed before execution
- **Confidence preservation**: High-confidence entities are protected from merging
- **Fact count validation**: Ensures no facts are lost during merging
- **Rollback capability**: Each merge operation is tagged for potential reversal
- **LLM validation**: Every merge must be approved by LLM analysis
- **Rotation tracking**: Prevents duplicate processing in the same cycle

## Expected Impact

- **Improved Retrieval**: Fewer fragmented entities means better search results
- **Cleaner Graph**: Reduced redundancy in the knowledge graph
- **Better Linking**: Subsequent keyword linking will be more effective
- **Enhanced Navigation**: Agents can traverse more meaningful entity relationships

## Running the Job

### Via Maintenance Runner
```bash
# Add to existing jobs
MORAG_MAINT_JOBS="keyword_hierarchization,keyword_linking,keyword_deduplication"

# Run only deduplication
MORAG_MAINT_JOBS="keyword_deduplication"
```

### Standalone Execution
```bash
python -m morag_graph.maintenance.keyword_deduplication \
  --similarity-threshold 0.75 \
  --max-cluster-size 8 \
  --apply  # Remove for dry-run
```

### Environment Variables
- `MORAG_KWD_SIMILARITY_THRESHOLD`: Similarity threshold (default: 0.75)
- `MORAG_KWD_MAX_CLUSTER_SIZE`: Max entities per cluster (default: 8)
- `MORAG_KWD_MIN_FACTS`: Minimum facts to consider (default: 3)
- `MORAG_KWD_BATCH_SIZE`: Batch size for operations (default: 50)
- `MORAG_KWD_LIMIT_ENTITIES`: Max entities per run (default: 100)
- `MORAG_KWD_APPLY`: Apply changes (default: false, dry-run)
- `MORAG_KWD_JOB_TAG`: Custom job tag for tracking
- `MORAG_KWD_ENABLE_ROTATION`: Enable rotation (default: true)
- `MORAG_KWD_PROCESS_ALL_SMALL`: Process all if small dataset (default: true)

## Integration with Existing Jobs

Keyword deduplication should run **before** keyword linking and hierarchization:

1. **Deduplication** → Clean up similar entities
2. **Hierarchization** → Create parent-child relationships  
3. **Linking** → Create semantic connections

This order ensures that linking operates on a clean, deduplicated entity set.
