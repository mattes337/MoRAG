## Keyword Hierarchization Background Job

### Purpose
Automatically refine broad keywords into a small, balanced set of more specific sibling/child keywords by analyzing all attached facts. This reduces over-broad hubs (e.g., COMPUTER) and improves retrieval quality by connecting facts to more specialized or generalized keywords where appropriate.

### Quick Run (Standalone Maintenance Container)
- Build image:
```bash
docker build -f Dockerfile.maintenance -t morag-maintenance:latest .
```
- Run (dry-run by default):
```bash
docker run --rm \
  -e NEO4J_URI -e NEO4J_USERNAME -e NEO4J_PASSWORD -e NEO4J_DATABASE \
  morag-maintenance:latest
```
- Apply with detachment and entity links:
```bash
docker run --rm \
  -e NEO4J_URI -e NEO4J_USERNAME -e NEO4J_PASSWORD -e NEO4J_DATABASE \
  -e MORAG_KWH_APPLY=true -e MORAG_KWH_DETACH_MOVED=true \
  -e MORAG_KWH_LINK_ENTITIES=true -e MORAG_KWH_ENTITY_LINK_TYPE=NARROWS_TO \
  morag-maintenance:latest
```
- CLI (no Docker):
```bash
python -m morag_graph.maintenance.keyword_hierarchization \
  --threshold 50 --limit-keywords 5 --apply --detach-moved \
  --link-entities --entity-link-type NARROWS_TO
```

### Environment Variables (Overrides)
- MORAG_KWH_THRESHOLD (default 50)
- MORAG_KWH_MIN_NEW (3), MORAG_KWH_MAX_NEW (6)
- MORAG_KWH_MIN_PER (5)
- MORAG_KWH_MAX_MOVE_RATIO (0.8)
- MORAG_KWH_SHARE (0.18)
- MORAG_KWH_BATCH_SIZE (200)
- MORAG_KWH_DETACH_MOVED (false)
- MORAG_KWH_APPLY (false → dry-run; true applies)
- MORAG_KWH_JOB_TAG (auto if empty)
- MORAG_KWH_LIMIT_KEYWORDS (5)


### High-level Flow
1. Find candidate keywords with too many facts (default threshold: 50)
2. For each candidate keyword:
   - Load all attached facts and co-occurring entities/keywords
   - Propose 3–6 directly related keywords that specialize or generalize the candidate
   - For each fact, decide whether to keep it on the original keyword or move it to the best-fit new keyword
   - Perform a balanced reassignment (not all facts must move)

### Configuration
- threshold_min_facts: int (default 50)
- min_new_keywords: int (default 3)
- max_new_keywords: int (default 6)
- max_move_ratio: float (default 0.8) — keep at least 20% facts on the original keyword to preserve generality when appropriate
- min_per_new_keyword: int (default 5) — only keep a proposed keyword if it earns enough facts after scoring
- similarity_method: one of [co_occurrence, tfidf, pmi] (default co_occurrence)
- dry_run: bool (default true for first execution)
- batch_size: int (default 200 facts per reassignment batch)
- job_tag: string — used to tag writes for traceability and rollback

### Assumptions & Data Model Notes
- Keywords are regular Entity nodes (unique by name) produced by LLM normalization. There is no special "Keyword" static type; rather, names like "COMPUTER", "MOTHERBOARD" are entities.
- Facts are separate nodes (e.g., Fact) linked to entities via typed relationships (e.g., ABOUT, INVOLVES, RELATES_TO). Exact types may vary; adjust queries accordingly.
- Relationship types in Neo4j are normalized to uppercase singular forms per system guidelines.
- Entity normalization is in place to ensure we MERGE into existing nodes if they already exist.

### Candidate Selection (Cypher sketch)
- Replace relationship types to match your graph (ABOUT/INVOLVES/etc.).

```
// Find keywords (entities) that have too many facts attached
MATCH (k:Entity)
MATCH (f:Fact)-[r:ABOUT|INVOLVES|RELATES_TO]->(k)
WITH k, count(DISTINCT f) AS fact_count
WHERE fact_count >= $threshold_min_facts
RETURN k.name AS keyword, fact_count
ORDER BY fact_count DESC
```

### Deriving Specialized/Generalized Keywords
Goal: propose 3–6 related keywords that either specialize (narrower concepts frequently co-occurring in the facts) or generalize (broader parent concepts inferred from context).

Heuristics (combine for robustness):
- Co-occurrence scoring: count how often an entity co-occurs with the candidate across facts; weight by distinct fact coverage.
- PMI (pointwise mutual information): reward entities that are disproportionately common with the candidate relative to their global frequency.
- TF-IDF-like: treat each fact as a document; candidate keyword occurrences as terms; prefer entities frequent with candidate but not global.
- Lexical/structural signals: component-of/part-of relations suggest specialization; IS_A/TYPE_OF suggests generalization.

Cypher sketch for co-occurring entities:
```
MATCH (k:Entity {name: $keyword})
MATCH (f:Fact)-[:ABOUT|INVOLVES|RELATES_TO]->(k)
MATCH (f)-[:ABOUT|INVOLVES|RELATES_TO]->(e:Entity)
WHERE e <> k
WITH e, count(DISTINCT f) AS cofacts
ORDER BY cofacts DESC
LIMIT 50
RETURN e.name AS candidate, cofacts
```

Post-filtering and ranking:
- Remove stop-terms and overly broad terms (e.g., "THING", "SYSTEM") via a small configurable denylist.
- Prefer entities where a significant share of the candidate's facts mention them (e.g., cofacts / total_facts_for_keyword >= 0.15–0.25).
- Cap at max_new_keywords; preserve diversity by down-weighting highly similar proposals.

### Creating/Reusing Proposed Keywords
For each proposed keyword name p:
- Ensure entity normalization — MERGE by normalized name
- Optionally annotate via a short LLM check to label as "specialization" or "generalization" of the original keyword based on definitions and observed facts.

Cypher sketch:
```
MERGE (p:Entity {name: $proposed_name_normalized})
ON CREATE SET p.created_at = timestamp(), p.source = 'keyword_hierarchization', p.job_tag = $job_tag
ON MATCH  SET p.last_seen_at = timestamp(), p.job_tag = $job_tag
RETURN p
```

### Assigning Facts to New Keywords (Balanced Partition)
For each fact f currently attached to original keyword k:
1. Compute affinity(f, p) for each proposed keyword p based on:
   - Does f mention p directly as an entity? Strong signal
   - Overlap of f's entities with p's neighborhood (shared co-occurring entities)
   - Optional: textual similarity between f.text and p.name (embedding or keyword match)
2. If max affinity < tau_keep (e.g., 0.4), keep f attached to k only.
3. Otherwise, assign f to argmax_p affinity(f, p) with these constraints:
   - Maintain balanced partitions: enforce soft caps so that no single p receives an excessive fraction early. Use round-robin among near-ties.
   - Enforce min_per_new_keyword; drop p if after provisional assignment it would have < min_per_new_keyword.
4. Not all facts must be moved. Respect max_move_ratio to keep a portion on k.

Implementation pattern (batching):
```
// Example move: connect fact to new keyword (do not remove from k yet)
MATCH (f:Fact {id: $fact_id})
MATCH (p:Entity {name: $pname})
MERGE (f)-[r:ABOUT]->(p)
ON CREATE SET r.source = 'keyword_hierarchization', r.job_tag = $job_tag, r.created_at = timestamp()
```

Only after all provisional attachments:
- Compute final counts per p; remove underfilled p (< min_per_new_keyword) by deleting their provisional attachments for this job_tag
- Enforce max_move_ratio by trimming lowest-affinity moves back to k
- When satisfied, optionally remove the original (f)-[:ABOUT]->(k) for facts that were reassigned. If a fact remains relevant to both k and p, keep both edges.

Edge update sketch:
```
// Remove the connection from original keyword k if we decided to move
MATCH (f:Fact {id: $fact_id})- [r:ABOUT]->(k:Entity {name: $keyword})
WHERE $should_detach = true
DELETE r
```

### Idempotency, Traceability, Rollback
- All new writes tagged with job_tag and timestamp
- Idempotent MERGE on entities and relationships
- For rollback of a specific job run: find all relationships with the job_tag and delete them, then restore original edges if recorded in an audit log table/file
- Keep an audit log for: keyword, proposed keywords, per-fact assignments before/after

### Performance Considerations
- Process keywords one by one; limit to top-N by fact_count per run
- Stream/batch facts (batch_size) to avoid memory spikes
- Use indexes on Entity(name) and Fact(id)
- Pre-compute global entity frequencies to enable PMI scoring efficiently

### Example
- Original keyword: COMPUTER (50 facts)
- Proposed keywords: MOTHERBOARD, PROCESSOR, HARDDRIVE
- After scoring and balancing, move ~60–80% of highly specific facts to the new keywords, keep ambiguous/generic facts on COMPUTER

Example decision rule for a single fact:
- Mentions PROCESSOR and CLOCK SPEED explicitly -> assign to PROCESSOR
- Mentions STORAGE CAPACITY and DISK RPM -> assign to HARDDRIVE
- High-level fact like "A computer consists of several components" -> keep on COMPUTER

### Pseudocode (Python-like)
```
params = {
  'threshold_min_facts': 50,
  'min_new_keywords': 3,
  'max_new_keywords': 6,
  'max_move_ratio': 0.8,
  'min_per_new_keyword': 5,
  'similarity_method': 'co_occurrence',
  'batch_size': 200,
  'job_tag': f"kw_hier_{int(time.time())}",
  'dry_run': True,
}

for k in find_keywords_over_threshold(params['threshold_min_facts']):
    facts = load_facts_for_keyword(k)
    proposals = rank_related_keywords(k, facts, method=params['similarity_method'])
    proposals = diversify_and_trim(proposals, params['min_new_keywords'], params['max_new_keywords'])
    if not proposals:
        continue

    merged = [merge_keyword_entity(p) for p in proposals]

    assignments = {}
    for f in facts:
        scores = {p: affinity(f, p) for p in merged}
        p_star, s_max = argmax(scores)
        if s_max < 0.4:  # tau_keep
            continue  # keep on original
        assignments.setdefault(p_star, []).append((f, s_max))

    # Balance and thresholds
    assignments = enforce_balance(assignments, min_per=params['min_per_new_keyword'], max_move_ratio=params['max_move_ratio'], total=len(facts))

    if not params['dry_run']:
        # Attach new edges
        for p, items in assignments.items():
            for (f, _) in batched(items, params['batch_size']):
                attach_fact_to_keyword(f, p, job_tag=params['job_tag'])
        # Optionally remove edges from original if moved
        finalize_detachments(k, assignments)
    else:
        log_plan(k, merged, assignments)
```

### Safety & Logging
- Start with dry_run=true; emit a full plan (CSV/JSON) for manual review
- Log per keyword: total facts, proposals, move counts per proposal, kept count
- Add sampling-based validation prints (e.g., show 5 moved facts per proposal for spot check)

### Scheduling & Execution
- Run as a periodic background job (e.g., hourly or nightly) via the existing worker/queue setup
- Limit processing per run (e.g., top 5 bloated keywords) to keep runtime bounded
- Re-run safely thanks to MERGE and job_tag idempotency

### Testing Strategy
- Unit: scoring (affinity), balancing, proposal ranking, normalization
- Integration: dry-run plan generation on a fixture graph; verify counts and constraints (min_per, max_move_ratio) are upheld
- Property tests: small random graphs to ensure no keyword is left with zero facts unless intended

