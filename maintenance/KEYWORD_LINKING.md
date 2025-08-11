# Keyword Linking Maintenance Job

Create inter-entity (keyword-to-keyword) links based on co-occurrence of shared facts. This improves traversal for intelligent retrieval by exposing meaningful paths between concepts (e.g., ADHS → KONZENTRATION).

The job always attempts to use an LLM (Gemini) to infer a specific, normalized relationship type and direction for each link. If no specific type can be identified, it falls back to ASSOCIATED_WITH.

## What it does
- Proposes candidate links for each parent keyword using co-occurrence on shared facts
- Uses Gemini to infer a normalized, uppercase, singular relationship type and a direction (A_TO_B or B_TO_A)
- MERGE-s the link with properties:
  - created_from: "keyword_linking"
  - job_tag: generated per run (e.g., kw_link_...)
  - created_at: datetime on create
- Writes in small batches; designed to be re-runnable/idempotent-ish via MERGE + job_tag

## What it doesn’t do
- It does not rewire facts (see KEYWORD_HIERARCHIZATION.md for that)
- It does not create entity nodes; targets must already exist (created during ingestion or by other jobs)

## Defaults and safety
- Dry-run by default (use --apply to write)
- LLM is always attempted first (Gemini); if inference fails, fallback type ASSOCIATED_WITH is used
- job_tag is included on created relationships for auditing
- Small-batch writes to avoid long transactions

## Running the job

### Maintenance Runner (recommended)
- Run all jobs by default (if MORAG_MAINT_JOBS not set):
  - This includes keyword_deduplication, keyword_hierarchization and keyword_linking
  - Jobs run in optimal order: deduplication → hierarchization → linking

- Only keyword_linking:
```
MORAG_MAINT_JOBS=keyword_linking python scripts/maintenance_runner.py
```

- Apply mode (writes changes):
```
MORAG_MAINT_JOBS=keyword_linking python -m morag_graph.maintenance.keyword_linking --apply
```

### Docker one-shot container
Ensure you set Neo4j connection env and GEMINI_API_KEY.
```
docker run --rm \
  -e NEO4J_URI -e NEO4J_USERNAME -e NEO4J_PASSWORD -e NEO4J_DATABASE \
  -e GEMINI_API_KEY \
  morag-maintenance:latest
```
Apply mode:
```
docker run --rm \
  -e NEO4J_URI -e NEO4J_USERNAME -e NEO4J_PASSWORD -e NEO4J_DATABASE \
  -e GEMINI_API_KEY \
  -e MORAG_MAINT_JOBS=keyword_linking \
  morag-maintenance:latest python -m morag_graph.maintenance.keyword_linking --apply
```

### CLI (no Docker)
```
python -m morag_graph.maintenance.keyword_linking \
  --share 0.18 \
  --limit-parents 10 \
  --max-per-parent 6 \
  --apply
```

## Configuration (CLI flags)
- --share: Min co-occurrence share (default 0.18)
- --limit-parents: Number of parent keywords to consider per run (default 10)
- --max-per-parent: Max links to create per parent (default 6)
- --batch-size: Batch size for writes (default 200)
- --fallback-type: Fallback relation type if LLM cannot infer (default ASSOCIATED_WITH)
- --apply: Apply changes (disable dry-run)
- --job-tag: Optional stable tag for created relationships (auto-generated otherwise)

Note: LLM usage is always on; no flag/env to disable. The implementation prefers Gemini (via morag_reasoning LLMClient settings) and falls back only if inference fails.

## Verification queries
- Links created by this run:
```
MATCH (a:Entity)-[h]->(b:Entity)
WHERE h.created_from = 'keyword_linking' AND h.job_tag = $job_tag
RETURN type(h) AS rel, a.name AS from, b.name AS to, count(*) AS cnt
ORDER BY cnt DESC;
```

- Sample traversal from a specific keyword (e.g., ADHS):
```
MATCH (a:Entity {name:'ADHS'})-[h]->(b:Entity)
RETURN type(h) AS rel, b.name AS to
ORDER BY to;
```

## Operational notes
- LLM prompts follow normalization rules:
  - Uppercase, singular type names (e.g., NARROWS_TO, CAUSES, TREATS, ASSOCIATED_WITH)
  - Direction returned as A_TO_B or B_TO_A; applied accordingly when creating the link
  - Generic types like RELATES_TO/RELATED_TO are normalized to ASSOCIATED_WITH
- The job is re-runnable; MERGE + job_tag minimizes duplicate links
- If rate limits or transient errors occur, the LLM client uses retry/backoff (see morag_reasoning.llm)

## Troubleshooting
- No links created
  - Check that candidate proposals meet the --share threshold
  - Ensure GEMINI_API_KEY is set; otherwise fallback ASSOCIATED_WITH will be used where inference fails
- Links present but type is ASSOCIATED_WITH
  - This is the fallback if the model couldn’t confidently infer a specific type
- Confirm graph writes
  - Use the verification queries with the job_tag printed in the job output

