# Improving KG Extraction Robustness with BAML Fuzzy Parsing — Implications for MoRAG

Source: https://levelup.gitconnected.com/improving-langchain-knowledge-graph-extraction-with-baml-fuzzy-parsing-a2413b2a4613 (Aug 2025)

## Summary
The article demonstrates using BAML (typed schema + compiler) with a fuzzy JSON parser to harden LangChain-based knowledge graph (KG) extraction. Core idea: enforce strongly typed outputs with graceful recovery from malformed LLM responses via fuzzy parsing and automatic repair. This increases extraction throughput and reduces failure rates without over-constraining the model.

## Key Ideas
- Strong, compile-time-like schemas for LLM outputs (BAML types) instead of loose dicts.
- Fuzzy parsing to tolerate partial/invalid JSON and salvage usable fields.
- Automatic field-level validation + repair steps to bring responses back to schema.
- Better observability: per-field errors, partial successes, counters for failure modes.
- Separation of concerns: prompt logic vs. output schema vs. repair strategies.

## Technical Pattern
- Define a strict output schema (entities, relations, attributes) with types and enums.
- Call LLM → attempt strict parse → if failure, run fuzzy parser to extract partial fields.
- Apply validator/repair strategies (defaults, coercions, enum normalization).
- Emit structured result + error annotations for logging/metrics.

## Comparison to MoRAG
Current MoRAG already:
- Uses Pydantic models across services and ServiceResultWrapper for consistency.
- Adds retries/backoff for Gemini.
- Implements entity normalization and fact validation layers.

Gaps/opportunities:
- Parsing hardening: move from “assume-valid JSON” to “strict + fuzzy recovery” for extraction tasks (entities/relations/facts), with per-field repair and error annotations.
- Schema-first prompts: generate prompts from types to reduce drift.
- Granular observability: metrics per field (invalid enum, missing required, truncated arrays) to prioritize prompt fixes.

## Risks/Trade-offs
- Slight latency increase (fuzzy parse + repair) but overall throughput improves by reducing retries and manual triage.
- Over-aggressive repair may mask poor prompts—ensure error rates are surfaced.

## Implementation Sketch for MoRAG
- Replace ad-hoc JSON parsing in extractors with Pydantic models plus a fuzzy recovery layer.
- Introduce a small “fuzzy_parse” utility:
  - Attempt model.parse_raw.
  - On failure: regex/partial-JSON salvage → map to fields → coerce types → collect errors.
  - Return (model_instance, issues), never raise for content errors.
- Add RepairPolicy config per extractor: enum normalization to UPPERCASE, trimming long strings, dedup arrays, default confidences.
- Emit structured parse_metrics into logs; aggregate in tests.

## Testing Strategy
- Golden prompts producing valid JSON → zero-issues path.
- Corrupted outputs (missing braces, extra text, trailing commas) → fuzzy salvage should recover ≥80% of fields.
- Enum normalization: verify all relation types are singular UPPERCASE (matches MoRAG Neo4j rules).
- Property coercions (ints, floats) and defaulting behavior.

## Integration Fit
- Aligns with MoRAG’s Pydantic-first design.
- Improves reliability of langextract_relation_extractor and fact extraction.
- Complements existing exponential backoff to reduce overall retries.

## Takeaways we can adapt in MoRAG
- Add a generic fuzzy_parse_json utility with:
  - strict Pydantic parse → fallback fuzzy salvage → repair → issues[]
  - per-field validators for enums, casing, deduplication
- Generate prompts from schemas to reduce output drift
- Log parse_issue metrics and include in ServiceResultWrapper metadata
- Add unit tests for malformed outputs across extractors to guarantee graceful degradation
- Treat “partial but typed” as success: never fail fact/entity extraction solely on JSON shape



## Detailed Architecture Comparison
- BAML-style typed outputs map naturally to MoRAG Pydantic models. Key addition is a tolerant parsing layer between LLM and models.
- Introduce ParseResult wrapper: {data: Optional[T], issues: List[ParseIssue], raw: str}.
- Add RepairPolicy abstractions per extractor (entity, relation, fact) with reusable normalizers.

## Example Schemas for MoRAG (Pydantic)
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class EntityOut(BaseModel):
    name: str = Field(min_length=1)
    labels: List[str] = []  # broad labels instead of generic 'Entity'
    aliases: List[str] = []

class RelationOut(BaseModel):
    source: str
    target: str
    type: str  # normalized to singular UPPERCASE (CURES, SOLVES, ...)
    evidence: Optional[str] = None  # machine-readable source tag

class FactOut(BaseModel):
    subject: str
    predicate: str  # normalized to UPPERCASE
    object: str
    source: str  # [filename:chunk_index:topic]
    confidence: float = 0.75
```

## Robust Parser Pseudocode
```python
# packages/morag-services/src/morag_services/parsers/fuzzy_json.py
from typing import Tuple, Type, TypeVar, List
from pydantic import BaseModel, ValidationError
import json, re

T = TypeVar('T', bound=BaseModel)

class ParseIssue(BaseModel):
    field: str
    issue: str
    severity: str = "WARN"

class RepairPolicy:
    def normalize_enum(self, s: str) -> str:
        return re.sub(r's$', '', s.strip()).upper()
    def clamp_conf(self, x: float) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.5

policy = RepairPolicy()

def strip_nonjson(text: str) -> str:
    # heuristic: grab longest {...} or [...] block
    candidates = re.findall(r"\{.*\}|\[.*\]", text, flags=re.S)
    return max(candidates, key=len) if candidates else text

def fuzzy_parse(model: Type[T], raw: str) -> Tuple[Optional[T], List[ParseIssue]]:
    issues: List[ParseIssue] = []
    try:
        return model.model_validate_json(raw), issues
    except ValidationError as ve:
        issues.append(ParseIssue(field="__root__", issue=str(ve)))
    # salvage path
    core = strip_nonjson(raw)
    try:
        data = json.loads(core)
    except Exception as e:
        issues.append(ParseIssue(field="__json__", issue=f"json load failed: {e}", severity="ERROR"))
        return None, issues
    # field-level repairs
    if isinstance(data, dict):
        if 'type' in data:
            data['type'] = policy.normalize_enum(str(data['type']))
        if 'predicate' in data:
            data['predicate'] = policy.normalize_enum(str(data['predicate']))
        if 'confidence' in data:
            data['confidence'] = policy.clamp_conf(data['confidence'])
    try:
        return model.model_validate(data), issues
    except ValidationError as ve:
        issues.append(ParseIssue(field="__final__", issue=str(ve), severity="ERROR"))
        return None, issues
```

## Integration Points in MoRAG
- Use fuzzy_parse in:
  - packages/morag-graph: langextract_relation_extractor when parsing LLM outputs.
  - packages/morag-reasoning: fact extraction JSON outputs.
- Extend ServiceResultWrapper.metadata with parse_issues: List[ParseIssue].
- Logging: structlog key-value entries per issue.

```python
# logging example
logger.warn("parse_issue", field=iss.field, issue=iss.issue, severity=iss.severity,
            extractor="relation", doc_id=doc_id, chunk=idx)
```

## Prompt-from-Schema Generation
- Generate a JSON schema example in prompts directly from Pydantic models using model_json_schema().
- Enforce: return only JSON, no prose; include machine-readable source tag.

## Tests (pytest)
```python
import pytest
from morag_services.parsers.fuzzy_json import fuzzy_parse
from pydantic import BaseModel

class M(BaseModel):
    predicate: str
    confidence: float

@pytest.mark.parametrize("raw, ok", [
    ("{""predicate"": ""handles"", ""confidence"": 0.9}", True),
    ("garbage before {\"predicate\": \"cures\", \"confidence\": 1.2} trailing", True),
    ("{\"predicate\": 123, \"confidence\": \"0.4\"}", True),
])
def test_fuzzy_parse_salvage(raw, ok):
    m, issues = fuzzy_parse(M, raw)
    assert (m is not None) == ok
    if m:
        assert m.predicate.isupper()
        assert 0.0 <= m.confidence <= 1.0
```

## Migration & Rollout
- Phase 1: wrap extractors with fuzzy_parse; log issues, do not fail pipeline.
- Phase 2: add metrics dashboard for parse_issue distribution; tighten prompts.
- Phase 3: promote to hard requirement (alert on ERROR issues > threshold).
