# Advanced DSPy QA (Marktechpost) — Implications for MoRAG

Source: https://github.com/Marktechpost/AI-Tutorial-Codes-Included/blob/main/advanced_dspy_qa_Marktechpost.ipynb

## Summary
The notebook demonstrates composing question answering pipelines with DSPy: declarative modules, signatures, and optimizers that tune prompts and routing using example data. Emphasis on programmatic prompt optimization and evaluation.

## Key Ideas
- Declarative task Signatures and Modules; automatic prompt/program synthesis.
- Training‑time optimization using few labeled examples and metrics.
- Modular composition: Retriever → Reranker → Answerer with guardrails.
- Built‑in evaluation harness and ablation to choose best variants.

## Comparison to MoRAG
MoRAG could leverage DSPy‑style optimization to auto‑tune:
- Entity extraction prompts, relation type normalization rules, and fact extraction prompts.
- Query classification thresholds and hop budgets.
- Reranking prompts for fact selection and citation formatting.

## Implementation Sketch
- Define Signatures for: extract_entities, normalize_entity, extract_relations, extract_facts, synthesize_answer.
- Collect small training sets from existing uploads and tests; define metrics (precision/recall of entities, citation accuracy, fact coverage).
- Run DSPy optimizer offline to generate tuned prompts/parameters written to llm/*.md guidelines.

## Testing
- Split fixed validation set; ensure improvements vs current prompts.
- Regression tests to lock in tuned prompts; fail on degradation.

## Fit with MoRAG
- Matches user’s requirement for high‑quality extraction and exhaustive fact retrieval; can run offline, no runtime dependency.

## Takeaways we can adapt in MoRAG
- Introduce a lightweight prompt optimization loop (DSPy‑style) offline
- Define evaluation metrics for entities, relations, facts, and citations
- Store tuned prompts and parameters versioned in llm/ and configs
- Add regression tests to prevent prompt drift



## Technical Deep Dive: DSPy-Style Optimization for MoRAG

### Define Signatures (Pydantic-like)
```python
from pydantic import BaseModel

class ExtractEntitiesSig(BaseModel):
    text: str
    language: str | None = None

class ExtractEntitiesOut(BaseModel):
    entities: list[str]

class ExtractRelationsSig(BaseModel):
    text: str
    entities: list[str]

class ExtractRelationsOut(BaseModel):
    triples: list[tuple[str, str, str]]  # (subj, REL, obj) REL=UPPERCASE
```

### Modules and Tunables
```python
class PromptModule:
    def __init__(self, template: str, params: dict):
        self.template = template
        self.params = params
    def render(self, **kwargs) -> str:
        return self.template.format(**self.params, **kwargs)

entity_module = PromptModule(
    template=(
        "Extract entities (broad labels). Return JSON: {\"entities\":[...]}\n"
        "Text: {text}\nLanguage: {language}\n"
    ),
    params={}
)

# Tunables (optimized by search):
TUNABLES = {
    "temperature": [0.1, 0.2, 0.3],
    "top_p": [0.7, 0.9],
    "system_preamble": ["You are precise.", "You are recall-optimized."],
}
```

### Optimizer Loop (Pseudocode)
```python
from itertools import product

def evaluate(run_conf, dataset):
    # returns metrics dict
    pass

best, best_score = None, -1
for vals in product(*TUNABLES.values()):
    run_conf = dict(zip(TUNABLES.keys(), vals))
    metrics = evaluate(run_conf, validation_set)
    score = 0.5*metrics['entity_f1'] + 0.5*metrics['citation_acc']
    if score > best_score:
        best, best_score = (run_conf, metrics), score

# Persist best to llm/EXTRACTION.md parameters section
```

### Evaluation Metrics
- entity_f1: against annotated entities per chunk
- relation_f1: normalized relation type accuracy (singular UPPERCASE)
- fact_coverage: proportion of relevant facts extracted above threshold
- citation_acc: exact match on [filename:chunk:topic]

### Regression Tests
```python
def test_tuned_prompts_improve_entities(baseline, tuned, dataset):
    b = evaluate(baseline, dataset)['entity_f1']
    t = evaluate(tuned, dataset)['entity_f1']
    assert t >= b
```

### Integration Plan
- Offline script under tools/prompt_tuning.py
- Outputs: configs/llm_tuned.yaml + updated llm/*.md with chosen templates
- CI gate: run small validation to ensure no regression before merge
