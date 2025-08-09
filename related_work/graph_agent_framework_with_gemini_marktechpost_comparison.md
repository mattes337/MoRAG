# Graph Agent Framework with Gemini (Marktechpost) — Implications for MoRAG

Source: https://github.com/Marktechpost/AI-Tutorial-Codes-Included/blob/main/graph_agent_framework_with_gemini_Marktechpost.ipynb

## Summary
The notebook implements a graph‑aware agent using Gemini: entity extraction, relation inference, and path reasoning over a lightweight knowledge base. Focus on tool‑augmented Gemini calls and structured outputs.

## Key Ideas
- Gemini as the primary LLM with tool/function calling for graph ops.
- Entity/Relation extraction prompts that return typed structures.
- Path reasoning: select relevant neighbors, expand depth‑limited, and justify hops.
- Built‑in citations and source attribution from retrieved chunks.

## Comparison to MoRAG
- Aligns with MoRAG’s Gemini preference and recursive retrieval design.
- Opportunity: formalize function‑calling schema for graph operations (list_neighbors, get_chunk, add_fact) and let Gemini choose actions.
- Enhance source attribution formatting to always be machine‑readable (filename:chunk:topic).

## Implementation Sketch
- Define function specs for: extract_entities, match_entity, expand_neighbors, fetch_chunk, extract_facts.
- Controller enforces hop limits, score thresholds, and factsOnly/skip‑fact‑evaluation flags.
- Store action traces and justifications for auditability.

## Testing
- Simulate small graph with fixtures; ensure Gemini tool calls follow allowed actions and respect limits.
- Verify citations point to actual documents with page/timecode metadata.

## Fit with MoRAG
- Directly compatible with existing Gemini integration and user requirements (processing and ingest modes, batch embeddings, robust citations).

## Takeaways we can adapt in MoRAG
- Define explicit Gemini function‑calling schema for graph operations
- Record action traces (function calls) for observability and debugging
- Enforce machine‑readable citation format throughout retrieval and synthesis
- Add unit tests for tool‑use control (max depth, allowed actions)



## Gemini Tool-Calling Schema for MoRAG

### Function Specs (JSON Schema-ish)
```json
[
  {
    "name": "extract_entities",
    "description": "Extract entities with broad labels from text",
    "parameters": {"type": "object", "properties": {"text": {"type":"string"}}}
  },
  {
    "name": "match_entity",
    "description": "Resolve mention to canonical entity in graph",
    "parameters": {"type": "object", "properties": {"name": {"type":"string"}}}
  },
  {
    "name": "expand_neighbors",
    "description": "List neighbors up to depth=1",
    "parameters": {"type": "object", "properties": {"entity_id": {"type":"string"}}}
  },
  {
    "name": "fetch_chunk",
    "description": "Load document chunks for entity",
    "parameters": {"type": "object", "properties": {"entity_id": {"type":"string"}}}
  },
  {
    "name": "extract_facts",
    "description": "Extract actionable facts with machine-readable sources",
    "parameters": {"type": "object", "properties": {"text": {"type":"string"}}}
  }
]
```

### Controller Pseudocode
```python
allowed = {"extract_entities", "match_entity", "expand_neighbors", "fetch_chunk", "extract_facts"}

async def handle_tool_call(call, state):
    assert call.name in allowed
    if call.name == "extract_entities":
        ents = await llm_extract_entities(call.args["text"])  # reuse MoRAG service
        return {"entities": ents}
    if call.name == "match_entity":
        return await graph_store.resolve(call.args["name"])  # normalized unique by name
    if call.name == "expand_neighbors":
        return await graph_store.neighbors(call.args["entity_id"], depth=1)
    if call.name == "fetch_chunk":
        chunks = await chunk_store.by_entity(call.args["entity_id"])
        return [{"text": c.text, "source": c.tag} for c in chunks]
    if call.name == "extract_facts":
        facts = await llm_fact_extract(call.args["text"])  # ensure sources as [file:chunk:topic]
        return facts
```

### Citation Enforcement
- In prompts and validators, require source format [filename:chunk_index:topic] not entity labels.
- Reject/repair facts missing source tags.

### Tests
```python
async def test_tool_policy_enforced():
    with pytest.raises(AssertionError):
        await handle_tool_call(ToolCall(name="delete_node", args={}), state)

async def test_citation_format():
    facts = await llm_fact_extract("...text...")
    for f in facts:
        assert re.match(r"^.+:\d+:.*$", f['source'])
```

## Performance Considerations
- Batch neighbor expansions per step; cache action traces
- Use existing embedding batch APIs for entity-chunk mapping
- Respect hop limits and score thresholds to bound token usage
