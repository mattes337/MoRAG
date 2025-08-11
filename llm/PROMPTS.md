# LLM Prompt Templates

## Fact Extraction Prompts

### Base Fact Extraction Template
```
Extract structured facts from the following text. Focus on actionable, specific information that can be used to answer questions.

For each fact, provide:
- subject: The main entity or concept
- object: What the subject relates to or acts upon
- approach: How something is done or implemented
- solution: What problem is solved or benefit provided
- condition: Under what circumstances this applies
- remarks: Additional context or limitations
- fact_type: Type of fact (procedural, declarative, regulatory, etc.)
- confidence: 0.0-1.0 confidence score
- keywords: List of relevant technical terms

Domain: {domain}
Target Language: {language}

Text: {chunk_text}

Extract at most {max_facts} high-quality facts from the text.
```

### Domain-Specific Fact Extraction

**Research Domain Focus**:
- Research findings and results
- Methodologies and approaches
- Statistical data and measurements
- Experimental procedures
- Theoretical frameworks
- Limitations and future work

**Technical Domain Focus**:
- Implementation details and procedures
- Configuration and setup instructions
- Technical specifications
- Troubleshooting information
- Best practices and recommendations
- System requirements and dependencies

**Business Domain Focus**:
- Business processes and workflows
- Strategic decisions and rationale
- Performance metrics and KPIs
- Market analysis and insights
- Organizational structures and roles
- Financial information and projections

## Entity Extraction Prompts

### Entity Extraction Template
```
Extract and normalize entities from the text in order of appearance.
Focus on important entities like people, organizations, locations, concepts, and objects.
Use exact text for extractions. Do not paraphrase or overlap entities.
Provide meaningful attributes for each entity to add context.

For each entity:
1. Extract exact mention from text
2. Determine canonical form (normalized name)
3. Identify entity type and attributes
4. Assess confidence and provide disambiguation context

Return JSON array of entities with:
- name: Exact text mention
- canonical_name: Normalized form
- type: Entity type (PERSON, ORGANIZATION, CONCEPT, etc.)
- confidence: 0.0-1.0 confidence score
- attributes: Additional context and properties
- start_offset: Character position in text
- end_offset: End character position

Text: {text}
```

## Relation Extraction Prompts

### Relation Extraction Template
```
Extract meaningful relationships between entities in the given text.
Focus on semantic relationships that provide valuable connections.

For each relation, identify:
- source_entity: The entity that is the subject of the relation
- target_entity: The entity that is the object of the relation
- relation_type: Semantic relationship type (domain-specific)
- description: Natural language description of the relationship
- confidence: Your confidence in this extraction (0.0-1.0)
- context: The specific text snippet that supports this relation

Use domain-appropriate relation types:
- Medical: TREATS, CAUSES, PREVENTS, PRESCRIBES, INDICATES
- Business: PARTNERS_WITH, COMPETES_WITH, SUPPLIES, INVESTS_IN
- Technical: IMPLEMENTS, EXTENDS, DEPENDS_ON, INTEGRATES_WITH
- General: LOCATED_IN, PART_OF, CREATED_BY, RELATED_TO

Text: {text}
Entities: {entities}
```

## Reasoning and Path Selection Prompts

### Path Selection Template
```
You are an expert reasoning system. Analyze the available graph paths and select the most relevant ones for answering the query.

Query: {query}
Available Paths: {paths}

For each path, consider:
1. Relevance to the query topic
2. Semantic coherence of the path
3. Strength of entity connections
4. Potential to provide useful information

Select the top {max_paths} most promising paths and rank them by relevance.
Provide reasoning for your selections.

Return JSON with selected paths and relevance scores.
```

### Context Analysis Template
```
Analyze the current context to determine if it's sufficient to answer the query.

Query: {query}
Current Context:
- Entities: {entities}
- Documents: {documents}
- Relationships: {relationships}

Evaluate:
1. Is the context sufficient to answer the query? (true/false)
2. What is your confidence level? (0.0-1.0)
3. What information gaps exist?
4. What additional information would be helpful?

Return JSON:
{
  "is_sufficient": boolean,
  "confidence": float,
  "gaps": ["gap1", "gap2"],
  "reasoning": "explanation"
}
```

## Response Generation Prompts

### Response Synthesis Template
```
You are an expert research assistant with advanced analytical capabilities. Generate a comprehensive, well-reasoned response to the user query based on the provided facts.

Query: {query}

Facts Analysis:
{fact_analysis}

Available Facts:
{facts}

SYNTHESIS GUIDELINES:
- Prioritize high-confidence facts from reliable sources
- Address any conflicts by weighing evidence quality
- Maintain logical flow and coherence
- Include specific details and examples where relevant
- Acknowledge limitations or uncertainties
- Use transitional phrases to maintain flow

RESPONSE STRUCTURE GUIDELINES:
- Start with a clear, direct answer to the main query
- Present supporting evidence in logical order
- Address any nuances, exceptions, or conflicting information
- Conclude with synthesis and implications
- Use transitional phrases to maintain flow

Format your response as JSON:
{
  "content": "Main response content",
  "summary": "Brief summary of key findings",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "reasoning": "Explanation of reasoning process",
  "confidence_score": 0.85
}
```

## Fact Validation Prompts

### Fact Quality Assessment
```
Rate the quality of this extracted fact:

Fact: {fact_json}

Rate the fact on:
1. Specificity (0-1): Is it specific rather than generic?
2. Actionability (0-1): Does it provide useful, applicable information?
3. Completeness (0-1): Does it contain sufficient context?
4. Verifiability (0-1): Can it be traced to source text?

Respond with JSON:
{
  "overall_score": 0.0-1.0,
  "specificity": 0.0-1.0,
  "actionability": 0.0-1.0,
  "completeness": 0.0-1.0,
  "verifiability": 0.0-1.0,
  "issues": ["list of specific issues"],
  "suggestions": ["improvement suggestions"]
}
```

### Fact Relationship Analysis
```
Analyze relationships between these facts:

Facts: {facts_list}

Identify relationships like:
- SUPPORTS: One fact provides evidence for another
- ELABORATES: One fact provides more detail about another
- CONTRADICTS: Facts that present conflicting information
- SEQUENCE: Facts that represent steps in a process
- COMPARISON: Facts that compare different approaches/solutions
- CAUSATION: One fact describes the cause of another
- PREREQUISITE: Required condition/dependency
- ALTERNATIVE: Alternative approach/solution
- HIERARCHY: Parent-child or containment relationship

Respond with JSON array:
[
  {
    "source_fact_id": "fact_id_1",
    "target_fact_id": "fact_id_2",
    "relation_type": "SUPPORTS|ELABORATES|CONTRADICTS|...",
    "confidence": 0.0-1.0,
    "relationship_strength": "direct|inferred|contextual",
    "evidence_quality": "explicit|implicit|speculative",
    "context": "explanation of the relationship",
    "source_evidence": "text span supporting this relationship"
  }
]
```

## Source Attribution Prompts

### Source Mapping Template
```
Map the extracted facts to their supporting source chunks.

Query: {query}
Facts: {facts}
Source Chunks: {chunks}

Guidelines:
- A chunk supports a fact if it contains direct evidence, data, or statements that validate the fact
- A chunk may partially support a fact if it provides relevant context or background
- Be conservative - only map chunks that genuinely support the fact
- Provide clear reasoning for each mapping
- A fact may be supported by multiple chunks
- Some facts may not be supported by any specific chunk (if they are inferred or synthesized)

Return mappings with high confidence only for clear, direct support.
```

## Configuration Parameters

### Standard LLM Settings
```json
{
  "temperature": 0.1,
  "max_tokens": 2000,
  "max_retries": 5,
  "timeout": 30
}
```

### Domain-Specific Settings
```json
{
  "fact_extraction": {
    "temperature": 0.1,
    "max_tokens": 1500
  },
  "entity_extraction": {
    "temperature": 0.0,
    "max_tokens": 1000
  },
  "response_generation": {
    "temperature": 0.2,
    "max_tokens": 4000
  },
  "reasoning": {
    "temperature": 0.1,
    "max_tokens": 1000
  }
}
```
