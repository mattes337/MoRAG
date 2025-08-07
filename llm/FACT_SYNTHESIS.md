# Fact Synthesis and Response Generation

## Fact Combination Strategies

### Fact Analysis for Synthesis
```python
def analyze_facts_for_synthesis(facts):
    """Analyze facts to guide synthesis and identify conflicts."""
    analysis = {
        'summary': '',
        'conflict_guidance': 'No significant conflicts detected'
    }
    
    # Analyze fact distribution by confidence
    high_confidence_facts = [f for f in facts if f.score >= 0.8]
    medium_confidence_facts = [f for f in facts if 0.5 <= f.score < 0.8]
    low_confidence_facts = [f for f in facts if f.score < 0.5]
    
    # Analyze fact types
    fact_types = {}
    for fact in facts:
        fact_type = fact.fact.fact_type.value
        fact_types[fact_type] = fact_types.get(fact_type, 0) + 1
    
    # Analyze source diversity
    all_sources = set()
    for fact in facts:
        all_sources.update(fact.sources)
    
    # Create summary
    summary_parts = [
        f"Total facts: {len(facts)}",
        f"High confidence (≥0.8): {len(high_confidence_facts)}",
        f"Medium confidence (0.5-0.8): {len(medium_confidence_facts)}",
        f"Low confidence (<0.5): {len(low_confidence_facts)}",
        f"Unique sources: {len(all_sources)}",
        f"Fact types: {', '.join([f'{k}({v})' for k, v in fact_types.items()])}"
    ]
    analysis['summary'] = " | ".join(summary_parts)
    
    # Detect potential conflicts
    conflicts = detect_fact_conflicts(facts)
    if conflicts:
        analysis['conflict_guidance'] = f"Address {len(conflicts)} potential conflicts by weighing evidence quality and source reliability"
    
    return analysis
```

### Conflict Detection
```python
def detect_fact_conflicts(facts):
    """Detect potential conflicts between facts."""
    conflicts = []
    
    for i, fact1 in enumerate(facts):
        for j, fact2 in enumerate(facts[i+1:], i+1):
            if are_facts_contradictory(fact1, fact2):
                conflicts.append({
                    'fact1_id': i,
                    'fact2_id': j,
                    'fact1_content': fact1.fact.content,
                    'fact2_content': fact2.fact.content,
                    'fact1_confidence': fact1.score,
                    'fact2_confidence': fact2.score,
                    'type': 'contradiction'
                })
    
    return conflicts

def are_facts_contradictory(fact1, fact2):
    """Simple contradiction detection based on content analysis."""
    # Check for opposing keywords
    opposing_pairs = [
        ("increases", "decreases"),
        ("effective", "ineffective"),
        ("safe", "unsafe"),
        ("approved", "rejected"),
        ("positive", "negative")
    ]
    
    content1 = fact1.fact.content.lower()
    content2 = fact2.fact.content.lower()
    
    for word1, word2 in opposing_pairs:
        if word1 in content1 and word2 in content2:
            return True
        if word2 in content1 and word1 in content2:
            return True
    
    return False
```

## Source Attribution Patterns

### Source Reference Creation
```python
def create_source_reference(fact, chunk_metadata):
    """Create detailed source reference for a fact."""
    # Extract document information
    document_id = chunk_metadata.get('document_id')
    chunk_id = chunk_metadata.get('chunk_id')
    
    # Extract page/timestamp information
    page_number = extract_page_number(chunk_metadata)
    timestamp = extract_timestamp(chunk_metadata)
    
    return SourceReference(
        document_id=document_id,
        document_title=chunk_metadata.get('title', f"Document {document_id}"),
        chunk_id=chunk_id,
        page_number=page_number,
        timestamp=timestamp,
        confidence=calculate_source_confidence(fact, chunk_metadata),
        url=chunk_metadata.get('url'),
        author=chunk_metadata.get('author'),
        publication_date=chunk_metadata.get('publication_date'),
        publisher=chunk_metadata.get('publisher'),
        metadata=chunk_metadata.get('metadata', {})
    )

def extract_page_number(metadata):
    """Extract page number from metadata."""
    # Try multiple possible keys
    for key in ['page_number', 'page', 'page_num']:
        if key in metadata:
            return metadata[key]
    
    # Try to extract from chunk text if it contains page markers
    chunk_text = metadata.get('text', '')
    page_match = re.search(r'\[Page (\d+)\]', chunk_text)
    if page_match:
        return int(page_match.group(1))
    
    return None

def extract_timestamp(metadata):
    """Extract timestamp from metadata."""
    # For audio/video content
    chunk_text = metadata.get('text', '')
    timestamp_match = re.search(r'\[(\d{1,2}:\d{2}:\d{2})\]', chunk_text)
    if timestamp_match:
        return timestamp_match.group(1)
    
    # Try metadata fields
    for key in ['timestamp', 'time', 'start_time']:
        if key in metadata:
            return metadata[key]
    
    return None
```

### Citation Integration
```python
class CitationStyle(Enum):
    INLINE = "inline"          # Citations within sentences
    FOOTNOTE = "footnote"      # Numbered footnotes
    ENDNOTE = "endnote"        # References at end
    PARENTHETICAL = "parenthetical"  # (Author, Year) style
    SUPERSCRIPT = "superscript"      # Superscript numbers

async def integrate_citations(response, facts, style=CitationStyle.INLINE):
    """Integrate citations into response content."""
    # Find citation points in content
    citation_matches = find_citation_points(response.content, facts)
    
    # Apply citation style
    if style == CitationStyle.INLINE:
        cited_content = integrate_inline_citations(
            response.content, citation_matches
        )
    elif style == CitationStyle.FOOTNOTE:
        cited_content = integrate_footnote_citations(
            response.content, citation_matches
        )
    elif style == CitationStyle.PARENTHETICAL:
        cited_content = integrate_parenthetical_citations(
            response.content, citation_matches
        )
    else:
        cited_content = response.content
    
    return cited_content

def find_citation_points(content, facts):
    """Find points in content where citations should be added."""
    matches = []
    
    for fact in facts:
        # Extract keywords from fact
        fact_keywords = extract_keywords(fact.fact.content)
        
        for keyword in fact_keywords:
            if len(keyword) < 3:  # Skip very short keywords
                continue
            
            # Find all occurrences of the keyword
            for match in re.finditer(re.escape(keyword.lower()), content.lower()):
                start, end = match.span()
                
                citation_match = CitationMatch(
                    fact_id=fact.fact.fact_id,
                    content_span=(start, end),
                    confidence=fact.score,
                    citation_text=format_citation_text(fact),
                    source_reference=fact.sources[0] if fact.sources else None
                )
                matches.append(citation_match)
    
    return matches
```

## Response Generation Templates

### Comprehensive Response Template
```python
def create_response_generation_prompt(query, facts, options):
    """Create enhanced prompt for LLM response generation."""
    # Analyze facts for conflicts and relationships
    fact_analysis = analyze_facts_for_synthesis(facts)
    
    # Format facts with enhanced context
    facts_text = format_facts_for_prompt(facts, fact_analysis)
    
    prompt = f"""You are an expert research assistant with advanced analytical capabilities. Generate a comprehensive, well-reasoned response to the user query based on the provided facts.

Query: {query}

Facts Analysis:
{fact_analysis['summary']}

Available Facts:
{facts_text}

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

Generate a response that demonstrates sophisticated understanding and synthesis of the provided information.

Format your response as JSON:
{{
  "content": "Main response content",
  "summary": "Brief summary of key findings",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "reasoning": "Explanation of reasoning process",
  "confidence_score": 0.85
}}"""
    
    return prompt
```

### Fact Formatting for Prompts
```python
def format_facts_for_prompt(facts, analysis):
    """Format facts with enhanced context for LLM prompt."""
    formatted_facts = []
    
    for i, fact in enumerate(facts):
        # Create fact entry with metadata
        fact_entry = f"""
Fact {i+1}:
Content: {fact.fact.content}
Confidence: {fact.score:.2f}
Type: {fact.fact.fact_type.value}
Sources: {len(fact.sources)} source(s)
"""
        
        # Add source information
        if fact.sources:
            source_info = []
            for source in fact.sources[:2]:  # Limit to top 2 sources
                source_desc = f"{source.document_title}"
                if source.page_number:
                    source_desc += f" (Page {source.page_number})"
                if source.timestamp:
                    source_desc += f" (Time {source.timestamp})"
                source_info.append(source_desc)
            
            fact_entry += f"Source Details: {'; '.join(source_info)}\n"
        
        formatted_facts.append(fact_entry)
    
    return "\n".join(formatted_facts)
```

## Answer Generation Patterns

### Structured Answer Generation
```python
async def generate_structured_answer(query, facts, language="en"):
    """Generate structured answer with proper citations."""
    
    # Filter and prepare facts
    prepared_facts = prepare_facts_for_synthesis(facts)
    
    # Create synthesis prompt
    prompt = f"""
User Question: "{query}"

Relevant Facts:
{format_facts_context(prepared_facts)}

Please synthesize these facts into a coherent, well-structured answer. Focus on directly addressing the user's question while incorporating the most relevant information from the facts. If the facts don't fully answer the question, acknowledge what information is available and what might be missing.

Structure your response with:
1. Direct answer to the question
2. Supporting evidence and details
3. Any limitations or caveats
4. Conclusion with key takeaways

Language: {language}
"""
    
    # Generate response using stronger LLM
    response = await stronger_llm_client.generate(
        prompt=prompt,
        max_tokens=4000,
        temperature=0.2
    )
    
    return response

def prepare_facts_for_synthesis(facts):
    """Prepare facts for synthesis by filtering and organizing."""
    # Sort by relevance and confidence
    sorted_facts = sorted(
        facts,
        key=lambda f: (f.relevance_to_query, f.score),
        reverse=True
    )
    
    # Group by fact type
    grouped_facts = {}
    for fact in sorted_facts:
        fact_type = fact.fact.fact_type.value
        if fact_type not in grouped_facts:
            grouped_facts[fact_type] = []
        grouped_facts[fact_type].append(fact)
    
    return grouped_facts
```

### Reference List Generation
```python
def generate_reference_list(facts):
    """Generate formatted reference list from fact sources."""
    references = {}
    
    for fact in facts:
        for source in fact.sources:
            ref_key = source.document_id
            if ref_key not in references:
                references[ref_key] = {
                    'title': source.document_title,
                    'author': source.author,
                    'publication_date': source.publication_date,
                    'publisher': source.publisher,
                    'url': source.url,
                    'pages': set(),
                    'timestamps': set()
                }
            
            # Collect page numbers and timestamps
            if source.page_number:
                references[ref_key]['pages'].add(source.page_number)
            if source.timestamp:
                references[ref_key]['timestamps'].add(source.timestamp)
    
    # Format references
    formatted_refs = []
    for i, (ref_key, ref_data) in enumerate(references.items(), 1):
        ref_text = f"{i}. {ref_data['title']}"
        
        if ref_data['author']:
            ref_text += f" by {ref_data['author']}"
        
        if ref_data['publication_date']:
            ref_text += f" ({ref_data['publication_date']})"
        
        if ref_data['pages']:
            pages = sorted(ref_data['pages'])
            ref_text += f", pages {', '.join(map(str, pages))}"
        
        if ref_data['timestamps']:
            timestamps = sorted(ref_data['timestamps'])
            ref_text += f", timestamps {', '.join(timestamps)}"
        
        if ref_data['url']:
            ref_text += f". Available at: {ref_data['url']}"
        
        formatted_refs.append(ref_text)
    
    return formatted_refs
```

## Quality Assessment

### Response Quality Metrics
```python
def assess_response_quality(response, facts, query):
    """Assess the quality of generated response."""
    metrics = {
        'completeness': 0.0,
        'accuracy': 0.0,
        'coherence': 0.0,
        'citation_coverage': 0.0,
        'overall_score': 0.0
    }
    
    # Completeness: How well does response address the query
    query_keywords = extract_keywords(query)
    response_keywords = extract_keywords(response.content)
    keyword_overlap = len(set(query_keywords) & set(response_keywords))
    metrics['completeness'] = min(1.0, keyword_overlap / len(query_keywords))
    
    # Accuracy: Based on fact confidence scores
    if facts:
        avg_fact_confidence = sum(f.score for f in facts) / len(facts)
        metrics['accuracy'] = avg_fact_confidence
    
    # Citation coverage: Percentage of facts referenced
    cited_facts = count_cited_facts(response.content, facts)
    if facts:
        metrics['citation_coverage'] = cited_facts / len(facts)
    
    # Coherence: Length and structure assessment
    word_count = len(response.content.split())
    if 50 <= word_count <= 500:  # Reasonable length
        metrics['coherence'] = 0.8
    else:
        metrics['coherence'] = 0.6
    
    # Overall score
    metrics['overall_score'] = (
        metrics['completeness'] * 0.3 +
        metrics['accuracy'] * 0.3 +
        metrics['coherence'] * 0.2 +
        metrics['citation_coverage'] * 0.2
    )
    
    return metrics
```
