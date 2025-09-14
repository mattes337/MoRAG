# Fact Synthesis and Response Generation

## Fact Combination Strategies

### Fact Analysis for Synthesis
**Purpose**: Analyze facts to guide synthesis process and identify potential conflicts.

**Analysis Components**:
- **Confidence Distribution**: Categorize facts by confidence levels (high ≥0.8, medium 0.5-0.8, low <0.5)
- **Fact Type Analysis**: Count and categorize facts by type for balanced synthesis
- **Source Diversity**: Track unique sources to assess information breadth
- **Conflict Detection**: Identify potential contradictions between facts
- **Summary Generation**: Create comprehensive analysis summary for LLM guidance

### Conflict Detection
**Approach**: Identify potential contradictions between facts using content analysis.

**Detection Method**:
- **Pairwise Comparison**: Compare each fact against all other facts
- **Keyword Analysis**: Look for opposing keyword pairs in fact content
- **Opposing Pairs**: Common contradictory terms like increases/decreases, effective/ineffective, safe/unsafe, approved/rejected, positive/negative
- **Conflict Recording**: Document conflicting facts with IDs, content, confidence scores, and conflict type

## Source Attribution Patterns

### Source Reference Creation
**Purpose**: Create detailed source references for facts with comprehensive metadata.

**Reference Components**:
- **Document Information**: Document ID, title, chunk ID
- **Location Data**: Page numbers for documents, timestamps for audio/video
- **Confidence Scoring**: Calculate source confidence based on fact and metadata
- **Publication Details**: URL, author, publication date, publisher
- **Additional Metadata**: Extended metadata for context

**Extraction Methods**:
- **Page Numbers**: Check multiple metadata keys (page_number, page, page_num) or extract from text markers
- **Timestamps**: Extract from audio/video content using regex patterns or metadata fields
- **Fallback Handling**: Graceful handling when information is not available

### Citation Integration
**Citation Styles**:
- **INLINE**: Citations within sentences
- **FOOTNOTE**: Numbered footnotes
- **ENDNOTE**: References at end
- **PARENTHETICAL**: (Author, Year) style
- **SUPERSCRIPT**: Superscript numbers

**Integration Process**:
1. **Citation Point Detection**: Find locations in content where citations should be added
2. **Style Application**: Apply selected citation style to identified points
3. **Content Integration**: Merge citations into response content
4. **Format Consistency**: Ensure consistent citation formatting throughout

**Citation Point Finding**:
- **Keyword Extraction**: Extract keywords from fact content
- **Content Matching**: Find keyword occurrences in response content
- **Citation Mapping**: Create citation matches with fact IDs, spans, confidence, and source references
- **Filtering**: Skip very short keywords to avoid over-citation

## Response Generation Templates

### Comprehensive Response Template
**Purpose**: Create enhanced prompt for LLM response generation with fact analysis and synthesis guidelines.

**Template Components**:
- **Expert Role Definition**: Position LLM as expert research assistant
- **Query Context**: Include user query and fact analysis summary
- **Fact Presentation**: Format facts with enhanced context and metadata
- **Synthesis Guidelines**: Prioritize high-confidence facts, address conflicts, maintain logical flow
- **Structure Guidelines**: Clear answer, supporting evidence, nuances, synthesis conclusion
- **Output Format**: Structured JSON with content, summary, key points, reasoning, confidence score

**Process**:
1. **Fact Analysis**: Analyze facts for conflicts and relationships
2. **Context Formatting**: Format facts with enhanced context for LLM
3. **Prompt Construction**: Build comprehensive prompt with guidelines
4. **Response Generation**: Generate structured response following guidelines

### Fact Formatting for Prompts
**Purpose**: Format facts with enhanced context and metadata for optimal LLM processing.

**Formatting Structure**:
- **Fact Numbering**: Sequential numbering for reference
- **Core Information**: Content, confidence score, fact type
- **Source Count**: Number of supporting sources
- **Source Details**: Document titles with page numbers or timestamps (limited to top 2 sources)
- **Structured Layout**: Clear separation and organization for LLM comprehension

**Enhancement Features**:
- **Confidence Display**: Two decimal precision for confidence scores
- **Source Attribution**: Detailed source information with location data
- **Metadata Integration**: Include relevant metadata for context

## Answer Generation Patterns

### Structured Answer Generation
**Purpose**: Generate comprehensive, well-structured answers with proper citations and fact synthesis.

**Process**:
1. **Fact Preparation**: Filter and organize facts by relevance and confidence
2. **Prompt Creation**: Build synthesis prompt with structured guidelines
3. **LLM Generation**: Use stronger LLM with appropriate parameters (4000 tokens, 0.2 temperature)
4. **Response Structure**: Direct answer, supporting evidence, limitations, conclusion

**Fact Preparation Steps**:
- **Sorting**: Order facts by relevance to query and confidence score
- **Grouping**: Organize facts by type for balanced synthesis
- **Filtering**: Select most relevant facts for synthesis

**Response Guidelines**:
- **Direct Addressing**: Focus on directly answering the user's question
- **Evidence Integration**: Incorporate most relevant fact information
- **Limitation Acknowledgment**: Note missing information or gaps
- **Language Support**: Generate responses in specified language

### Reference List Generation
**Purpose**: Generate formatted reference list from fact sources with comprehensive citation information.

**Process**:
1. **Source Aggregation**: Collect unique sources from all facts
2. **Reference Compilation**: Build reference entries with title, author, publication details
3. **Location Collection**: Aggregate page numbers and timestamps for each source
4. **Formatting**: Create numbered reference list with consistent formatting

**Reference Components**:
- **Basic Information**: Title, author, publication date, publisher
- **Location Data**: Page numbers (sorted), timestamps (sorted)
- **Access Information**: URLs when available
- **Numbering**: Sequential numbering for easy reference

**Formatting Rules**:
- **Title First**: Start with document title
- **Author Attribution**: Include author when available
- **Date Parentheses**: Publication date in parentheses
- **Location Details**: Pages and timestamps as comma-separated lists
- **URL Suffix**: Available at URL when provided

## Quality Assessment

### Response Quality Metrics
**Purpose**: Assess the quality of generated responses using multiple evaluation criteria.

**Quality Metrics**:
- **Completeness** (30% weight): How well response addresses the query based on keyword overlap
- **Accuracy** (30% weight): Based on average confidence scores of supporting facts
- **Coherence** (20% weight): Length and structure assessment (optimal 50-500 words)
- **Citation Coverage** (20% weight): Percentage of facts properly referenced in response

**Assessment Process**:
1. **Keyword Analysis**: Compare query and response keywords for completeness
2. **Confidence Evaluation**: Calculate average fact confidence for accuracy
3. **Citation Counting**: Count facts properly cited in response
4. **Structure Analysis**: Evaluate response length and organization
5. **Overall Scoring**: Weighted combination of all metrics

**Scoring Guidelines**:
- **Completeness**: Keyword overlap ratio (capped at 1.0)
- **Accuracy**: Average fact confidence score
- **Citation Coverage**: Ratio of cited facts to total facts
- **Coherence**: 0.8 for optimal length (50-500 words), 0.6 otherwise
