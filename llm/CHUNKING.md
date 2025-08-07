# Chunking Strategies by Content Type

## Audio/Video Content

### Strategy: Line-Based Topic Chunking
**Rule**: Always split by line, never inside a line
**Rationale**: Each line represents a speaker utterance with timestamp context

```markdown
# Introduction [0]
SPEAKER_00: Welcome to today's discussion.
SPEAKER_01: Thank you for having me.

# Main Topic [45]  
SPEAKER_00: Let's discuss machine learning.
SPEAKER_01: It's a fascinating field.
```

**Chunking Result**:
- Chunk 1: "SPEAKER_00: Welcome to today's discussion.\nSPEAKER_01: Thank you for having me."
- Chunk 2: "SPEAKER_00: Let's discuss machine learning.\nSPEAKER_01: It's a fascinating field."

**Configuration**:
- chunk_strategy: "topic_based"
- preserve_speaker_turns: true
- include_topic_headers: true
- max_chunk_size: 4000 characters
- respect_line_boundaries: true

## Document Content

### Strategy: Chapter/Page-Aware Semantic Chunking
**Rule**: Split by chapter/page boundaries when possible, maintain semantic coherence
**Rationale**: Preserve document structure while respecting size limits

```markdown
## Chapter 1: Introduction

This chapter introduces machine learning concepts. The field has evolved significantly over the past decade.

### Section 1.1: Background

Machine learning algorithms have become sophisticated. They power many daily applications.

## Chapter 2: Methodology

The research methodology follows established protocols.
```

**Chunking Logic**:
1. Prefer chapter boundaries
2. Fall back to section boundaries
3. Use semantic boundaries within sections
4. Add margin above/below target chunk size

**Configuration**:
- chunk_strategy: "semantic_with_structure"
- target_chunk_size: 4000 characters
- size_margin: 500 characters (3500-4500 range)
- preserve_headings: true
- min_chunk_size: 1000 characters

## Image Content

### Strategy: Section-Based Chunking
**Rule**: Split by content sections (visual, text, metadata)
**Rationale**: Different content types require different processing

```markdown
## Visual Content
The image shows a modern office environment...

## Text Content (OCR)
"Welcome to TechCorp"
"Innovation Through Technology"

## Objects Detected
- Computers: 12
- Chairs: 15
```

**Chunking Result**:
- Chunk 1: Visual description
- Chunk 2: OCR text content
- Chunk 3: Object detection results + metadata

**Configuration**:
- chunk_strategy: "section_based"
- preserve_sections: true
- combine_short_sections: true
- min_section_size: 200 characters

## Web Content

### Strategy: Article Structure Chunking
**Rule**: Follow article hierarchy, combine related paragraphs
**Rationale**: Maintain readability and context

```markdown
## Main Content
The main article discusses AI developments...

### Subsection Title
This subsection covers specific applications...

## Links
- [Related Article](url)
```

**Chunking Logic**:
1. Split by major sections
2. Group related paragraphs
3. Keep subsections together when possible
4. Separate metadata and links

**Configuration**:
- chunk_strategy: "hierarchical"
- target_chunk_size: 3000 characters
- preserve_hierarchy: true
- group_related_paragraphs: true

## Text Files

### Strategy: Paragraph-Based Semantic Chunking
**Rule**: Use paragraph boundaries, apply semantic analysis
**Rationale**: Natural text flow with intelligent boundaries

**Chunking Logic**:
1. Identify paragraph boundaries
2. Apply semantic boundary detection
3. Merge short paragraphs
4. Split long paragraphs at sentence boundaries

**Configuration**:
- chunk_strategy: "semantic"
- target_chunk_size: 4000 characters
- min_chunk_size: 1000 characters
- sentence_boundary_split: true

## Code Files

### Strategy: Function/Class Boundary Chunking
**Rule**: Split at function/class definitions, preserve complete units
**Rationale**: Maintain code structure and context

**Chunking Logic**:
1. Identify function/class boundaries
2. Keep complete functions together
3. Group related functions
4. Include necessary imports/context

**Configuration**:
- chunk_strategy: "structural"
- preserve_functions: true
- include_context: true
- max_chunk_size: 6000 characters

## Archive Content

### Strategy: File-Based Chunking
**Rule**: Process each extracted file separately
**Rationale**: Different files have different content types

**Chunking Logic**:
1. Extract archive contents
2. Process each file by its type
3. Maintain file hierarchy context
4. Cross-reference related files

**Configuration**:
- chunk_strategy: "file_based"
- preserve_hierarchy: true
- process_by_type: true
- include_file_context: true

## General Chunking Rules

### Word Boundary Preservation
- Never split words
- Respect sentence boundaries
- Maintain paragraph integrity where possible

### Size Constraints
- Default target: 4000 characters
- Minimum: 1000 characters
- Maximum: 6000 characters
- Margin: ±500 characters for semantic boundaries

### Context Preservation
- Include relevant headers/metadata
- Maintain topic continuity
- Preserve speaker/author attribution
- Keep timestamps and references intact
