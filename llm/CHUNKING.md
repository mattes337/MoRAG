# Chunking Strategies by Content Type

## Audio/Video Content

### Strategy: Line-Based Topic Chunking
**Rule**: Always split by line, never inside a line
**Rationale**: Each line represents a speaker utterance with precise timestamp context

**Content Format**: Audio and video content uses the format `[timecode][speaker] text` when both speaker diarization and topic segmentation are enabled. Lines may also be `[timecode] text` when only timestamps are available.

**Topic Organization**: When topic segmentation is enabled, content is organized under topic headers with timestamps in seconds (e.g., `# Introduction [0]`). When disabled, content flows continuously.

**Chunking Approach**:
- **With Topic Segmentation**: Split primarily at topic boundaries, keeping complete topics together when possible
- **Without Topic Segmentation**: Split at natural speaker turn boundaries or logical speech segments
- **Line Preservation**: Never split within a line to maintain timestamp and speaker context integrity

### Format Variants and Chunking

**With Speaker Diarization and Topic Segmentation**:
Content organized by topics with speaker-identified lines. Chunk at topic boundaries first, then at speaker turn boundaries if size limits require.

**With Speaker Diarization Only**:
Continuous content with speaker identification. Chunk at natural speaker turn boundaries or conversation breaks.

**With Topic Segmentation Only**:
Content organized by topics without speaker identification. Chunk primarily at topic boundaries.

**With Neither Feature**:
Simple timestamped transcript. Chunk at natural speech pauses or logical content breaks while preserving line integrity.

**Configuration**:
- chunk_strategy: "topic_based" (when topics available) or "line_based" (when topics disabled)
- preserve_speaker_turns: true
- include_topic_headers: true (when available)
- preserve_timestamps: true
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
