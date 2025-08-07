# LLM Documentation vs Codebase Implementation Differences

This document analyzes the differences between the LLM documentation specifications and the current codebase implementation, organized as tasks that need to be addressed.

## Audio/Video Processing Format Differences

### [ ] Task: Implement [timecode][speaker] format for audio/video lines
**Current Implementation**: Uses separate timestamp and speaker formatting with spaces
- Current format: `[HH:MM:SS] **SPEAKER_XX**: text` or `**SPEAKER_XX (HH:MM:SS - HH:MM:SS)**: text`
- Expected format: `[MM:SS][SPEAKER_XX] text` or `[HH:MM:SS][SPEAKER_XX] text`
- **Location**: `packages/morag-audio/src/morag_audio/converters/audio_converter.py` lines 438-454
- **Action**: Modify `_format_segments()` method to use concatenated format without spaces

### [ ] Task: Implement per-line timestamps for audio/video content
**Current Implementation**: Groups segments by speaker with range timestamps
- Current: Groups consecutive segments by same speaker with start-end range
- Expected: Each line should have individual timestamp for precise timing
- **Location**: `packages/morag-audio/src/morag_audio/converters/audio_converter.py` lines 369-427
- **Action**: Disable speaker grouping by default and use individual segment timestamps

### [ ] Task: Update topic header format for audio/video
**Current Implementation**: Topic headers may not follow the specified format
- Expected format: `# Topic Name [timestamp_in_seconds]`
- **Location**: Topic segmentation and markdown conversion logic
- **Action**: Ensure topic headers include timestamp in seconds in square brackets

### [ ] Task: Implement structure variants based on enabled features
**Current Implementation**: Limited handling of different feature combinations
- Expected: Different output formats based on speaker diarization and topic segmentation settings
- **Location**: `packages/morag-audio/src/morag_audio/service.py` lines 153-157
- **Action**: Implement conditional formatting based on enabled features

## Video Processing Differences

### [ ] Task: Align video markdown format with audio specifications
**Current Implementation**: Video uses different formatting approach
- Current: Uses separate sections and different timestamp handling
- Expected: Same format as audio files with video-specific metadata
- **Location**: `packages/morag-video/src/morag_video/converters/video.py`
- **Action**: Align video transcript formatting with audio format specifications

### [ ] Task: Implement video structure variants
**Current Implementation**: Video processing doesn't implement the same structure variants as audio
- Expected: Same structure variants as audio (with/without speaker diarization and topic segmentation)
- **Location**: `packages/morag-video/src/morag_video/converters/video.py` lines 232-242
- **Action**: Implement the same structure variant logic as audio processing

## Chunking Strategy Differences

### [ ] Task: Update chunking to preserve new line format
**Current Implementation**: Chunking may not properly handle the new [timecode][speaker] format
- Expected: Respect line boundaries and preserve timestamp-speaker context
- **Location**: Chunking logic in ingestion coordinator
- **Action**: Update chunking strategies to handle the new line format properly

### [ ] Task: Implement topic-based chunking for audio/video
**Current Implementation**: May not properly chunk by topic boundaries
- Expected: Chunk primarily at topic boundaries when topic segmentation is enabled
- **Location**: `packages/morag/src/morag/ingestion_coordinator.py` lines 417-436
- **Action**: Implement topic-aware chunking that respects topic headers

## Configuration and Options Differences

### [ ] Task: Update AudioConversionOptions defaults
**Current Implementation**: Default options may not match documentation expectations
- Current: `group_by_speaker: bool = True` and `timestamp_format: str = "[%H:%M:%S]"`
- Expected: Individual line timestamps and [timecode][speaker] format
- **Location**: `packages/morag-audio/src/morag_audio/converters/audio_converter.py` lines 24-36
- **Action**: Update default options to match documentation specifications

### [ ] Task: Implement conditional feature handling
**Current Implementation**: Limited conditional logic based on enabled features
- Expected: Different output based on speaker diarization and topic segmentation settings
- **Location**: Audio and video service classes
- **Action**: Add conditional logic to handle different feature combinations

## Metadata and Header Differences

### [ ] Task: Standardize metadata section format
**Current Implementation**: Metadata format may not match documentation
- Expected: Specific metadata fields and format as documented
- **Location**: Audio and video conversion metadata sections
- **Action**: Ensure metadata sections match documented format

### [ ] Task: Implement proper header titles
**Current Implementation**: Headers may not follow exact format
- Expected: "Audio Transcription: filename.ext" and "Video Analysis: filename.ext"
- **Location**: Audio and video converters
- **Action**: Standardize header title formats

## Testing and Validation Tasks

### [ ] Task: Create tests for new format implementation
**Description**: Comprehensive tests for the new [timecode][speaker] format and structure variants
- **Action**: Create unit tests covering all format variants and feature combinations

### [ ] Task: Validate chunking with new format
**Description**: Ensure chunking works correctly with the new line format
- **Action**: Test chunking strategies with various audio/video content formats

### [ ] Task: Integration testing for format consistency
**Description**: End-to-end testing to ensure format consistency across the pipeline
- **Action**: Test complete ingestion and retrieval pipeline with new format
