# Intermediate Markdown Format Specifications

## Audio Files (.mp3, .wav, .m4a, .flac, .aac, .ogg)

### Structure
Audio files are converted to structured markdown with the following components:

**Header Section**: Contains the title "Audio Transcription: filename.ext"

**Audio Information Section**: Includes metadata such as duration, number of speakers detected, number of topics identified, and whether speaker diarization and topic segmentation are enabled.

**Content Structure**: The main content is organized by topics when topic segmentation is enabled, or as a continuous transcript when disabled. Each topic section has a header with the topic name and timestamp in seconds (e.g., "# Introduction [0]").

**Line Format**: Each line contains a timestamp in [MM:SS] or [HH:MM:SS] format, followed by the speaker identifier (SPEAKER_00, SPEAKER_01, etc.), and then the spoken text. The format is: [timecode][speaker] text

**Timestamp Format**: Timestamps use the format [MM:SS] for content under one hour, or [HH:MM:SS] for longer content. Examples: [02:15], [01:23:45]. The timestamp represents the start time of that particular utterance or segment.

### Key Requirements
- Topic headers with single timestamp in seconds when topic segmentation is enabled: `# Topic Name [timestamp_in_seconds]`
- Each line includes individual timestamps for precise timing in [MM:SS] or [HH:MM:SS] format
- Speaker labels: SPEAKER_00, SPEAKER_01, etc. when speaker diarization is enabled
- Line format: `[timecode][speaker] text` when both features enabled, `[timecode] text` when only timestamps
- One line per speaker utterance or logical speech segment
- No separate transcript sections - all content is integrated into the topic structure
- Timestamps mark both topic boundaries (in headers) and individual utterances (per line)

### Structure Variants

**With Speaker Diarization and Topic Segmentation Enabled**:
Content is organized by topics with headers containing timestamps in seconds. Each line includes individual timestamps and speaker identification in the format `[MM:SS][SPEAKER_XX] text` or `[HH:MM:SS][SPEAKER_XX] text`.

**With Speaker Diarization Only (No Topic Segmentation)**:
Content flows continuously without topic headers. Each line includes timestamps and speaker identification in the format `[MM:SS][SPEAKER_XX] text` or `[HH:MM:SS][SPEAKER_XX] text`.

**With Topic Segmentation Only (No Speaker Diarization)**:
Content is organized by topics with headers containing timestamps in seconds. Lines contain only timestamps and text in the format `[MM:SS] text` or `[HH:MM:SS] text`.

**With Neither Feature Enabled**:
Content flows as a simple timestamped transcript with lines in the format `[MM:SS] text` or `[HH:MM:SS] text`, without topic organization or speaker identification.

## Video Files (.mp4, .avi, .mov, .mkv, .webm)

### Structure
Video files follow the same transcription format as audio files, with additional video-specific metadata.

**Header Section**: Contains the title "Video Analysis: filename.ext"

**Video Information Section**: Includes video-specific metadata such as duration, resolution, audio tracks, number of speakers detected, and processing options enabled.

**Content Structure**: Identical to audio files - organized by topics when topic segmentation is enabled, or as continuous transcript when disabled. Audio content from the video is formatted with the same timestamp and speaker structure.

### Key Requirements
- Same format as audio for transcript sections
- Video-specific metadata in information section (resolution, audio tracks, etc.)
- Audio content formatted identically to audio files
- Same `[timecode][speaker] text` format for dialogue when both features enabled
- Same structure variants apply as for audio files

### Structure Variants

**With Speaker Diarization and Topic Segmentation Enabled**:
Content is organized by topics with headers containing timestamps in seconds. Each line includes individual timestamps and speaker identification in the format `[MM:SS][SPEAKER_XX] text` or `[HH:MM:SS][SPEAKER_XX] text`.

**With Speaker Diarization Only (No Topic Segmentation)**:
Content flows continuously without topic headers. Each line includes timestamps and speaker identification in the format `[MM:SS][SPEAKER_XX] text` or `[HH:MM:SS][SPEAKER_XX] text`.

**With Topic Segmentation Only (No Speaker Diarization)**:
Content is organized by topics with headers containing timestamps in seconds. Lines contain only timestamps and text in the format `[MM:SS] text` or `[HH:MM:SS] text`.

**With Neither Feature Enabled**:
Content flows as a simple timestamped transcript with lines in the format `[MM:SS] text` or `[HH:MM:SS] text`, without topic organization or speaker identification.

## Document Files (.pdf, .docx, .txt, .md)

### Structure
Document files are converted to structured markdown preserving the original document hierarchy and formatting.

**Header Section**: Contains the title "Document: filename.ext"

**Document Information Section**: Includes metadata such as page count, word count, document type, and language.

**Content Structure**: Preserves the original document structure including chapters, sections, and subsections. Hierarchical headings are maintained using appropriate markdown heading levels. Paragraphs are kept intact to preserve readability and context.

### Key Requirements
- Preserve original document structure (chapters, sections, subsections)
- Maintain hierarchical headings using appropriate markdown levels
- Keep paragraphs intact to preserve context and readability
- Include page/chapter metadata where available
- Maintain formatting elements like lists, tables, and emphasis

## Image Files (.jpg, .png, .gif, .bmp, .tiff)

### Structure
Image files are analyzed and converted to structured markdown containing visual analysis, text extraction, and metadata.

**Header Section**: Contains the title "Image Analysis: filename.ext"

**Image Information Section**: Includes technical metadata such as dimensions, file size, format, and color space.

**Visual Content Section**: Contains a descriptive analysis of the visual elements in the image, including objects, people, settings, lighting, and composition.

**Text Content (OCR) Section**: Contains any text extracted from the image using optical character recognition, presented as quoted strings or structured text.

**Objects Detected Section**: Lists detected objects with counts or descriptions, typically generated by computer vision analysis.

### Key Requirements
- Descriptive visual content analysis covering all significant elements
- OCR text extraction in a separate, clearly marked section
- Object detection results with counts or descriptions
- Technical metadata including file properties
- Clear separation between different types of analysis

## Web Content (.html, .url)

### Structure
Web content is extracted and converted to clean, structured markdown preserving the original article hierarchy.

**Header Section**: Contains the title "Web Page: page-title" where page-title is extracted from the HTML title tag or main heading.

**Page Information Section**: Includes web-specific metadata such as URL, title, author, publication date, and word count.

**Main Content Section**: Contains the primary article content with navigation elements, advertisements, and extraneous content removed. The original paragraph structure and formatting are preserved.

**Subsection Structure**: Subsections maintain their hierarchical structure using appropriate markdown heading levels, following the same formatting rules as document processing.

**Links Section**: Contains important links extracted from the content, presented as a bulleted list with descriptive text and URLs.

**Metadata Section**: Additional metadata such as last modified date, language, keywords, and other relevant information.

### Key Requirements
- Clean content extraction removing navigation, advertisements, and non-content elements
- Preserve original article structure and hierarchy
- Extract and list important links in a separate section
- Include comprehensive web-specific metadata
- Maintain readability and formatting of the original content
