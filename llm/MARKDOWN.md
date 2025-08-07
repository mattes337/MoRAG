# Intermediate Markdown Format Specifications

## Audio Files (.mp3, .wav, .m4a, .flac, .aac, .ogg)

### Structure
```markdown
# Audio Transcription: filename.mp3

## Audio Information
- **Duration**: 5.2 minutes
- **Speakers Detected**: 2
- **Topics Identified**: 3
- **Speaker Diarization**: Yes
- **Topic Segmentation**: Yes

# Introduction [0]
SPEAKER_00: Welcome to today's discussion about artificial intelligence.
SPEAKER_01: Thank you for having me on the show.

# Technology Overview [45]
SPEAKER_00: Let's start with the basics of machine learning.
SPEAKER_01: Machine learning is a subset of artificial intelligence.
SPEAKER_00: It enables computers to learn without explicit programming.

# Future Implications [120]
SPEAKER_01: The future of AI looks very promising.
SPEAKER_00: We're seeing applications in healthcare and education.
```

### Key Requirements
- Topic headers with single timestamp in seconds: `# Topic [timestamp]`
- Speaker labels: `SPEAKER_00:`, `SPEAKER_01:`, etc.
- One line per speaker utterance
- No separate transcript sections
- Timestamps mark topic boundaries, not individual lines

## Video Files (.mp4, .avi, .mov, .mkv, .webm)

### Structure
```markdown
# Video Analysis: filename.mp4

## Video Information
- **Duration**: 8.5 minutes
- **Resolution**: 1920x1080
- **Audio Tracks**: 1
- **Speakers Detected**: 1

# Introduction [0]
SPEAKER_00: Today we'll explore the fundamentals of neural networks.
SPEAKER_00: This presentation covers basic concepts and applications.

# Neural Network Basics [90]
SPEAKER_00: A neural network consists of interconnected nodes.
SPEAKER_00: Each node processes information and passes it forward.

# Applications [300]
SPEAKER_00: Neural networks are used in image recognition.
SPEAKER_00: They're also essential for natural language processing.
```

### Key Requirements
- Same format as audio for transcript sections
- Video-specific metadata in information section
- Audio content formatted as topics with timestamps
- Speaker-labeled dialogue format

## Document Files (.pdf, .docx, .txt, .md)

### Structure
```markdown
# Document: filename.pdf

## Document Information
- **Pages**: 15
- **Word Count**: 3,500
- **Document Type**: Research Paper
- **Language**: English

## Chapter 1: Introduction

This chapter introduces the fundamental concepts of machine learning and artificial intelligence. The field has evolved significantly over the past decade.

### Section 1.1: Background

Machine learning algorithms have become increasingly sophisticated. They now power many applications we use daily.

## Chapter 2: Methodology

The research methodology employed in this study follows established protocols. Data collection occurred over six months.

### Section 2.1: Data Collection

Participants were recruited from three universities. The sample size was 150 individuals.
```

### Key Requirements
- Preserve original document structure (chapters, sections)
- Maintain hierarchical headings
- Keep paragraphs intact
- Include page/chapter metadata where available

## Image Files (.jpg, .png, .gif, .bmp, .tiff)

### Structure
```markdown
# Image Analysis: filename.jpg

## Image Information
- **Dimensions**: 1920x1080
- **File Size**: 2.3 MB
- **Format**: JPEG
- **Color Space**: RGB

## Visual Content

The image shows a modern office environment with multiple workstations. There are approximately 12 people working at computers. The lighting is natural, coming from large windows on the left side.

## Text Content (OCR)

"Welcome to TechCorp"
"Innovation Through Technology"
"Established 2010"

## Objects Detected

- Computers: 12
- Chairs: 15
- Desks: 6
- Windows: 4
- People: 12
```

### Key Requirements
- Descriptive visual content analysis
- OCR text extraction in separate section
- Object detection results
- Technical metadata

## Web Content (.html, .url)

### Structure
```markdown
# Web Page: page-title

## Page Information
- **URL**: https://example.com/article
- **Title**: Article Title
- **Author**: John Doe
- **Published**: 2024-01-15
- **Word Count**: 1,200

## Main Content

The main article content appears here, cleaned of navigation elements and advertisements. Paragraphs are preserved as they appear in the original.

### Subsection Title

Subsection content follows the same formatting rules as document processing.

## Links

- [Related Article](https://example.com/related)
- [External Reference](https://external.com/ref)

## Metadata

- **Last Modified**: 2024-01-20
- **Language**: en
- **Keywords**: technology, innovation, AI
```

### Key Requirements
- Clean content extraction (no navigation/ads)
- Preserve article structure
- Extract and list important links
- Include web-specific metadata
