# Task 13: HTML to Markdown Conversion

## Overview
Create a dedicated content conversion service that provides robust HTML to Markdown conversion with advanced features beyond the basic conversion already implemented in the web processor. This service will handle complex HTML structures, preserve formatting, and provide configurable conversion options.

## Objectives
- Create a dedicated `ContentConverter` service for HTML to Markdown conversion
- Support advanced conversion features (tables, code blocks, images, links)
- Provide configurable conversion options and templates
- Handle complex HTML structures and edge cases
- Support bidirectional conversion (HTML ↔ Markdown)
- Add content cleaning and normalization features
- Integrate with existing web processing pipeline

## Technical Requirements

### Core Components
1. **ContentConverter Service** (`src/morag/services/content_converter.py`)
   - HTML to Markdown conversion with advanced options
   - Markdown to HTML conversion
   - Content cleaning and normalization
   - Template-based conversion
   - Custom conversion rules

2. **Conversion Configuration** 
   - Configurable conversion options
   - Custom CSS selectors for content extraction
   - Template definitions for different content types
   - Output formatting preferences

3. **Content Processors**
   - Table processor for complex table structures
   - Code block processor with syntax highlighting preservation
   - Image processor with alt text and caption handling
   - Link processor with reference-style links
   - List processor for nested lists

### Features
- **Advanced HTML Parsing**: Handle complex nested structures
- **Smart Content Extraction**: Identify main content vs navigation/ads
- **Table Preservation**: Convert HTML tables to Markdown tables
- **Code Block Handling**: Preserve syntax highlighting information
- **Image Processing**: Handle images with captions and alt text
- **Link Management**: Support reference-style links and link validation
- **Content Cleaning**: Remove unwanted elements and normalize text
- **Template Support**: Use templates for consistent output formatting

## Implementation Plan

### Step 1: Create ContentConverter Service
```python
# src/morag/services/content_converter.py
class ContentConverter:
    def __init__(self, config: ConversionConfig = None)
    async def html_to_markdown(self, html: str, options: ConversionOptions = None) -> ConversionResult
    async def markdown_to_html(self, markdown: str, options: ConversionOptions = None) -> ConversionResult
    def clean_content(self, content: str, content_type: str) -> str
    def extract_main_content(self, html: str, selectors: List[str] = None) -> str
```

### Step 2: Configuration Classes
```python
@dataclass
class ConversionConfig:
    preserve_tables: bool = True
    preserve_code_blocks: bool = True
    preserve_images: bool = True
    reference_style_links: bool = False
    clean_whitespace: bool = True
    remove_empty_elements: bool = True
    custom_selectors: Dict[str, str] = field(default_factory=dict)

@dataclass
class ConversionOptions:
    heading_style: str = "ATX"  # ATX (#) or SETEXT (===)
    bullet_style: str = "-"     # -, *, +
    code_fence_style: str = "```"  # ``` or ~~~
    emphasis_style: str = "*"   # * or _
    strong_style: str = "**"    # ** or __
    link_style: str = "inline"  # inline or reference
```

### Step 3: Specialized Processors
```python
class TableProcessor:
    def convert_table(self, table_element: BeautifulSoup) -> str
    def handle_complex_tables(self, table: BeautifulSoup) -> str

class CodeBlockProcessor:
    def extract_code_blocks(self, soup: BeautifulSoup) -> List[CodeBlock]
    def preserve_syntax_highlighting(self, code_element: BeautifulSoup) -> str

class ImageProcessor:
    def process_images(self, soup: BeautifulSoup, base_url: str) -> List[ImageInfo]
    def handle_figure_captions(self, figure_element: BeautifulSoup) -> str
```

### Step 4: Integration with Web Processor
- Update `WebProcessor` to use `ContentConverter`
- Add conversion options to `WebScrapingConfig`
- Enhance markdown output quality

### Step 5: Testing
- Unit tests for each conversion feature
- Integration tests with real-world HTML samples
- Performance tests for large documents
- Edge case testing (malformed HTML, complex structures)

## Dependencies
```toml
# Add to pyproject.toml
markdownify = "^0.11.6"  # Already included
html2text = "^2024.2.26"  # Alternative converter
beautifulsoup4 = "^4.12.2"  # Already included
lxml = "^4.9.3"  # Fast XML/HTML parser
bleach = "^6.1.0"  # HTML sanitization
markdown = "^3.5.1"  # Markdown to HTML conversion
```

## Testing Strategy
1. **Unit Tests**: Test individual conversion functions
2. **Integration Tests**: Test with web processor integration
3. **Performance Tests**: Test with large HTML documents
4. **Edge Case Tests**: Test malformed HTML, complex structures
5. **Comparison Tests**: Compare output with other converters

## Success Criteria
- [ ] ContentConverter service implemented with all core features
- [ ] Advanced HTML to Markdown conversion working
- [ ] Markdown to HTML conversion working
- [ ] Table conversion preserves structure and formatting
- [ ] Code blocks preserve syntax highlighting information
- [ ] Images converted with proper alt text and captions
- [ ] Links converted with configurable styles
- [ ] Content cleaning removes unwanted elements
- [ ] Integration with WebProcessor completed
- [ ] All tests passing with >95% coverage
- [ ] Performance benchmarks meet requirements
- [ ] Documentation updated

## Files to Create/Modify
- `src/morag/services/content_converter.py` (new)
- `src/morag/processors/web.py` (modify - integrate ContentConverter)
- `tests/unit/test_content_converter.py` (new)
- `tests/integration/test_content_conversion_integration.py` (new)
- `scripts/test_content_conversion.py` (new)

## Notes
- Focus on preserving semantic meaning during conversion
- Handle edge cases gracefully (malformed HTML, missing elements)
- Provide fallback options for unsupported HTML elements
- Consider accessibility features (alt text, semantic markup)
- Support for custom conversion templates
- Integration with existing chunking and processing pipeline
