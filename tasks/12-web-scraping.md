# Task 12: Web Scraping Implementation

## Overview
Implement comprehensive web scraping capabilities to extract and process content from websites. This task builds upon the existing document processing pipeline to handle web content extraction, cleaning, and conversion to structured format.

## Objectives
- [x] Create web content processor with URL validation
- [x] Implement HTML parsing and content extraction
- [x] Add HTML to Markdown conversion
- [x] Extract metadata from web pages
- [x] Implement respectful scraping with rate limiting
- [x] Add comprehensive error handling
- [x] Create async task integration
- [x] Add comprehensive test coverage

## Implementation Steps

### Step 1: Web Processor Core
Create `src/morag/processors/web.py` with:
- URL validation and normalization
- HTTP client with proper headers and timeouts
- Content type detection and handling
- Basic HTML parsing with BeautifulSoup4

### Step 2: Content Extraction
Implement content extraction features:
- Main content identification
- Navigation and boilerplate removal
- Text extraction and cleaning
- Link extraction and processing

### Step 3: Metadata Extraction
Add metadata extraction capabilities:
- Page title and description
- Meta tags (keywords, author, etc.)
- Open Graph and Twitter Card data
- Structured data (JSON-LD, microdata)

### Step 4: HTML to Markdown Conversion
Implement conversion features:
- Clean HTML to Markdown transformation
- Table and list preservation
- Image and link handling
- Code block preservation

### Step 5: Error Handling & Rate Limiting
Add robust error handling:
- Network timeout and retry logic
- HTTP status code handling
- Content type validation
- Rate limiting for respectful scraping

### Step 6: Task Integration
Update web tasks:
- Replace placeholder implementation
- Add progress tracking
- Integrate with chunking pipeline
- Add result formatting

### Step 7: Testing
Create comprehensive tests:
- Unit tests for web processor
- Integration tests for web tasks
- Mock HTTP responses for testing
- Edge case handling tests

## Dependencies
- `httpx` - Async HTTP client
- `beautifulsoup4` - HTML parsing
- `markdownify` - HTML to Markdown conversion
- `urllib.parse` - URL handling
- `aiofiles` - Async file operations

## Configuration
```python
@dataclass
class WebScrapingConfig:
    timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    user_agent: str = "MoRAG/1.0 (+https://github.com/yourusername/morag)"
    max_content_length: int = 10 * 1024 * 1024  # 10MB
    allowed_content_types: List[str] = field(default_factory=lambda: [
        'text/html', 'application/xhtml+xml'
    ])
    extract_links: bool = True
    convert_to_markdown: bool = True
    clean_content: bool = True
```

## Data Models
```python
@dataclass
class WebContent:
    url: str
    title: str
    content: str
    markdown_content: str
    metadata: Dict[str, Any]
    links: List[str]
    images: List[str]
    extraction_time: float
    content_length: int
    content_type: str

@dataclass
class WebScrapingResult:
    url: str
    content: WebContent
    chunks: List[DocumentChunk]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
```

## Testing Strategy
1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test full pipeline
3. **Mock Tests**: Use mock HTTP responses
4. **Error Tests**: Test error handling scenarios
5. **Performance Tests**: Test with large content

## Success Criteria
- [ ] Successfully extract content from various website types
- [ ] Convert HTML to clean Markdown format
- [ ] Extract comprehensive metadata
- [ ] Handle errors gracefully
- [ ] Maintain >95% test coverage
- [ ] Process content within reasonable time limits
- [ ] Integrate seamlessly with existing pipeline

## Implementation Notes
- Use async/await for all HTTP operations
- Implement proper rate limiting to be respectful
- Handle various HTML structures and edge cases
- Preserve important formatting in Markdown conversion
- Extract and preserve link relationships
- Support both single page and batch processing

## Next Steps
After completion, this enables:
- Task 13: Content Conversion (enhanced HTML processing)
- Task 14: Gemini Integration (web content embeddings)
- Task 17: Ingestion API (web URL endpoints)
