# Task 29: Enhanced Web Content to Markdown Conversion

## Status: FRAMEWORK READY ✅
**Note**: The universal document conversion framework (Task 24) has been implemented with basic web conversion. This task focuses on adding advanced features like JavaScript rendering and intelligent content detection.

## Objective
Enhance the existing web converter in the universal document conversion framework with advanced content extraction, JavaScript rendering, multi-page crawling, and intelligent content detection capabilities.

## Research Phase

### Web Scraping and Content Extraction Libraries
1. **Playwright** (Recommended for dynamic content)
   - Pros: Handles JavaScript, modern web apps, multiple browsers
   - Cons: Heavier resource usage, slower than static scrapers
   
2. **Selenium** (Alternative for dynamic content)
   - Pros: Mature, extensive browser support, good documentation
   - Cons: Slower, more complex setup, resource intensive
   
3. **Beautiful Soup** (Current implementation - static content)
   - Pros: Fast, lightweight, excellent HTML parsing
   - Cons: No JavaScript support, limited to static content
   
4. **Scrapy** (For large-scale scraping)
   - Pros: High performance, built-in handling of robots.txt, rate limiting
   - Cons: Overkill for simple extraction, learning curve

### Content Extraction and Cleaning
1. **Readability** - Extract main content from web pages
2. **Trafilatura** - Web content extraction optimized for text
3. **newspaper3k** - Article extraction and processing
4. **Mercury Parser** - Content extraction API

### HTML to Markdown Conversion
1. **html2text** (Current implementation)
2. **markdownify** - More configurable HTML to markdown
3. **pandoc** - Universal document converter
4. **turndown** (JavaScript-based, via Node.js)

## Implementation Strategy

### Phase 1: Enhanced Web Content Extractor
```python
from playwright.async_api import async_playwright
import asyncio
from readability import Document
import trafilatura

class EnhancedWebToMarkdownConverter(BaseConverter):
    def __init__(self):
        self.static_extractor = StaticWebExtractor()
        self.dynamic_extractor = DynamicWebExtractor()
        self.content_cleaner = WebContentCleaner()
        self.markdown_formatter = WebMarkdownFormatter()
        self.url_analyzer = URLAnalyzer()
    
    async def convert(self, url: str, options: ConversionOptions) -> ConversionResult:
        # Analyze URL to determine extraction strategy
        url_info = await self.url_analyzer.analyze_url(url)
        
        # Choose extraction method based on content type
        if url_info.requires_javascript or options.force_dynamic:
            content_data = await self.dynamic_extractor.extract(url, options)
        else:
            content_data = await self.static_extractor.extract(url, options)
        
        # Clean and structure content
        cleaned_content = await self.content_cleaner.clean_content(content_data, options)
        
        # Convert to structured markdown
        markdown_content = await self.markdown_formatter.format(cleaned_content, options)
        
        return ConversionResult(
            content=markdown_content,
            metadata=self.extract_web_metadata(content_data, url),
            quality_score=self.assess_extraction_quality(content_data, markdown_content)
        )
```

### Phase 2: Dynamic Content Extraction with Playwright
```python
class DynamicWebExtractor:
    def __init__(self):
        self.browser_pool = BrowserPool()
        self.content_detector = ContentDetector()
    
    async def extract(self, url: str, options: ConversionOptions) -> WebContentData:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            try:
                page = await browser.new_page()
                
                # Set viewport and user agent
                await page.set_viewport_size({"width": 1920, "height": 1080})
                await page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (compatible; MoRAG-Bot/1.0)"
                })
                
                # Navigate to page with timeout
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Wait for dynamic content to load
                if options.wait_for_selector:
                    await page.wait_for_selector(options.wait_for_selector, timeout=10000)
                elif options.wait_time:
                    await page.wait_for_timeout(options.wait_time * 1000)
                else:
                    # Smart waiting - detect when content stops changing
                    await self.wait_for_content_stability(page)
                
                # Extract content
                content_data = await self.extract_page_content(page, options)
                
                return content_data
                
            finally:
                await browser.close()
    
    async def wait_for_content_stability(self, page, max_wait=10000, check_interval=500):
        """Wait until page content stops changing"""
        previous_content = ""
        stable_count = 0
        
        for _ in range(max_wait // check_interval):
            current_content = await page.content()
            
            if current_content == previous_content:
                stable_count += 1
                if stable_count >= 3:  # Content stable for 3 checks
                    break
            else:
                stable_count = 0
                previous_content = current_content
            
            await page.wait_for_timeout(check_interval)
    
    async def extract_page_content(self, page, options: ConversionOptions) -> WebContentData:
        # Extract main content using multiple strategies
        content_strategies = []
        
        # Strategy 1: Use readability algorithm
        html_content = await page.content()
        doc = Document(html_content)
        main_content = doc.summary()
        content_strategies.append(("readability", main_content))
        
        # Strategy 2: Extract by semantic elements
        semantic_content = await self.extract_semantic_content(page)
        content_strategies.append(("semantic", semantic_content))
        
        # Strategy 3: Custom selectors if provided
        if options.content_selectors:
            custom_content = await self.extract_by_selectors(page, options.content_selectors)
            content_strategies.append(("custom", custom_content))
        
        # Choose best extraction result
        best_content = self.select_best_content(content_strategies)
        
        # Extract metadata
        metadata = await self.extract_page_metadata(page)
        
        # Extract images if requested
        images = []
        if options.extract_images:
            images = await self.extract_images(page, options)
        
        return WebContentData(
            url=page.url,
            title=await page.title(),
            content=best_content,
            metadata=metadata,
            images=images,
            extraction_method="dynamic"
        )
```

### Phase 3: Intelligent Content Cleaning
```python
class WebContentCleaner:
    def __init__(self):
        self.noise_detector = NoiseDetector()
        self.structure_analyzer = ContentStructureAnalyzer()
    
    async def clean_content(self, content_data: WebContentData, options: ConversionOptions) -> CleanedContent:
        # Parse HTML content
        soup = BeautifulSoup(content_data.content, 'html.parser')
        
        # Remove noise elements
        if options.remove_noise:
            soup = await self.remove_noise_elements(soup, options)
        
        # Extract and structure content
        structured_content = await self.structure_analyzer.analyze_structure(soup)
        
        # Clean text content
        cleaned_text = await self.clean_text_content(structured_content, options)
        
        return CleanedContent(
            title=content_data.title,
            structured_content=structured_content,
            cleaned_text=cleaned_text,
            metadata=content_data.metadata,
            images=content_data.images
        )
    
    async def remove_noise_elements(self, soup, options: ConversionOptions):
        # Remove common noise elements
        noise_selectors = [
            'nav', 'header', 'footer', 'aside',
            '.advertisement', '.ads', '.sidebar',
            '.social-share', '.comments', '.related-posts',
            'script', 'style', 'noscript'
        ]
        
        # Add custom noise selectors
        if options.noise_selectors:
            noise_selectors.extend(options.noise_selectors)
        
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove elements with low text-to-HTML ratio
        for element in soup.find_all():
            if self.noise_detector.is_likely_noise(element):
                element.decompose()
        
        return soup
```

### Phase 4: Advanced Markdown Formatting
```python
class WebMarkdownFormatter:
    def __init__(self):
        self.html_converter = AdvancedHTMLToMarkdown()
        self.structure_formatter = ContentStructureFormatter()
    
    async def format(self, cleaned_content: CleanedContent, options: ConversionOptions) -> str:
        sections = []
        
        # Add page header
        sections.append(self.create_page_header(cleaned_content))
        
        # Add table of contents if requested
        if options.include_toc:
            toc = self.generate_table_of_contents(cleaned_content.structured_content)
            sections.append(toc)
        
        # Convert main content
        main_markdown = await self.html_converter.convert(
            cleaned_content.cleaned_text, 
            options.conversion_options
        )
        sections.append(main_markdown)
        
        # Add images section
        if cleaned_content.images and options.include_images:
            images_section = self.format_images_section(cleaned_content.images)
            sections.append(images_section)
        
        # Add metadata section
        if options.include_metadata:
            metadata_section = self.format_metadata_section(cleaned_content.metadata)
            sections.append(metadata_section)
        
        return "\n\n".join(sections)
```

## Structured Web Content Markdown Output

### Web Page Template
```markdown
# {page_title}

**URL**: {original_url}
**Extracted**: {extraction_timestamp}
**Method**: {extraction_method}
**Quality Score**: {quality_score}

## Page Summary
{ai_generated_summary}

## Table of Contents
- [Introduction](#introduction)
- [Main Content](#main-content)
- [Key Points](#key-points)
- [Images](#images)
- [Metadata](#metadata)

## Main Content

### Introduction
{introduction_content}

### Section 1: {section_heading}
{section_content}

#### Subsection 1.1
{subsection_content}

### Key Points
- Point 1: {extracted_key_point}
- Point 2: {extracted_key_point}
- Point 3: {extracted_key_point}

## Images

### Image 1
![{alt_text}]({image_url})
**Caption**: {image_caption}
**Context**: {surrounding_text}

## Links and References
- [External Link 1]({url}) - {link_description}
- [External Link 2]({url}) - {link_description}

## Metadata
- **Author**: {author}
- **Published**: {publish_date}
- **Modified**: {modified_date}
- **Language**: {detected_language}
- **Word Count**: {word_count}
- **Reading Time**: {estimated_reading_time}
- **Domain**: {domain}
- **Content Type**: {content_type}
```

### Configuration Options
```yaml
web_conversion:
  extraction:
    method: "auto"  # static, dynamic, auto
    wait_for_selector: null
    wait_time: 3  # seconds
    timeout: 30  # seconds
    
  content_selection:
    use_readability: true
    content_selectors: []
    exclude_selectors: [".ads", ".sidebar", "nav"]
    min_content_length: 100
    
  cleaning:
    remove_noise: true
    remove_navigation: true
    remove_ads: true
    remove_social_widgets: true
    preserve_links: true
    preserve_images: true
    
  conversion:
    include_toc: true
    include_metadata: true
    include_images: true
    include_links: true
    convert_tables: true
    preserve_formatting: true
    
  quality:
    min_text_ratio: 0.3
    min_word_count: 50
    require_title: true
```

## Integration with MoRAG System

### Enhanced Web Processing Service
```python
# Update services/web_scraper.py
class WebScraper:
    def __init__(self):
        self.converter = EnhancedWebToMarkdownConverter()
        self.chunker = WebContentChunker()
        self.embedder = EmbeddingService()
        self.url_validator = URLValidator()
    
    async def scrape_and_process(self, url: str, options: ProcessingOptions) -> ProcessedWebContent:
        # Validate URL
        if not await self.url_validator.is_valid_url(url):
            raise InvalidURLError(f"Invalid or inaccessible URL: {url}")
        
        # Convert to markdown
        conversion_result = await self.converter.convert(url, options.conversion)
        
        # Create chunks based on content structure
        chunks = await self.chunker.create_semantic_chunks(
            conversion_result.content,
            options.chunking
        )
        
        # Generate embeddings for each chunk
        for chunk in chunks:
            chunk.embedding = await self.embedder.embed_text(chunk.content)
        
        return ProcessedWebContent(
            url=url,
            markdown_content=conversion_result.content,
            chunks=chunks,
            metadata=conversion_result.metadata,
            quality_score=conversion_result.quality_score
        )
```

### Database Schema Updates
```sql
-- Web-specific metadata
ALTER TABLE documents ADD COLUMN source_url TEXT;
ALTER TABLE documents ADD COLUMN domain VARCHAR(255);
ALTER TABLE documents ADD COLUMN extraction_method VARCHAR(50);
ALTER TABLE documents ADD COLUMN page_title TEXT;
ALTER TABLE documents ADD COLUMN author VARCHAR(255);
ALTER TABLE documents ADD COLUMN publish_date TIMESTAMP;
ALTER TABLE documents ADD COLUMN word_count INTEGER;
ALTER TABLE documents ADD COLUMN reading_time INTEGER; -- minutes

-- Web page structure
CREATE TABLE web_page_structure (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    element_type VARCHAR(50), -- heading, paragraph, list, table, image
    element_level INTEGER,
    element_order INTEGER,
    content TEXT,
    xpath VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted links
CREATE TABLE web_page_links (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    link_url TEXT,
    link_text VARCHAR(500),
    link_type VARCHAR(50), -- internal, external, anchor
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted images
CREATE TABLE web_page_images (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    image_url TEXT,
    alt_text TEXT,
    caption TEXT,
    context TEXT,
    image_path VARCHAR(500), -- if downloaded locally
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Testing Requirements

### Unit Tests
- [ ] Test static content extraction
- [ ] Test dynamic content extraction with Playwright
- [ ] Test content cleaning and noise removal
- [ ] Test markdown conversion quality
- [ ] Test metadata extraction

### Integration Tests
- [ ] Test with various website types (news, blogs, documentation)
- [ ] Test with JavaScript-heavy single-page applications
- [ ] Test with different content management systems
- [ ] Test with password-protected or login-required content
- [ ] Test with large pages and complex layouts

### Quality Validation Tests
- [ ] Compare extraction quality across methods
- [ ] Validate content completeness and accuracy
- [ ] Test noise removal effectiveness
- [ ] Validate markdown structure and formatting

## Performance Optimization

### Processing Strategies
1. **Browser Pool**: Reuse browser instances for multiple extractions
2. **Caching**: Cache extracted content for identical URLs
3. **Parallel Processing**: Process multiple URLs concurrently
4. **Smart Timeouts**: Adaptive timeouts based on page complexity

### Performance Targets
- **Speed**: <10 seconds per page for typical websites
- **Memory**: <500MB peak usage per browser instance
- **Accuracy**: >90% content extraction completeness
- **Quality**: >85% noise removal effectiveness

## Implementation Steps

### Step 1: Enhanced Static Extraction (1-2 days)
- [ ] Improve current Beautiful Soup implementation
- [ ] Add readability and trafilatura integration
- [ ] Enhance content cleaning algorithms
- [ ] Add better metadata extraction

### Step 2: Dynamic Content Extraction (2-3 days)
- [ ] Integrate Playwright for JavaScript rendering
- [ ] Implement smart waiting strategies
- [ ] Add content stability detection
- [ ] Optimize browser resource usage

### Step 3: Advanced Content Processing (2-3 days)
- [ ] Implement intelligent noise detection
- [ ] Add semantic content structuring
- [ ] Enhance markdown conversion
- [ ] Add image and link extraction

### Step 4: Integration and Testing (1-2 days)
- [ ] Integrate with MoRAG system
- [ ] Update database schema
- [ ] Comprehensive testing
- [ ] Performance optimization

## Success Criteria
- [ ] Support for both static and dynamic web content
- [ ] >90% content extraction accuracy
- [ ] Effective noise removal and content cleaning
- [ ] High-quality markdown output with proper structure
- [ ] Processing time <10 seconds for typical pages
- [ ] Seamless integration with existing pipeline

## Dependencies
- Playwright browser automation
- Enhanced content extraction libraries
- Improved HTML to markdown conversion
- Updated database schema

## Risks and Mitigation
- **Risk**: JavaScript-heavy sites causing extraction failures
  - **Mitigation**: Multiple extraction strategies, fallback methods
- **Risk**: Anti-bot measures blocking access
  - **Mitigation**: Respectful scraping practices, user agent rotation
- **Risk**: Performance issues with complex pages
  - **Mitigation**: Timeouts, resource limits, optimization
