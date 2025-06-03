# Task 25: Enhanced PDF to Markdown Conversion with Docling Integration

## Status: FRAMEWORK READY ✅
**Note**: The universal document conversion framework (Task 24) has been implemented. This task now focuses on enhancing the existing PDF converter with advanced docling features.

## Objective
Enhance the existing PDF converter in the universal document conversion framework with advanced docling features, improved table extraction, OCR capabilities, and enhanced quality assessment.

## Research Phase

### Docling Library Analysis
- **Primary Tool**: Docling - Advanced PDF parsing for AI applications
- **Capabilities**: Text extraction, table detection, image extraction, layout analysis
- **Advantages**: Designed for AI-readable output, better structure preservation
- **Integration**: Python library with async support

### Alternative Libraries (Fallback)
1. **PyMuPDF (fitz)**: Fast, comprehensive PDF processing
2. **pdfplumber**: Excellent table extraction capabilities  
3. **Unstructured.io**: Current implementation (backup option)
4. **PDFMiner**: Low-level PDF parsing for complex documents

## Current Implementation Status

### ✅ Completed (Task 24)
- Basic PDF converter integrated with universal framework
- Docling integration with fallback to unstructured.io
- Page-level chunking support
- Quality assessment framework
- Structured markdown output

### 🔄 Enhancement Strategy

### Phase 1: Advanced Docling Features
```python
# Enhance existing src/morag/converters/pdf.py
class EnhancedPDFConverter(BaseConverter):
    def __init__(self):
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfPipelineOptions(
                    do_ocr=True,
                    do_table_structure=True,
                    table_structure_options=TableStructureOptions(
                        do_cell_matching=True
                    )
                )
            }
        )
    
    async def convert(self, pdf_path: str, options: ConversionOptions) -> ConversionResult:
        # Convert PDF using docling
        result = self.converter.convert(pdf_path)
        
        # Process into structured markdown
        markdown_content = await self.process_docling_result(result, options)
        
        return ConversionResult(
            content=markdown_content,
            metadata=self.extract_metadata(result),
            quality_score=self.calculate_quality(result)
        )
```

### Phase 2: Enhanced Processing Pipeline
```python
class PDFMarkdownProcessor:
    def __init__(self):
        self.table_processor = TableMarkdownProcessor()
        self.image_processor = ImageMarkdownProcessor()
        self.text_processor = TextMarkdownProcessor()
    
    async def process_docling_result(self, docling_result, options: ConversionOptions) -> str:
        markdown_sections = []
        
        # Process by pages (user's preferred chunking)
        for page_num, page in enumerate(docling_result.pages, 1):
            page_markdown = await self.process_page(page, page_num, options)
            markdown_sections.append(page_markdown)
        
        return self.combine_sections(markdown_sections, options)
    
    async def process_page(self, page, page_num: int, options: ConversionOptions) -> str:
        sections = [f"### Page {page_num}\n"]
        
        # Process text blocks
        for text_block in page.text_blocks:
            sections.append(self.process_text_block(text_block))
        
        # Process tables
        for table in page.tables:
            table_md = await self.table_processor.convert_table(table)
            sections.append(table_md)
        
        # Process images
        for image in page.images:
            image_md = await self.image_processor.process_image(image, options)
            sections.append(image_md)
        
        return "\n\n".join(sections)
```

### Phase 3: Advanced Features

#### Table Processing
```python
class TableMarkdownProcessor:
    async def convert_table(self, table) -> str:
        # Extract table structure from docling
        headers = self.extract_headers(table)
        rows = self.extract_rows(table)
        
        # Convert to markdown table format
        markdown_table = self.format_as_markdown_table(headers, rows)
        
        # Add table metadata
        table_info = f"**Table**: {table.caption or 'Untitled'}\n"
        
        return f"{table_info}\n{markdown_table}\n"
    
    def format_as_markdown_table(self, headers: List[str], rows: List[List[str]]) -> str:
        if not headers or not rows:
            return ""
        
        # Create markdown table
        table_lines = []
        table_lines.append("| " + " | ".join(headers) + " |")
        table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in rows:
            table_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(table_lines)
```

#### Image and OCR Processing
```python
class ImageMarkdownProcessor:
    def __init__(self):
        self.ocr_engine = OCREngine()  # Tesseract or similar
        self.vision_model = VisionModel()  # For image description
    
    async def process_image(self, image, options: ConversionOptions) -> str:
        sections = []
        
        # Generate image description
        if options.describe_images:
            description = await self.vision_model.describe_image(image.data)
            sections.append(f"![{description}](image_{image.id})")
            sections.append(f"*{image.caption or description}*")
        
        # Extract text via OCR
        if options.extract_image_text:
            ocr_text = await self.ocr_engine.extract_text(image.data)
            if ocr_text.strip():
                sections.append(f"**Image Text**: {ocr_text}")
        
        return "\n\n".join(sections)
```

## Configuration Options

### PDF Processing Settings
```yaml
pdf_conversion:
  # Docling-specific options
  docling:
    enable_ocr: true
    extract_tables: true
    extract_images: true
    preserve_layout: false
    table_structure_detection: true
    
  # Page-level chunking (user preference)
  chunking:
    strategy: "page"  # page, section, paragraph
    include_page_numbers: true
    preserve_page_breaks: true
    
  # Quality settings
  quality:
    min_text_confidence: 0.8
    min_table_confidence: 0.7
    skip_low_quality_images: true
    
  # Output formatting
  output:
    include_metadata: true
    include_toc: true
    preserve_formatting: true
    clean_whitespace: true
```

### Fallback Configuration
```yaml
pdf_fallback:
  primary: "docling"
  secondary: "pymupdf"
  tertiary: "pdfplumber"
  final: "unstructured"
  
  fallback_triggers:
    - docling_error
    - low_quality_score
    - processing_timeout
    - memory_limit_exceeded
```

## Integration with MoRAG System

### Current System Updates
```python
# Update services/document_parser.py
class DocumentParser:
    def __init__(self):
        self.pdf_converter = DoclingPDFConverter()
        self.fallback_converters = [
            PyMuPDFConverter(),
            PDFPlumberConverter(),
            UnstructuredConverter()
        ]
    
    async def parse_pdf(self, file_path: str) -> ParsedDocument:
        try:
            # Try primary converter (docling)
            result = await self.pdf_converter.convert(file_path)
            
            if result.quality_score >= self.min_quality_threshold:
                return result
            
        except Exception as e:
            logger.warning(f"Docling conversion failed: {e}")
        
        # Try fallback converters
        return await self.try_fallback_converters(file_path)
```

### Database Schema Updates
```sql
-- Track PDF-specific conversion details
ALTER TABLE documents ADD COLUMN pdf_pages INTEGER;
ALTER TABLE documents ADD COLUMN pdf_has_tables BOOLEAN DEFAULT FALSE;
ALTER TABLE documents ADD COLUMN pdf_has_images BOOLEAN DEFAULT FALSE;
ALTER TABLE documents ADD COLUMN pdf_ocr_used BOOLEAN DEFAULT FALSE;
ALTER TABLE documents ADD COLUMN pdf_converter_used VARCHAR(50);

-- Page-level chunks table
CREATE TABLE document_pages (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    page_number INTEGER,
    content TEXT,
    embedding VECTOR(768),
    has_tables BOOLEAN DEFAULT FALSE,
    has_images BOOLEAN DEFAULT FALSE,
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_document_pages_doc_id ON document_pages(document_id);
CREATE INDEX idx_document_pages_page_num ON document_pages(page_number);
```

## Error Handling and Quality Assurance

### Quality Metrics
```python
class PDFQualityAssessment:
    def assess_conversion_quality(self, original_pdf: str, markdown_result: str) -> QualityScore:
        return QualityScore(
            text_extraction_completeness=self.check_text_completeness(original_pdf, markdown_result),
            table_preservation_accuracy=self.check_table_accuracy(original_pdf, markdown_result),
            image_detection_rate=self.check_image_detection(original_pdf, markdown_result),
            structure_preservation=self.check_structure_preservation(original_pdf, markdown_result),
            readability_score=self.check_markdown_readability(markdown_result)
        )
```

### Error Recovery Strategies
1. **Docling Failure**: Fall back to PyMuPDF
2. **OCR Failure**: Skip OCR, use text-only extraction
3. **Table Detection Issues**: Use simpler table extraction
4. **Memory Issues**: Process in smaller chunks
5. **Timeout**: Use faster, less accurate methods

## Testing Requirements

### Unit Tests
- [ ] Test docling integration with various PDF types
- [ ] Test table extraction accuracy
- [ ] Test image processing and OCR
- [ ] Test page-level chunking
- [ ] Test fallback mechanisms

### Integration Tests
- [ ] Test with real-world PDF documents
- [ ] Test with scanned PDFs (OCR-heavy)
- [ ] Test with complex layouts and tables
- [ ] Test with image-heavy documents
- [ ] Test performance with large PDFs

### Quality Validation Tests
- [ ] Compare output quality with current implementation
- [ ] Validate table structure preservation
- [ ] Test OCR accuracy on various image types
- [ ] Validate metadata extraction completeness

## Performance Optimization

### Processing Strategies
1. **Parallel Page Processing**: Process pages concurrently
2. **Lazy Loading**: Load pages on demand for large PDFs
3. **Caching**: Cache processed results for identical PDFs
4. **Memory Management**: Stream processing for large files

### Benchmarking Targets
- **Speed**: <5 seconds per page for typical documents
- **Memory**: <500MB peak usage for 100-page documents
- **Quality**: >90% text extraction accuracy
- **Tables**: >85% table structure preservation

## Implementation Steps

### Step 1: Docling Setup and Basic Integration (2-3 days)
- [ ] Install and configure docling library
- [ ] Create basic PDF converter class
- [ ] Implement simple text extraction
- [ ] Add basic error handling

### Step 2: Advanced Features (3-4 days)
- [ ] Implement table extraction and markdown conversion
- [ ] Add image processing and OCR capabilities
- [ ] Implement page-level chunking
- [ ] Add metadata extraction

### Step 3: Quality and Fallback Systems (2-3 days)
- [ ] Implement quality assessment
- [ ] Add fallback converter chain
- [ ] Optimize performance
- [ ] Add comprehensive logging

### Step 4: Integration and Testing (2-3 days)
- [ ] Integrate with existing MoRAG system
- [ ] Update database schema
- [ ] Comprehensive testing
- [ ] Performance benchmarking

## Success Criteria
- [ ] Docling successfully integrated as primary PDF converter
- [ ] Page-level chunking implemented and working
- [ ] Table extraction accuracy >85%
- [ ] OCR functionality working for scanned documents
- [ ] Fallback system handles edge cases gracefully
- [ ] Performance meets or exceeds current implementation
- [ ] Quality scores consistently >0.8 for typical documents

## Dependencies
- Docling library installation and configuration
- OCR engine setup (Tesseract or cloud service)
- Vision model for image description (optional)
- Updated database schema
- Enhanced error handling framework

## Risks and Mitigation
- **Risk**: Docling library instability or breaking changes
  - **Mitigation**: Version pinning, comprehensive fallback system
- **Risk**: Performance degradation with complex PDFs
  - **Mitigation**: Performance monitoring, optimization, chunked processing
- **Risk**: Quality regression compared to current system
  - **Mitigation**: A/B testing, quality metrics, gradual rollout
