# Task 28: Enhanced Office Documents to Markdown Conversion

## Status: FRAMEWORK READY ✅
**Note**: The universal document conversion framework (Task 24) has been implemented with placeholder office conversion. This task focuses on implementing full office document processing capabilities.

## Objective
Replace the placeholder office converter in the universal document conversion framework with comprehensive Office document (Word, Excel, PowerPoint) processing, including formatting preservation, table extraction, and embedded object handling.

## Research Phase

### Office Document Processing Libraries
1. **python-docx** (Word Documents)
   - Pros: Native Python, good formatting support, active development
   - Cons: Limited to .docx format, complex styling extraction
   
2. **openpyxl** (Excel Documents)
   - Pros: Excellent Excel support, formula handling, chart extraction
   - Cons: .xlsx only, complex for large spreadsheets
   
3. **python-pptx** (PowerPoint Documents)
   - Pros: Good slide structure extraction, shape handling
   - Cons: Limited text formatting, complex layout preservation
   
4. **LibreOffice/UNO** (Universal Office)
   - Pros: Supports all formats including legacy, comprehensive
   - Cons: Heavy dependency, complex setup, slower processing

### Alternative Approaches
1. **pandoc** - Universal document converter
2. **mammoth** - Word to HTML/Markdown converter
3. **xlwings** - Excel automation and extraction
4. **Office365 API** - Cloud-based conversion

## Implementation Strategy

### Phase 1: Multi-Format Office Converter
```python
class OfficeToMarkdownConverter(BaseConverter):
    def __init__(self):
        self.word_converter = WordToMarkdownConverter()
        self.excel_converter = ExcelToMarkdownConverter()
        self.powerpoint_converter = PowerPointToMarkdownConverter()
        self.format_detector = OfficeFormatDetector()
    
    async def convert(self, file_path: str, options: ConversionOptions) -> ConversionResult:
        # Detect specific office format
        office_format = self.format_detector.detect_format(file_path)
        
        # Route to appropriate converter
        if office_format in ['docx', 'doc']:
            return await self.word_converter.convert(file_path, options)
        elif office_format in ['xlsx', 'xls']:
            return await self.excel_converter.convert(file_path, options)
        elif office_format in ['pptx', 'ppt']:
            return await self.powerpoint_converter.convert(file_path, options)
        else:
            raise UnsupportedFormatError(f"Unsupported office format: {office_format}")
```

### Phase 2: Word Document Converter
```python
from docx import Document
from docx.shared import Inches
import re

class WordToMarkdownConverter:
    def __init__(self):
        self.style_mapper = WordStyleMapper()
        self.table_processor = WordTableProcessor()
        self.image_processor = WordImageProcessor()
    
    async def convert(self, docx_path: str, options: ConversionOptions) -> ConversionResult:
        doc = Document(docx_path)
        
        # Extract document metadata
        metadata = self.extract_metadata(doc)
        
        # Process document structure
        markdown_sections = []
        
        # Add document header
        markdown_sections.append(self.create_document_header(metadata))
        
        # Process paragraphs, tables, and other elements
        for element in doc.element.body:
            if element.tag.endswith('p'):  # Paragraph
                paragraph = self.find_paragraph_by_element(doc, element)
                if paragraph:
                    md_paragraph = await self.process_paragraph(paragraph, options)
                    if md_paragraph.strip():
                        markdown_sections.append(md_paragraph)
            
            elif element.tag.endswith('tbl'):  # Table
                table = self.find_table_by_element(doc, element)
                if table:
                    md_table = await self.table_processor.convert_table(table, options)
                    markdown_sections.append(md_table)
        
        # Process embedded images
        if options.extract_images:
            images_section = await self.image_processor.process_images(doc, options)
            if images_section:
                markdown_sections.append(images_section)
        
        markdown_content = "\n\n".join(markdown_sections)
        
        return ConversionResult(
            content=markdown_content,
            metadata=metadata,
            quality_score=self.assess_conversion_quality(doc, markdown_content)
        )
    
    async def process_paragraph(self, paragraph, options: ConversionOptions) -> str:
        # Handle different paragraph styles
        style_name = paragraph.style.name if paragraph.style else "Normal"
        
        # Extract text with formatting
        text_parts = []
        for run in paragraph.runs:
            text = run.text
            
            # Apply formatting
            if run.bold:
                text = f"**{text}**"
            if run.italic:
                text = f"*{text}*"
            if run.underline:
                text = f"<u>{text}</u>"
            
            text_parts.append(text)
        
        full_text = "".join(text_parts).strip()
        
        if not full_text:
            return ""
        
        # Convert based on style
        return self.style_mapper.convert_style_to_markdown(style_name, full_text)
```

### Phase 3: Excel Converter
```python
import openpyxl
from openpyxl.utils import get_column_letter

class ExcelToMarkdownConverter:
    def __init__(self):
        self.chart_processor = ExcelChartProcessor()
        self.formula_processor = ExcelFormulaProcessor()
    
    async def convert(self, xlsx_path: str, options: ConversionOptions) -> ConversionResult:
        workbook = openpyxl.load_workbook(xlsx_path, data_only=False)
        
        markdown_sections = []
        
        # Add workbook header
        metadata = self.extract_workbook_metadata(workbook)
        markdown_sections.append(self.create_workbook_header(metadata))
        
        # Process each worksheet
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            sheet_markdown = await self.process_worksheet(sheet, options)
            markdown_sections.append(sheet_markdown)
        
        # Process charts if requested
        if options.extract_charts:
            charts_section = await self.chart_processor.process_charts(workbook, options)
            if charts_section:
                markdown_sections.append(charts_section)
        
        markdown_content = "\n\n".join(markdown_sections)
        
        return ConversionResult(
            content=markdown_content,
            metadata=metadata,
            quality_score=self.assess_excel_quality(workbook, markdown_content)
        )
    
    async def process_worksheet(self, sheet, options: ConversionOptions) -> str:
        sections = [f"## Worksheet: {sheet.title}\n"]
        
        # Find data ranges
        data_ranges = self.find_data_ranges(sheet)
        
        for data_range in data_ranges:
            # Convert range to markdown table
            table_markdown = self.convert_range_to_table(sheet, data_range, options)
            if table_markdown:
                sections.append(table_markdown)
        
        # Add formulas section if requested
        if options.include_formulas:
            formulas_section = self.extract_formulas(sheet)
            if formulas_section:
                sections.append(formulas_section)
        
        return "\n\n".join(sections)
    
    def convert_range_to_table(self, sheet, data_range, options: ConversionOptions) -> str:
        min_row, min_col, max_row, max_col = data_range
        
        # Extract data
        table_data = []
        for row in sheet.iter_rows(min_row=min_row, max_row=max_row, 
                                  min_col=min_col, max_col=max_col):
            row_data = []
            for cell in row:
                value = cell.value
                if value is None:
                    value = ""
                elif isinstance(value, (int, float)):
                    value = str(value)
                row_data.append(str(value))
            table_data.append(row_data)
        
        if not table_data:
            return ""
        
        # Format as markdown table
        markdown_lines = []
        
        # Header row
        if table_data:
            headers = table_data[0]
            markdown_lines.append("| " + " | ".join(headers) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            # Data rows
            for row in table_data[1:]:
                markdown_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(markdown_lines)
```

### Phase 4: PowerPoint Converter
```python
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

class PowerPointToMarkdownConverter:
    def __init__(self):
        self.shape_processor = PowerPointShapeProcessor()
        self.layout_analyzer = SlideLayoutAnalyzer()
    
    async def convert(self, pptx_path: str, options: ConversionOptions) -> ConversionResult:
        presentation = Presentation(pptx_path)
        
        markdown_sections = []
        
        # Add presentation header
        metadata = self.extract_presentation_metadata(presentation)
        markdown_sections.append(self.create_presentation_header(metadata))
        
        # Process each slide
        for slide_num, slide in enumerate(presentation.slides, 1):
            slide_markdown = await self.process_slide(slide, slide_num, options)
            markdown_sections.append(slide_markdown)
        
        markdown_content = "\n\n".join(markdown_sections)
        
        return ConversionResult(
            content=markdown_content,
            metadata=metadata,
            quality_score=self.assess_presentation_quality(presentation, markdown_content)
        )
    
    async def process_slide(self, slide, slide_num: int, options: ConversionOptions) -> str:
        sections = [f"## Slide {slide_num}\n"]
        
        # Analyze slide layout
        layout_info = self.layout_analyzer.analyze_layout(slide)
        
        # Process shapes in order of importance
        for shape in slide.shapes:
            shape_content = await self.shape_processor.process_shape(shape, options)
            if shape_content:
                sections.append(shape_content)
        
        # Add slide notes if available
        if slide.has_notes_slide and options.include_notes:
            notes_text = slide.notes_slide.notes_text_frame.text
            if notes_text.strip():
                sections.append(f"**Speaker Notes**: {notes_text}")
        
        return "\n\n".join(sections)
```

## Structured Office Document Markdown Output

### Word Document Template
```markdown
# Document: {filename}

**Source**: {original_filename}
**Author**: {author}
**Created**: {created_date}
**Modified**: {modified_date}
**Pages**: {page_count}
**Word Count**: {word_count}

## Document Summary
{ai_generated_summary}

## Content

### Section 1: {heading_text}
{paragraph_content}

#### Subsection 1.1
{subsection_content}

### Tables

#### Table 1: {table_title}
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

### Images and Figures
![Figure 1](image_1.png)
*Caption: {image_caption}*

## Document Metadata
- Template: {template_name}
- Language: {language}
- Revision: {revision_number}
- Comments: {comment_count}
```

### Excel Workbook Template
```markdown
# Workbook: {filename}

**Source**: {original_filename}
**Created**: {created_date}
**Modified**: {modified_date}
**Worksheets**: {sheet_count}
**Total Cells**: {cell_count}

## Workbook Summary
{ai_generated_summary}

## Worksheet: Sheet1

### Data Table 1
| Product | Q1 Sales | Q2 Sales | Q3 Sales | Q4 Sales |
|---------|----------|----------|----------|----------|
| Product A | 1000 | 1200 | 1100 | 1300 |
| Product B | 800 | 900 | 950 | 1000 |

### Formulas Used
- **Cell D2**: `=SUM(B2:C2)` - Total sales calculation
- **Cell E5**: `=AVERAGE(B2:B10)` - Average calculation

### Charts
![Sales Chart](chart_1.png)
*Chart showing quarterly sales trends*

## Worksheet: Summary

### Key Metrics
| Metric | Value |
|--------|-------|
| Total Revenue | $50,000 |
| Growth Rate | 15% |
```

### PowerPoint Presentation Template
```markdown
# Presentation: {filename}

**Source**: {original_filename}
**Created**: {created_date}
**Slides**: {slide_count}
**Theme**: {theme_name}

## Presentation Summary
{ai_generated_summary}

## Slide 1: Title Slide
**Title**: {slide_title}
**Subtitle**: {slide_subtitle}
**Author**: {presenter_name}

## Slide 2: Agenda
- Introduction
- Main Topics
- Conclusion
- Q&A

## Slide 3: Main Content
**Title**: {slide_title}

### Key Points
- Point 1: {bullet_point}
- Point 2: {bullet_point}
- Point 3: {bullet_point}

**Speaker Notes**: {notes_content}

### Visual Elements
![Slide Image](slide_3_image.png)
*Diagram showing process flow*
```

## Configuration Options
```yaml
office_conversion:
  word:
    preserve_formatting: true
    extract_images: true
    include_comments: false
    include_track_changes: false
    convert_tables: true
    
  excel:
    extract_charts: true
    include_formulas: true
    convert_all_sheets: true
    preserve_formatting: false
    max_table_size: 1000  # cells
    
  powerpoint:
    extract_images: true
    include_notes: true
    include_animations: false
    preserve_layout: false
    
  general:
    include_metadata: true
    generate_summary: true
    clean_whitespace: true
    max_file_size: "100MB"
```

## Integration with MoRAG System

### Enhanced Office Processing Service
```python
# Update services/document_parser.py
class DocumentParser:
    def __init__(self):
        self.office_converter = OfficeToMarkdownConverter()
        self.chunker = DocumentChunker()
        self.embedder = EmbeddingService()
    
    async def parse_office_document(self, file_path: str, options: ProcessingOptions) -> ParsedDocument:
        # Convert to markdown
        conversion_result = await self.office_converter.convert(file_path, options.conversion)
        
        # Create chunks based on document structure
        chunks = await self.chunker.create_structure_based_chunks(
            conversion_result.content,
            options.chunking
        )
        
        # Generate embeddings for each chunk
        for chunk in chunks:
            chunk.embedding = await self.embedder.embed_text(chunk.content)
        
        return ParsedDocument(
            markdown_content=conversion_result.content,
            chunks=chunks,
            metadata=conversion_result.metadata,
            quality_score=conversion_result.quality_score
        )
```

### Database Schema Updates
```sql
-- Office-specific metadata
ALTER TABLE documents ADD COLUMN office_type VARCHAR(20); -- word, excel, powerpoint
ALTER TABLE documents ADD COLUMN office_version VARCHAR(50);
ALTER TABLE documents ADD COLUMN page_count INTEGER;
ALTER TABLE documents ADD COLUMN word_count INTEGER;
ALTER TABLE documents ADD COLUMN has_tables BOOLEAN DEFAULT FALSE;
ALTER TABLE documents ADD COLUMN has_images BOOLEAN DEFAULT FALSE;
ALTER TABLE documents ADD COLUMN has_charts BOOLEAN DEFAULT FALSE;

-- Office document structure
CREATE TABLE document_structure (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    element_type VARCHAR(50), -- heading, paragraph, table, image, chart
    element_order INTEGER,
    level INTEGER, -- for headings
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tables extracted from office documents
CREATE TABLE document_tables (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    table_name VARCHAR(200),
    headers JSONB,
    data JSONB,
    row_count INTEGER,
    column_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Testing Requirements

### Unit Tests
- [ ] Test Word document conversion with various styles
- [ ] Test Excel workbook conversion with formulas and charts
- [ ] Test PowerPoint conversion with complex layouts
- [ ] Test metadata extraction for all formats
- [ ] Test error handling for corrupted files

### Integration Tests
- [ ] Test with real-world office documents
- [ ] Test with legacy formats (.doc, .xls, .ppt)
- [ ] Test with password-protected documents
- [ ] Test with large files and complex formatting
- [ ] Test with embedded objects and macros

### Quality Validation Tests
- [ ] Compare conversion quality across formats
- [ ] Validate table structure preservation
- [ ] Test formatting retention accuracy
- [ ] Validate metadata completeness

## Implementation Steps

### Step 1: Word Document Converter (2-3 days)
- [ ] Implement python-docx integration
- [ ] Add style mapping and formatting preservation
- [ ] Implement table conversion
- [ ] Add image extraction

### Step 2: Excel Converter (2-3 days)
- [ ] Implement openpyxl integration
- [ ] Add worksheet processing
- [ ] Implement table conversion from ranges
- [ ] Add formula extraction

### Step 3: PowerPoint Converter (2-3 days)
- [ ] Implement python-pptx integration
- [ ] Add slide processing
- [ ] Implement shape and text extraction
- [ ] Add notes and layout handling

### Step 4: Integration and Testing (2-3 days)
- [ ] Integrate with MoRAG system
- [ ] Update database schema
- [ ] Comprehensive testing
- [ ] Performance optimization

## Success Criteria
- [ ] Support for all major Office formats (Word, Excel, PowerPoint)
- [ ] >90% content extraction accuracy
- [ ] Table structure preservation >85%
- [ ] Metadata extraction completeness >95%
- [ ] Processing time <30 seconds for typical documents
- [ ] Seamless integration with existing pipeline

## Dependencies
- python-docx, openpyxl, python-pptx libraries
- Image processing capabilities
- Enhanced chunking strategies
- Updated database schema

## Risks and Mitigation
- **Risk**: Complex formatting loss during conversion
  - **Mitigation**: Configurable formatting preservation, fallback options
- **Risk**: Large Excel files causing memory issues
  - **Mitigation**: Streaming processing, size limits, chunked reading
- **Risk**: PowerPoint layout complexity
  - **Mitigation**: Layout analysis, simplified extraction modes
