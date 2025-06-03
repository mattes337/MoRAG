"""Web content to Markdown converter using existing MoRAG web scraper."""

import time
from pathlib import Path
from typing import Union
import structlog

from .base import BaseConverter, ConversionOptions, ConversionResult, QualityScore
from .quality import ConversionQualityValidator
from ..processors.web import web_processor

logger = structlog.get_logger(__name__)


class WebConverter(BaseConverter):
    """Web content to Markdown converter using MoRAG's web scraper."""
    
    def __init__(self):
        super().__init__("MoRAG Web Converter")
        self.supported_formats = ['web', 'html', 'htm', 'url']
        self.quality_validator = ConversionQualityValidator()
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this converter supports the given format."""
        return format_type.lower() in self.supported_formats
    
    async def convert(self, file_path: Union[str, Path], options: ConversionOptions) -> ConversionResult:
        """Convert web content to structured markdown.
        
        Args:
            file_path: URL or path to HTML file
            options: Conversion options
            
        Returns:
            ConversionResult with markdown content
        """
        start_time = time.time()
        
        # Handle both URLs and file paths
        if isinstance(file_path, Path) or (isinstance(file_path, str) and not file_path.startswith(('http://', 'https://'))):
            # Local HTML file
            file_path = Path(file_path)
            await self.validate_input(file_path)
            url = f"file://{file_path.absolute()}"
            is_local_file = True
        else:
            # URL
            url = str(file_path)
            is_local_file = False
        
        logger.info(
            "Starting web content conversion",
            url=url,
            is_local_file=is_local_file,
            extract_main_content=options.format_options.get('extract_main_content', True)
        )
        
        try:
            if is_local_file:
                # Read local HTML file
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Use content converter to convert HTML to markdown
                from ..services.content_converter import content_converter
                conversion_result = await content_converter.html_to_markdown(html_content)
                
                web_result = type('WebResult', (), {
                    'content': conversion_result.content,
                    'metadata': {
                        'url': url,
                        'title': 'Local HTML File',
                        'filename': file_path.name,
                        **conversion_result.metadata
                    },
                    'success': conversion_result.success
                })()
            else:
                # Use existing MoRAG web processor
                web_result = await web_processor.process_url(url)
            
            # Convert to structured markdown
            markdown_content = await self._create_structured_markdown(web_result, options)
            
            # Calculate quality score
            quality_score = self.quality_validator.validate_conversion(str(file_path), ConversionResult(
                content=markdown_content,
                metadata=web_result.metadata
            ))
            
            processing_time = time.time() - start_time
            
            result = ConversionResult(
                content=markdown_content,
                metadata=self._enhance_metadata(web_result.metadata, file_path),
                quality_score=quality_score,
                processing_time=processing_time,
                success=web_result.success,
                original_format='web',
                converter_used=self.name
            )
            
            logger.info(
                "Web content conversion completed",
                processing_time=processing_time,
                quality_score=quality_score.overall_score,
                word_count=result.word_count,
                url=url
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Web content conversion failed: {str(e)}"
            
            logger.error(
                "Web content conversion failed",
                error=str(e),
                error_type=type(e).__name__,
                processing_time=processing_time,
                url=url
            )
            
            return ConversionResult(
                content="",
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=error_msg,
                original_format='web',
                converter_used=self.name
            )
    
    async def _create_structured_markdown(self, web_result, options: ConversionOptions) -> str:
        """Create structured markdown from web scraping result.
        
        Args:
            web_result: Web scraping result from MoRAG scraper
            options: Conversion options
            
        Returns:
            Structured markdown content
        """
        sections = []
        
        # Document header
        title = web_result.metadata.get('title', 'Web Page')
        sections.append(f"# {title}")
        sections.append("")
        
        # Metadata section
        if options.include_metadata:
            sections.append("## Page Information")
            sections.append("")
            
            metadata_items = [
                ("**URL**", web_result.metadata.get('url', 'Unknown')),
                ("**Title**", web_result.metadata.get('title', 'Unknown')),
                ("**Author**", web_result.metadata.get('author', 'Unknown')),
                ("**Published**", web_result.metadata.get('published_date', 'Unknown')),
                ("**Language**", web_result.metadata.get('language', 'Unknown')),
                ("**Extraction Method**", web_result.metadata.get('extraction_method', 'MoRAG Web Scraper'))
            ]
            
            for label, value in metadata_items:
                if value and value != 'Unknown':
                    sections.append(f"{label}: {value}")
            
            sections.append("")
        
        # Table of contents (if requested)
        if options.include_toc:
            sections.append("## Table of Contents")
            sections.append("")
            sections.append("- [Main Content](#main-content)")
            
            if hasattr(web_result, 'links') and web_result.links:
                sections.append("- [Links](#links)")
            
            if hasattr(web_result, 'images') and web_result.images:
                sections.append("- [Images](#images)")
            
            sections.append("- [Page Information](#page-information)")
            sections.append("")
        
        # Main content
        sections.append("## Main Content")
        sections.append("")
        
        if hasattr(web_result, 'content') and web_result.content:
            sections.append(web_result.content)
        else:
            sections.append("*No content extracted*")
        
        sections.append("")
        
        # Links section
        if hasattr(web_result, 'links') and web_result.links and options.format_options.get('include_links', True):
            sections.append("## Links")
            sections.append("")
            
            for link in web_result.links[:20]:  # Limit to first 20 links
                link_text = link.get('text', 'Link')
                link_url = link.get('url', '#')
                sections.append(f"- [{link_text}]({link_url})")
            
            if len(web_result.links) > 20:
                sections.append(f"- *... and {len(web_result.links) - 20} more links*")
            
            sections.append("")
        
        # Images section
        if hasattr(web_result, 'images') and web_result.images and options.extract_images:
            sections.append("## Images")
            sections.append("")
            
            for i, image in enumerate(web_result.images[:10], 1):  # Limit to first 10 images
                alt_text = image.get('alt', f'Image {i}')
                image_url = image.get('src', '#')
                sections.append(f"### Image {i}")
                sections.append(f"![{alt_text}]({image_url})")
                
                if image.get('caption'):
                    sections.append(f"**Caption**: {image['caption']}")
                
                sections.append("")
            
            if len(web_result.images) > 10:
                sections.append(f"*... and {len(web_result.images) - 10} more images*")
                sections.append("")
        
        # Processing details
        sections.append("## Processing Details")
        sections.append("")
        sections.append(f"**Extraction Method**: {web_result.metadata.get('extraction_method', 'MoRAG Web Scraper')}")
        
        if 'word_count' in web_result.metadata:
            sections.append(f"**Word Count**: {web_result.metadata['word_count']}")
        
        if 'extraction_time' in web_result.metadata:
            sections.append(f"**Extraction Time**: {web_result.metadata['extraction_time']:.2f} seconds")
        
        return "\n".join(sections)
    
    def _enhance_metadata(self, original_metadata: dict, file_path: Union[str, Path]) -> dict:
        """Enhance metadata with additional information.
        
        Args:
            original_metadata: Original metadata from scraper
            file_path: Original file path or URL
            
        Returns:
            Enhanced metadata dictionary
        """
        enhanced = original_metadata.copy()
        
        # Add conversion information
        enhanced.update({
            'conversion_format': 'web_to_markdown',
            'converter_version': '1.0.0'
        })
        
        # Add file information if it's a local file
        if isinstance(file_path, Path):
            enhanced.update({
                'original_filename': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_extension': file_path.suffix.lower()
            })
        else:
            enhanced['source_url'] = str(file_path)
        
        return enhanced
