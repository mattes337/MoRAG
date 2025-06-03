"""Universal document conversion service for MoRAG."""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import structlog

from ..converters import DocumentConverter, ConversionOptions, ConversionResult, ChunkingStrategy
from ..converters.config import get_conversion_config
from ..core.exceptions import ValidationError, ProcessingError
from ..services.chunking import ChunkingService
from ..services.embedding import gemini_service

logger = structlog.get_logger(__name__)


class UniversalConverterService:
    """Service that provides universal document conversion capabilities."""
    
    def __init__(self):
        self.document_converter = DocumentConverter()
        self.chunking_service = ChunkingService()
        self.embedding_service = gemini_service
        self.config = get_conversion_config()
        
        # Conversion statistics
        self.stats = {
            'total_conversions': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'conversions_by_format': {},
            'average_processing_time': 0.0,
            'average_quality_score': 0.0
        }
    
    async def convert_document(
        self,
        file_path: Union[str, Path],
        options: Optional[ConversionOptions] = None,
        generate_embeddings: bool = True
    ) -> Dict:
        """Convert document to markdown with optional embedding generation.
        
        Args:
            file_path: Path to document to convert
            options: Conversion options (optional)
            generate_embeddings: Whether to generate embeddings for chunks
            
        Returns:
            Dictionary with conversion results and metadata
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # Validate input
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        max_size = self.config.performance_settings['max_file_size']
        if file_size > max_size:
            raise ValidationError(f"File too large: {file_size} bytes (max: {max_size})")
        
        # Use default options if not provided
        if options is None:
            format_type = self.document_converter.detect_format(file_path)
            options = ConversionOptions.for_format(format_type)
            
            # Apply config defaults
            for key, value in self.config.default_options.items():
                if hasattr(options, key):
                    setattr(options, key, value)
            
            # Apply format-specific config
            format_options = self.config.get_format_options(format_type)
            options.format_options.update(format_options)
        
        logger.info(
            "Starting universal document conversion",
            file_path=str(file_path),
            file_size=file_size,
            format_type=self.document_converter.detect_format(file_path),
            chunking_strategy=options.chunking_strategy.value
        )
        
        try:
            # Convert document
            conversion_result = await self.document_converter.convert_to_markdown(file_path, options)
            
            # Generate chunks if successful
            chunks = []
            embeddings = []
            
            if conversion_result.success and conversion_result.content:
                chunks = await self._generate_chunks(conversion_result.content, options)
                
                if generate_embeddings and chunks:
                    embeddings = await self._generate_embeddings(chunks)
            
            # Update statistics
            self._update_stats(conversion_result, time.time() - start_time)
            
            result = {
                'success': conversion_result.success,
                'content': conversion_result.content,
                'metadata': conversion_result.metadata,
                'quality_score': conversion_result.quality_score.__dict__ if conversion_result.quality_score else None,
                'processing_time': conversion_result.processing_time,
                'chunks': [{'text': chunk.text, 'metadata': chunk.__dict__} for chunk in chunks],
                'embeddings': embeddings,
                'warnings': conversion_result.warnings,
                'error_message': conversion_result.error_message,
                'converter_used': conversion_result.converter_used,
                'fallback_used': conversion_result.fallback_used,
                'original_format': conversion_result.original_format,
                'images': conversion_result.images,
                'word_count': conversion_result.word_count,
                'total_processing_time': time.time() - start_time
            }
            
            logger.info(
                "Universal document conversion completed",
                success=conversion_result.success,
                quality_score=conversion_result.quality_score.overall_score if conversion_result.quality_score else None,
                chunks_count=len(chunks),
                embeddings_count=len(embeddings),
                total_time=result['total_processing_time']
            )
            
            return result
            
        except Exception as e:
            self.stats['failed_conversions'] += 1
            self.stats['total_conversions'] += 1
            
            error_msg = f"Universal conversion failed: {str(e)}"
            logger.error(
                "Universal document conversion failed",
                error=str(e),
                error_type=type(e).__name__,
                processing_time=time.time() - start_time
            )
            
            return {
                'success': False,
                'content': '',
                'metadata': {},
                'quality_score': None,
                'processing_time': time.time() - start_time,
                'chunks': [],
                'embeddings': [],
                'warnings': [],
                'error_message': error_msg,
                'converter_used': None,
                'fallback_used': False,
                'original_format': self.document_converter.detect_format(file_path),
                'images': [],
                'word_count': 0,
                'total_processing_time': time.time() - start_time
            }
    
    async def _generate_chunks(self, content: str, options: ConversionOptions) -> List:
        """Generate chunks from converted content.
        
        Args:
            content: Converted markdown content
            options: Conversion options
            
        Returns:
            List of chunk objects
        """
        try:
            chunks = await self.chunking_service.chunk_with_metadata(
                content,
                strategy=options.chunking_strategy.value
            )
            
            logger.info(f"Generated {len(chunks)} chunks using {options.chunking_strategy.value} strategy")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to generate chunks: {str(e)}")
            return []
    
    async def _generate_embeddings(self, chunks: List) -> List[List[float]]:
        """Generate embeddings for chunks.
        
        Args:
            chunks: List of chunk objects
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            
            for chunk in chunks:
                embedding_result = await self.embedding_service.generate_embedding(chunk.text)
                embeddings.append(embedding_result.embedding)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            return []
    
    def _update_stats(self, result: ConversionResult, processing_time: float):
        """Update conversion statistics.
        
        Args:
            result: Conversion result
            processing_time: Total processing time
        """
        self.stats['total_conversions'] += 1
        
        if result.success:
            self.stats['successful_conversions'] += 1
        else:
            self.stats['failed_conversions'] += 1
        
        # Update format statistics
        format_type = result.original_format or 'unknown'
        if format_type not in self.stats['conversions_by_format']:
            self.stats['conversions_by_format'][format_type] = {
                'count': 0,
                'successful': 0,
                'failed': 0,
                'avg_quality': 0.0,
                'avg_time': 0.0
            }
        
        format_stats = self.stats['conversions_by_format'][format_type]
        format_stats['count'] += 1
        
        if result.success:
            format_stats['successful'] += 1
            
            if result.quality_score:
                # Update average quality score
                old_avg = format_stats['avg_quality']
                new_count = format_stats['successful']
                format_stats['avg_quality'] = (old_avg * (new_count - 1) + result.quality_score.overall_score) / new_count
        else:
            format_stats['failed'] += 1
        
        # Update average processing time
        old_avg = format_stats['avg_time']
        new_count = format_stats['count']
        format_stats['avg_time'] = (old_avg * (new_count - 1) + processing_time) / new_count
        
        # Update global averages
        total = self.stats['total_conversions']
        self.stats['average_processing_time'] = (
            self.stats['average_processing_time'] * (total - 1) + processing_time
        ) / total
        
        if result.success and result.quality_score:
            successful = self.stats['successful_conversions']
            self.stats['average_quality_score'] = (
                self.stats['average_quality_score'] * (successful - 1) + result.quality_score.overall_score
            ) / successful
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats.
        
        Returns:
            List of supported format types
        """
        return self.document_converter.list_supported_formats()
    
    def get_converter_info(self) -> Dict:
        """Get information about registered converters.
        
        Returns:
            Dictionary with converter information
        """
        return self.document_converter.get_converter_info()
    
    def get_statistics(self) -> Dict:
        """Get conversion statistics.
        
        Returns:
            Dictionary with conversion statistics
        """
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset conversion statistics."""
        self.stats = {
            'total_conversions': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'conversions_by_format': {},
            'average_processing_time': 0.0,
            'average_quality_score': 0.0
        }
        logger.info("Reset conversion statistics")


# Global service instance
universal_converter_service = UniversalConverterService()
