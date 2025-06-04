"""Quality assessment for document conversion results."""

import re
import math
from typing import Dict, List, Optional
from pathlib import Path
import structlog

from .base import ConversionResult, QualityScore

logger = structlog.get_logger(__name__)


class ConversionQualityValidator:
    """Validates and assesses the quality of document conversions."""
    
    def __init__(self):
        self.min_content_length = 50
        self.min_word_count = 10
        self.max_noise_ratio = 0.3
    
    def validate_conversion(self, original_file: str, result: ConversionResult) -> QualityScore:
        """Comprehensive quality validation of conversion result.
        
        Args:
            original_file: Path to original document
            result: Conversion result to validate
            
        Returns:
            QualityScore with detailed assessment
        """
        logger.info("Starting quality validation", original_file=original_file)
        
        # Calculate individual quality metrics
        completeness_score = self._check_completeness(original_file, result)
        readability_score = self._check_readability(result.content)
        structure_score = self._check_structure(result.content)
        metadata_preservation = self._check_metadata_preservation(result)
        
        # Calculate overall score (weighted average)
        overall_score = (
            completeness_score * 0.3 +
            readability_score * 0.25 +
            structure_score * 0.25 +
            metadata_preservation * 0.2
        )
        
        quality_score = QualityScore(
            overall_score=overall_score,
            completeness_score=completeness_score,
            readability_score=readability_score,
            structure_score=structure_score,
            metadata_preservation=metadata_preservation
        )
        
        logger.info(
            "Quality validation completed",
            overall_score=overall_score,
            completeness=completeness_score,
            readability=readability_score,
            structure=structure_score,
            metadata=metadata_preservation
        )
        
        return quality_score
    
    def _check_completeness(self, original_file: str, result: ConversionResult) -> float:
        """Check conversion completeness based on content analysis.

        Args:
            original_file: Path to original document
            result: Conversion result

        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not result.content:
            return 0.0

        # Basic completeness checks
        content_length = len(result.content)
        word_count = len(result.content.split())

        # Minimum content requirements
        if content_length < self.min_content_length:
            return 0.1

        if word_count < self.min_word_count:
            return 0.2

        # Handle different file types with appropriate completeness assessment
        try:
            file_path = Path(original_file)
            file_size = file_path.stat().st_size
            file_extension = file_path.suffix.lower()

            # For media files (audio/video), use different heuristics
            if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wav', '.m4a', '.flac']:
                # Media files: assess based on transcription quality
                # For video/audio, expect 1-5 words per second of content
                if result.metadata and 'duration' in result.metadata:
                    duration_seconds = result.metadata['duration']
                    expected_words = duration_seconds * 2  # Conservative estimate: 2 words/second

                    if word_count >= expected_words * 0.5:  # At least 50% of expected
                        return 0.9
                    elif word_count >= expected_words * 0.25:  # At least 25% of expected
                        return 0.7
                    else:
                        return 0.5
                else:
                    # No duration metadata, use content-based assessment
                    # For media files, any substantial transcription is good
                    if word_count > 100:
                        return 0.8
                    elif word_count > 50:
                        return 0.6
                    else:
                        return 0.4

            # For document files, use file size heuristics
            else:
                # Rough heuristic: expect ~1-10 chars per byte for text content
                expected_min_chars = file_size * 0.1
                expected_max_chars = file_size * 10

                if content_length < expected_min_chars:
                    # Content seems too short
                    ratio = content_length / expected_min_chars
                    return min(0.8, ratio)
                elif content_length > expected_max_chars:
                    # Content seems reasonable (could be expanded with formatting)
                    return 0.9
                else:
                    # Content length in reasonable range
                    return 0.85

        except Exception as e:
            logger.warning("Could not assess completeness from file size", error=str(e))
            # Fallback to content-based assessment
            return min(0.8, math.log10(content_length) / 4)  # Log scale up to 10k chars
    
    def _check_readability(self, content: str) -> float:
        """Check readability and coherence of converted content.
        
        Args:
            content: Converted markdown content
            
        Returns:
            Readability score (0.0 to 1.0)
        """
        if not content:
            return 0.0
        
        score = 0.0
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', content)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) > 0:
            score += 0.3
        
        # Check for reasonable word distribution
        words = content.split()
        if len(words) > 0:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 3 <= avg_word_length <= 8:  # Reasonable average word length
                score += 0.2
        
        # Check for proper capitalization
        capitalized_sentences = sum(1 for s in valid_sentences if s and s[0].isupper())
        if len(valid_sentences) > 0:
            capitalization_ratio = capitalized_sentences / len(valid_sentences)
            score += capitalization_ratio * 0.2
        
        # Check for excessive noise (repeated characters, garbled text)
        noise_patterns = [
            r'(.)\1{5,}',  # Repeated characters
            r'[^\w\s]{10,}',  # Long sequences of special characters
            r'\b\w{20,}\b',  # Extremely long words (likely garbled)
        ]
        
        noise_count = sum(len(re.findall(pattern, content)) for pattern in noise_patterns)
        noise_ratio = noise_count / len(words) if words else 0
        
        if noise_ratio < self.max_noise_ratio:
            score += 0.3
        else:
            score += max(0, 0.3 * (1 - noise_ratio / self.max_noise_ratio))
        
        return min(1.0, score)
    
    def _check_structure(self, content: str) -> float:
        """Check markdown structure quality.
        
        Args:
            content: Converted markdown content
            
        Returns:
            Structure score (0.0 to 1.0)
        """
        if not content:
            return 0.0
        
        score = 0.0
        
        # Check for proper markdown headers
        headers = re.findall(r'^#{1,6}\s+.+$', content, re.MULTILINE)
        if headers:
            score += 0.3
            
            # Check for hierarchical structure
            header_levels = [len(h.split()[0]) for h in headers]
            if len(set(header_levels)) > 1:  # Multiple header levels
                score += 0.1
        
        # Check for proper paragraph structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            score += 0.2
        
        # Check for proper list formatting
        lists = re.findall(r'^[\s]*[-*+]\s+.+$', content, re.MULTILINE)
        if lists:
            score += 0.1
        
        # Check for proper table formatting
        tables = re.findall(r'\|.*\|', content)
        if tables:
            score += 0.1
        
        # Check for proper link formatting
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        if links:
            score += 0.1
        
        # Check for excessive empty lines or formatting issues
        empty_lines = content.count('\n\n\n')
        if empty_lines < len(content) / 100:  # Less than 1% empty line groups
            score += 0.1
        
        return min(1.0, score)
    
    def _check_metadata_preservation(self, result: ConversionResult) -> float:
        """Check how well metadata was preserved and extracted.
        
        Args:
            result: Conversion result
            
        Returns:
            Metadata preservation score (0.0 to 1.0)
        """
        if not result.metadata:
            return 0.0
        
        score = 0.0
        
        # Check for essential metadata fields
        essential_fields = ['title', 'author', 'created_date', 'format']
        present_fields = sum(1 for field in essential_fields if field in result.metadata)
        score += (present_fields / len(essential_fields)) * 0.4
        
        # Check for format-specific metadata
        if result.original_format == 'pdf':
            pdf_fields = ['page_count', 'word_count']
            present_pdf = sum(1 for field in pdf_fields if field in result.metadata)
            score += (present_pdf / len(pdf_fields)) * 0.2
        elif result.original_format in ['audio', 'video']:
            media_fields = ['duration', 'language']
            present_media = sum(1 for field in media_fields if field in result.metadata)
            score += (present_media / len(media_fields)) * 0.2
        
        # Check for processing metadata
        processing_fields = ['processing_time', 'converter_used']
        present_processing = sum(1 for field in processing_fields if hasattr(result, field) and getattr(result, field))
        score += (present_processing / len(processing_fields)) * 0.2
        
        # Check for content-derived metadata
        if result.content:
            if 'word_count' not in result.metadata:
                result.metadata['word_count'] = len(result.content.split())
            score += 0.2
        
        return min(1.0, score)
    
    def assess_batch_quality(self, results: List[ConversionResult]) -> Dict[str, float]:
        """Assess quality across a batch of conversion results.
        
        Args:
            results: List of conversion results
            
        Returns:
            Dictionary with batch quality metrics
        """
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success and r.quality_score]
        
        if not successful_results:
            return {'success_rate': 0.0}
        
        # Calculate aggregate metrics
        success_rate = len(successful_results) / len(results)
        avg_quality = sum(r.quality_score.overall_score for r in successful_results) / len(successful_results)
        avg_completeness = sum(r.quality_score.completeness_score for r in successful_results) / len(successful_results)
        avg_readability = sum(r.quality_score.readability_score for r in successful_results) / len(successful_results)
        avg_structure = sum(r.quality_score.structure_score for r in successful_results) / len(successful_results)
        
        return {
            'success_rate': success_rate,
            'average_quality': avg_quality,
            'average_completeness': avg_completeness,
            'average_readability': avg_readability,
            'average_structure': avg_structure,
            'total_processed': len(results),
            'successful_conversions': len(successful_results)
        }
