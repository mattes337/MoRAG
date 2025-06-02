"""Integration tests for enhanced summarization pipeline."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from morag.services.summarization import (
    enhanced_summarization_service,
    SummaryConfig,
    SummaryStrategy,
    DocumentType
)
from morag.tasks.document_tasks import _process_document_impl
from morag.services.embedding import SummaryResult, EmbeddingResult

@pytest.fixture
def mock_gemini_service():
    """Mock Gemini service for integration tests."""
    with patch('morag.services.summarization.gemini_service') as mock:
        mock.generate_summary = AsyncMock(return_value=SummaryResult(
            summary="Enhanced test summary with improved quality and coherence.",
            token_count=10,
            model="gemini-2.0-flash-001"
        ))
        yield mock

@pytest.fixture
def mock_qdrant_service():
    """Mock Qdrant service for integration tests."""
    with patch('morag.tasks.document_tasks.qdrant_service') as mock:
        mock.store_chunks = AsyncMock(return_value=["point_1", "point_2"])
        yield mock

@pytest.fixture
def mock_enhanced_gemini_service():
    """Mock enhanced Gemini service with batch operations."""
    with patch('morag.tasks.document_tasks.gemini_service') as mock:
        # Mock summary generation
        mock.generate_summary = AsyncMock(return_value=SummaryResult(
            summary="Enhanced summary with better quality assessment.",
            token_count=12,
            model="gemini-2.0-flash-001"
        ))
        
        # Mock batch embedding generation
        mock.generate_embeddings_batch = AsyncMock(return_value=[
            EmbeddingResult(
                embedding=[0.1] * 768,
                token_count=10,
                model="text-embedding-004"
            )
        ])
        yield mock

class TestEnhancedSummarizationIntegration:
    """Test enhanced summarization integration with document processing."""
    
    @pytest.mark.asyncio
    async def test_enhanced_summary_with_academic_content(self, mock_gemini_service):
        """Test enhanced summarization with academic content."""
        academic_text = """
        This research study presents a novel methodology for analyzing complex datasets.
        The experiment was conducted using a randomized controlled trial design.
        Our hypothesis stated that the new algorithm would significantly improve accuracy.
        The analysis of results shows a 25% improvement over existing methods.
        In conclusion, this methodology offers substantial benefits for data analysis.
        """
        
        config = SummaryConfig(
            strategy=SummaryStrategy.ABSTRACTIVE,
            document_type=DocumentType.ACADEMIC,
            style="abstract"
        )
        
        result = await enhanced_summarization_service.generate_summary(academic_text, config)
        
        assert result.strategy == SummaryStrategy.ABSTRACTIVE
        assert result.config.document_type == DocumentType.ACADEMIC
        assert result.quality.overall > 0.0
        assert result.processing_time > 0.0
        assert len(result.summary) > 0
    
    @pytest.mark.asyncio
    async def test_enhanced_summary_with_technical_content(self, mock_gemini_service):
        """Test enhanced summarization with technical content."""
        technical_text = """
        To implement this algorithm, follow these configuration steps:
        1. Install the required dependencies using pip install requirements.txt
        2. Configure the system settings in the config.yaml file
        3. Run the setup function to initialize the database
        4. Execute the main procedure using python main.py
        The documentation provides detailed implementation guidelines.
        """
        
        config = SummaryConfig(
            strategy=SummaryStrategy.HYBRID,
            document_type=DocumentType.TECHNICAL,
            preserve_structure=True
        )
        
        result = await enhanced_summarization_service.generate_summary(technical_text, config)
        
        assert result.strategy == SummaryStrategy.HYBRID
        assert result.config.document_type == DocumentType.TECHNICAL
        assert result.config.preserve_structure is True
    
    @pytest.mark.asyncio
    async def test_enhanced_summary_with_long_content(self, mock_gemini_service):
        """Test enhanced summarization with long content requiring hierarchical approach."""
        # Create long content
        long_text = """
        Chapter 1: Introduction
        This chapter introduces the fundamental concepts and background information.
        """ + " ".join([f"This is sentence {i} with important information." for i in range(200)])
        
        config = SummaryConfig(
            strategy=SummaryStrategy.HIERARCHICAL,
            context_window=100,  # Small window to force chunking
            max_length=150
        )
        
        result = await enhanced_summarization_service.generate_summary(long_text, config)
        
        assert result.strategy == SummaryStrategy.HIERARCHICAL
        assert len(result.summary.split()) <= config.max_length * 1.2  # Allow some flexibility
        assert mock_gemini_service.generate_summary.call_count > 1  # Multiple calls for chunks
    
    @pytest.mark.asyncio
    async def test_enhanced_summary_with_focus_areas(self, mock_gemini_service):
        """Test enhanced summarization with specific focus areas."""
        business_text = """
        The quarterly revenue report shows significant growth in all sectors.
        Market analysis indicates strong customer demand for our products.
        Strategic decisions were made to expand operations to new regions.
        Budget allocation for next quarter includes increased marketing spend.
        Action items include hiring additional sales staff and opening new offices.
        """
        
        config = SummaryConfig(
            strategy=SummaryStrategy.CONTEXTUAL,
            focus_areas=["revenue", "strategic decisions", "action items"],
            document_type=DocumentType.BUSINESS
        )
        
        result = await enhanced_summarization_service.generate_summary(business_text, config)
        
        assert result.strategy == SummaryStrategy.CONTEXTUAL
        assert len(result.config.focus_areas) == 3
    
    @pytest.mark.asyncio
    async def test_adaptive_configuration(self, mock_gemini_service):
        """Test adaptive configuration based on content analysis."""
        # Test with different types of content
        test_cases = [
            ("This research methodology analyzes experimental results.", DocumentType.ACADEMIC),
            ("Configure the system using these installation procedures.", DocumentType.TECHNICAL),
            ("Revenue growth and market strategy decisions.", DocumentType.BUSINESS),
            ("General information about various topics.", DocumentType.GENERAL)
        ]
        
        for text, expected_type in test_cases:
            result = await enhanced_summarization_service.generate_summary(text)
            
            # The service should automatically detect document type
            assert isinstance(result.config.document_type, DocumentType)
            assert result.quality.overall >= 0.0
    
    @pytest.mark.asyncio
    async def test_quality_assessment_and_refinement(self, mock_gemini_service):
        """Test quality assessment and refinement functionality."""
        # Configure for high quality threshold to trigger refinement
        config = SummaryConfig(
            quality_threshold=0.9,  # High threshold
            enable_refinement=True,
            max_length=100
        )
        
        text = "This is a test document with important information that needs summarization."
        
        result = await enhanced_summarization_service.generate_summary(text, config)
        
        assert result.config.enable_refinement is True
        assert result.quality.overall >= 0.0
        # Note: Refinement iterations depend on mock quality assessment
    
    @pytest.mark.asyncio
    async def test_document_processing_with_enhanced_summary(
        self, 
        mock_enhanced_gemini_service, 
        mock_qdrant_service
    ):
        """Test document processing pipeline with enhanced summarization."""
        # Create a test markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""
# Research Paper: Advanced Data Analysis

## Abstract
This paper presents a novel methodology for analyzing complex datasets.

## Introduction
The field of data analysis has evolved significantly in recent years.
New algorithms and techniques have emerged to handle large-scale data.

## Methodology
Our approach uses machine learning algorithms combined with statistical analysis.
The experimental design follows established research protocols.

## Results
The analysis shows significant improvements in accuracy and efficiency.
Performance metrics indicate a 25% improvement over existing methods.

## Conclusion
This methodology offers substantial benefits for data analysis applications.
Future work will focus on scaling to larger datasets.
            """)
            temp_path = f.name
        
        try:
            # Create mock task instance
            mock_task = MagicMock()
            mock_task.log_step = MagicMock()
            mock_task.update_progress = MagicMock()
            
            # Process document with enhanced summarization
            result = await _process_document_impl(
                mock_task,
                temp_path,
                "document",
                {"test": True, "enhanced_summary": True}
            )
            
            # Verify processing completed successfully
            assert result["status"] == "success"
            assert result["chunks_processed"] > 0
            assert result["word_count"] > 0
            
            # Verify enhanced summarization was used
            assert mock_enhanced_gemini_service.generate_summary.called
            assert mock_enhanced_gemini_service.generate_embeddings_batch.called
            assert mock_qdrant_service.store_chunks.called
            
            # Verify progress tracking
            assert mock_task.update_progress.called
            assert mock_task.log_step.called
            
        finally:
            Path(temp_path).unlink()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_enhanced_summary(self, mock_gemini_service):
        """Test error handling in enhanced summarization."""
        # Configure mock to raise an exception
        mock_gemini_service.generate_summary.side_effect = Exception("API Error")
        
        text = "Test content for error handling."
        
        with pytest.raises(Exception):
            await enhanced_summarization_service.generate_summary(text)
    
    @pytest.mark.asyncio
    async def test_extractive_summary_with_short_text(self, mock_gemini_service):
        """Test extractive summarization with very short text."""
        short_text = "Short text."
        config = SummaryConfig(strategy=SummaryStrategy.EXTRACTIVE)
        
        result = await enhanced_summarization_service.generate_summary(short_text, config)
        
        assert result.strategy == SummaryStrategy.EXTRACTIVE
        assert len(result.summary) > 0
    
    @pytest.mark.asyncio
    async def test_summary_strategies_comparison(self, mock_gemini_service):
        """Test different summarization strategies on the same content."""
        test_text = """
        This is a comprehensive test document with multiple sentences.
        It contains important information that needs to be summarized.
        The content includes key points and supporting details.
        Various summarization strategies should handle this differently.
        """
        
        strategies = [
            SummaryStrategy.EXTRACTIVE,
            SummaryStrategy.ABSTRACTIVE,
            SummaryStrategy.HYBRID
        ]
        
        results = []
        for strategy in strategies:
            config = SummaryConfig(strategy=strategy, max_length=50)
            result = await enhanced_summarization_service.generate_summary(test_text, config)
            results.append(result)
        
        # Verify all strategies produced results
        assert len(results) == 3
        assert all(result.summary for result in results)
        assert all(result.quality.overall >= 0.0 for result in results)
        
        # Verify strategies are correctly recorded
        for i, strategy in enumerate(strategies):
            assert results[i].strategy == strategy

class TestPerformanceAndScaling:
    """Test performance and scaling aspects of enhanced summarization."""
    
    @pytest.mark.asyncio
    async def test_large_document_processing(self, mock_gemini_service):
        """Test processing of large documents."""
        # Create a large document
        large_text = "\n".join([
            f"Paragraph {i}: " + " ".join([f"sentence {j}" for j in range(20)])
            for i in range(50)
        ])
        
        config = SummaryConfig(
            strategy=SummaryStrategy.HIERARCHICAL,
            max_length=200
        )
        
        result = await enhanced_summarization_service.generate_summary(large_text, config)
        
        assert result.processing_time > 0
        assert len(result.summary.split()) <= config.max_length * 1.2
    
    @pytest.mark.asyncio
    async def test_concurrent_summarization(self, mock_gemini_service):
        """Test concurrent summarization requests."""
        import asyncio
        
        texts = [
            f"Test document {i} with content to summarize." for i in range(5)
        ]
        
        # Process multiple documents concurrently
        tasks = [
            enhanced_summarization_service.generate_summary(text)
            for text in texts
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(result.summary for result in results)
        assert all(result.processing_time > 0 for result in results)
