"""Tests for MoRAG Pipeline Agent."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile

from morag.agents.morag_pipeline_agent import (
    MoRAGPipelineAgent,
    PipelineMode,
    ProcessingStage,
    IngestionOptions,
    ResolutionOptions,
    IngestionResult,
    ResolutionResult
)
from morag_services import ContentType
from morag_core.models import ProcessingResult


class TestMoRAGPipelineAgent:
    """Test the MoRAG pipeline agent."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_orchestrator(self):
        """Mock orchestrator for testing."""
        mock = MagicMock()
        mock.process_content = AsyncMock(return_value=ProcessingResult(
            content="Test content",
            metadata={"title": "Test Document"},
            processing_time=1.0,
            success=True
        ))
        return mock

    @pytest.fixture
    def pipeline_agent(self, temp_dir):
        """Create pipeline agent for testing."""
        return MoRAGPipelineAgent(intermediate_dir=temp_dir)

    @pytest.mark.asyncio
    async def test_agent_initialization(self, temp_dir):
        """Test pipeline agent initialization."""
        agent = MoRAGPipelineAgent(intermediate_dir=temp_dir)

        assert agent.intermediate_dir == temp_dir
        assert agent.orchestrator is not None
        assert agent.services is not None
        assert temp_dir.exists()

    @pytest.mark.asyncio
    async def test_basic_ingestion_pipeline(self, pipeline_agent, mock_orchestrator):
        """Test basic ingestion pipeline."""
        # Mock the orchestrator
        pipeline_agent.orchestrator = mock_orchestrator

        # Create ingestion options
        options = IngestionOptions(
            content_type=ContentType.DOCUMENT,
            enable_spacy_ner=True,
            enable_openie=True,
            generate_intermediate_files=True
        )

        # Run ingestion
        result = await pipeline_agent.process_ingestion("test_document.txt", options)

        # Verify result
        assert isinstance(result, IngestionResult)
        assert result.success
        assert result.document_id
        assert result.processing_time > 0
        assert len(result.intermediate_files) > 0

    @pytest.mark.asyncio
    async def test_basic_resolution_pipeline(self, pipeline_agent):
        """Test basic resolution pipeline."""
        # Create resolution options
        options = ResolutionOptions(
            max_depth=2,
            max_facts=10,
            enable_multi_hop=True,
            generate_intermediate_files=True
        )

        # Run resolution
        result = await pipeline_agent.process_resolution("What is AI?", options)

        # Verify result
        assert isinstance(result, ResolutionResult)
        assert result.success
        assert result.query == "What is AI?"
        assert result.response
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_ingestion_error_handling(self, pipeline_agent):
        """Test error handling in ingestion pipeline."""
        # Mock orchestrator to fail
        mock_orchestrator = MagicMock()
        mock_orchestrator.process_content = AsyncMock(return_value=ProcessingResult(
            content="",
            metadata={},
            processing_time=0.0,
            success=False,
            error_message="Processing failed"
        ))
        pipeline_agent.orchestrator = mock_orchestrator

        # Create ingestion options
        options = IngestionOptions(
            content_type=ContentType.DOCUMENT,
            generate_intermediate_files=False
        )

        # Run ingestion (should fail gracefully)
        result = await pipeline_agent.process_ingestion("invalid_source", options)

        # Verify error handling
        assert isinstance(result, IngestionResult)
        assert not result.success
        assert result.error_message
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_resolution_error_handling(self, pipeline_agent):
        """Test error handling in resolution pipeline."""
        # Create resolution options
        options = ResolutionOptions(
            max_depth=2,
            generate_intermediate_files=False
        )

        # Mock an exception in the resolution process
        with patch.object(pipeline_agent, '_process_basic_resolution', side_effect=Exception("Test error")):
            result = await pipeline_agent.process_resolution("Test query", options)

        # Verify error handling
        assert isinstance(result, ResolutionResult)
        assert not result.success
        assert result.error_message == "Test error"
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_intermediate_file_generation(self, pipeline_agent, mock_orchestrator, temp_dir):
        """Test intermediate file generation."""
        # Mock the orchestrator
        pipeline_agent.orchestrator = mock_orchestrator

        # Create ingestion options with intermediate files enabled
        options = IngestionOptions(
            content_type=ContentType.DOCUMENT,
            generate_intermediate_files=True
        )

        # Run ingestion
        result = await pipeline_agent.process_ingestion("test_document.txt", options)

        # Verify intermediate files were created
        assert result.success
        assert len(result.intermediate_files) > 0

        # Check that files actually exist
        for file_path in result.intermediate_files:
            assert file_path.exists()
            assert file_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_performance_metrics(self, pipeline_agent, mock_orchestrator):
        """Test performance metrics collection."""
        # Mock the orchestrator
        pipeline_agent.orchestrator = mock_orchestrator

        # Create ingestion options
        options = IngestionOptions(
            content_type=ContentType.DOCUMENT,
            generate_intermediate_files=False
        )

        # Run ingestion
        result = await pipeline_agent.process_ingestion("test_document.txt", options)

        # Get performance metrics
        metrics = pipeline_agent.get_performance_metrics()

        # Verify metrics
        assert 'total_processing_time' in metrics
        assert 'stage_timings' in metrics
        assert 'enhanced_components_available' in metrics
        assert metrics['total_processing_time'] > 0

    @pytest.mark.asyncio
    async def test_different_content_types(self, pipeline_agent, mock_orchestrator):
        """Test processing different content types."""
        # Mock the orchestrator
        pipeline_agent.orchestrator = mock_orchestrator

        content_types = [
            ContentType.DOCUMENT,
            ContentType.WEB,
            ContentType.AUDIO,
            ContentType.VIDEO
        ]

        for content_type in content_types:
            options = IngestionOptions(
                content_type=content_type,
                generate_intermediate_files=False
            )

            result = await pipeline_agent.process_ingestion(f"test_{content_type.value}", options)

            # Should succeed for all content types
            assert isinstance(result, IngestionResult)
            assert result.success

    @pytest.mark.asyncio
    async def test_stage_output_saving(self, pipeline_agent, temp_dir):
        """Test stage output saving functionality."""
        # Test saving stage output
        test_data = {"test": "data", "number": 42}

        file_path = await pipeline_agent._save_stage_output(
            ProcessingStage.CONTENT_CONVERSION,
            test_data,
            "test_source"
        )

        # Verify file was created
        assert file_path.exists()
        assert file_path.suffix == '.json'

        # Verify content
        import json
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data

    @pytest.mark.asyncio
    async def test_enhanced_components_fallback(self, pipeline_agent, mock_orchestrator):
        """Test fallback when enhanced components are not available."""
        # Mock the orchestrator
        pipeline_agent.orchestrator = mock_orchestrator

        # Disable enhanced components
        pipeline_agent.enhanced_graph_builder = None
        pipeline_agent.fact_retrieval_service = None

        # Test ingestion
        ingestion_options = IngestionOptions(
            content_type=ContentType.DOCUMENT,
            generate_intermediate_files=False
        )

        ingestion_result = await pipeline_agent.process_ingestion("test_doc", ingestion_options)
        assert ingestion_result.success
        assert ingestion_result.metadata.get('processing_mode') == 'basic'

        # Test resolution
        resolution_options = ResolutionOptions(
            max_depth=2,
            generate_intermediate_files=False
        )

        resolution_result = await pipeline_agent.process_resolution("test query", resolution_options)
        assert resolution_result.success
        assert resolution_result.metadata.get('processing_mode') == 'basic'
