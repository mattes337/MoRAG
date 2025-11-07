"""Tests for MoRAG Pipeline.

This module contains tests for the Pipeline class.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from morag_services.services import MoRAGServices, ProcessingResult, ContentType
from morag_services.pipeline import Pipeline, PipelineStep, PipelineContext, PipelineStepType

@pytest.fixture
def services():
    """Create a MoRAGServices instance for testing."""
    # Create services with mocked specialized services
    services = MoRAGServices()

    # Mock all specialized services
    services.document_service = AsyncMock()
    services.audio_service = AsyncMock()
    services.video_service = AsyncMock()
    services.image_service = AsyncMock()
    services.embedding_service = AsyncMock()
    services.web_service = AsyncMock()
    services.youtube_service = AsyncMock()

    # Mock process_content method
    services.process_content = AsyncMock()
    services.process_batch = AsyncMock()
    services.generate_embeddings = AsyncMock()

    return services

@pytest.fixture
def pipeline(services):
    """Create a Pipeline instance for testing."""
    return Pipeline(services, name="test_pipeline")

class TestPipelineStep:
    """Tests for PipelineStep."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful step execution."""
        # Create mock function
        async def mock_fn(data, context):
            return data.upper()

        # Create step
        step = PipelineStep(
            name="test_step",
            step_type=PipelineStepType.TRANSFORM,
            process_fn=mock_fn
        )

        # Execute step
        context = PipelineContext()
        result = await step.execute("test", context)

        # Verify result
        assert result == "TEST"
        assert len(context.errors) == 0

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """Test step execution with error."""
        # Create mock function that raises exception
        async def mock_fn(data, context):
            raise ValueError("Test error")

        # Create step
        step = PipelineStep(
            name="test_step",
            step_type=PipelineStepType.TRANSFORM,
            process_fn=mock_fn
        )

        # Execute step
        context = PipelineContext()

        # Verify exception is raised
        with pytest.raises(Exception):
            await step.execute("test", context)

        # Verify error is recorded in context
        assert len(context.errors) == 1
        assert "test" in context.errors
        assert "Test error" in context.errors["test"]

class TestPipeline:
    """Tests for Pipeline."""

    def test_add_step(self, pipeline):
        """Test adding steps to pipeline."""
        # Create mock step
        async def mock_fn(data, context):
            return data

        step = PipelineStep(
            name="test_step",
            step_type=PipelineStepType.TRANSFORM,
            process_fn=mock_fn
        )

        # Add step
        result = pipeline.add_step(step)

        # Verify step is added
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0] == step

        # Verify method chaining works
        assert result == pipeline

    def test_process_content_step(self, pipeline, services):
        """Test adding process_content step."""
        # Add step
        result = pipeline.process_content("test_process")

        # Verify step is added
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "test_process"
        assert pipeline.steps[0].step_type == PipelineStepType.PROCESS

        # Verify method chaining works
        assert result == pipeline

    def test_extract_text_step(self, pipeline):
        """Test adding extract_text step."""
        # Add step
        result = pipeline.extract_text("test_extract")

        # Verify step is added
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "test_extract"
        assert pipeline.steps[0].step_type == PipelineStepType.TRANSFORM
        assert pipeline.steps[0].input_key == "results"
        assert pipeline.steps[0].output_key == "texts"

        # Verify method chaining works
        assert result == pipeline

    def test_extract_metadata_step(self, pipeline):
        """Test adding extract_metadata step."""
        # Add step
        result = pipeline.extract_metadata("test_metadata")

        # Verify step is added
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "test_metadata"
        assert pipeline.steps[0].step_type == PipelineStepType.TRANSFORM
        assert pipeline.steps[0].input_key == "results"
        assert pipeline.steps[0].output_key == "metadata"

        # Verify method chaining works
        assert result == pipeline

    def test_generate_embeddings_step(self, pipeline):
        """Test adding generate_embeddings step."""
        # Add step
        result = pipeline.generate_embeddings("test_embeddings")

        # Verify step is added
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "test_embeddings"
        assert pipeline.steps[0].step_type == PipelineStepType.EMBED
        assert pipeline.steps[0].input_key == "texts"
        assert pipeline.steps[0].output_key == "embeddings"

        # Verify method chaining works
        assert result == pipeline

    def test_custom_step(self, pipeline):
        """Test adding custom step."""
        # Create mock function
        async def mock_fn(data, context):
            return data

        # Add step
        result = pipeline.custom_step(
            name="test_custom",
            process_fn=mock_fn,
            input_key="input",
            output_key="output",
            config={"key": "value"}
        )

        # Verify step is added
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "test_custom"
        assert pipeline.steps[0].step_type == PipelineStepType.CUSTOM
        assert pipeline.steps[0].process_fn == mock_fn
        assert pipeline.steps[0].input_key == "input"
        assert pipeline.steps[0].output_key == "output"
        assert pipeline.steps[0].config == {"key": "value"}

        # Verify method chaining works
        assert result == pipeline

    @pytest.mark.asyncio
    async def test_execute_empty_pipeline(self, pipeline):
        """Test executing empty pipeline."""
        # Execute pipeline
        context = await pipeline.execute()

        # Verify context
        assert context.current_step == 0
        assert context.total_steps == 0
        assert len(context.errors) == 0

    @pytest.mark.asyncio
    async def test_execute_process_step(self, pipeline, services):
        """Test executing pipeline with process step."""
        # Mock service response
        mock_result1 = ProcessingResult(
            content_type=ContentType.DOCUMENT,
            content_path="test.pdf",
            text_content="Document text",
            success=True
        )

        mock_result2 = ProcessingResult(
            content_type=ContentType.WEB,
            content_url="https://example.com",
            text_content="Web content",
            success=True
        )

        services.process_content.side_effect = [
            mock_result1,
            mock_result2
        ]

        # Add process step
        pipeline.process_content()

        # Execute pipeline
        context = await pipeline.execute(
            input_paths=["test.pdf"],
            input_urls=["https://example.com"]
        )

        # Verify context
        assert context.current_step == 1
        assert context.total_steps == 1
        assert len(context.results) == 2
        assert context.results["test.pdf"] == mock_result1
        assert context.results["https://example.com"] == mock_result2
        assert len(context.errors) == 0

    @pytest.mark.asyncio
    async def test_execute_transform_step(self, pipeline):
        """Test executing pipeline with transform step."""
        # Add results to context
        results = {
            "item1": ProcessingResult(
                content_type=ContentType.DOCUMENT,
                text_content="Text 1",
                success=True
            ),
            "item2": ProcessingResult(
                content_type=ContentType.WEB,
                text_content="Text 2",
                success=True
            )
        }

        # Create custom step
        async def extract_text(results, context):
            texts = {}
            for item_id, result in results.items():
                texts[item_id] = result.text_content.upper()
            return texts

        pipeline.custom_step(
            name="extract_text",
            process_fn=extract_text,
            input_key="results",
            output_key="texts"
        )

        # Create context with results
        context = PipelineContext(results=results)
        context.total_steps = 1

        # Execute pipeline
        result_context = await pipeline.execute()
        result_context.results = results  # Manually set results since we're not using process_content step

        # Execute step directly
        await pipeline.steps[0].execute(results, result_context)

        # Verify context
        assert result_context.current_step == 1
        assert result_context.total_steps == 1
        assert len(result_context.texts) == 2
        assert result_context.texts["item1"] == "TEXT 1"
        assert result_context.texts["item2"] == "TEXT 2"

    @pytest.mark.asyncio
    async def test_execute_embedding_step(self, pipeline, services):
        """Test executing pipeline with embedding step."""
        # Mock embedding service
        services.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]

        # Add texts to context
        texts = {
            "item1": "Text 1",
            "item2": "Text 2"
        }

        # Add embedding step
        pipeline.generate_embeddings()

        # Create context with texts
        context = PipelineContext(texts=texts)
        context.total_steps = 1

        # Execute step directly
        await pipeline.steps[0].execute(texts, context)

        # Verify context
        assert len(context.embeddings) == 2
        assert context.embeddings["item1"] == [0.1, 0.2, 0.3]
        assert context.embeddings["item2"] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_execute_multi_step_pipeline(self, pipeline, services):
        """Test executing pipeline with multiple steps."""
        # Mock service responses
        mock_result = ProcessingResult(
            content_type=ContentType.DOCUMENT,
            content_path="test.pdf",
            text_content="Document text",
            metadata={"pages": 5},
            success=True
        )

        services.process_content.return_value = mock_result
        services.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        # Add steps
        pipeline.process_content()
        pipeline.extract_text()
        pipeline.extract_metadata()
        pipeline.generate_embeddings()

        # Execute pipeline
        context = await pipeline.execute(input_paths=["test.pdf"])

        # Verify context
        assert context.current_step == 4
        assert context.total_steps == 4
        assert len(context.results) == 1
        assert context.results["test.pdf"] == mock_result
        assert context.texts["test.pdf"] == "Document text"
        assert context.metadata["test.pdf"] == {"pages": 5}
        assert context.embeddings["test.pdf"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_process_batch(self, pipeline, services):
        """Test process_batch method."""
        # Mock service response
        mock_result1 = ProcessingResult(
            content_type=ContentType.DOCUMENT,
            content_path="test.pdf",
            text_content="Document text",
            success=True
        )

        mock_result2 = ProcessingResult(
            content_type=ContentType.WEB,
            content_url="https://example.com",
            text_content="Web content",
            success=True
        )

        services.process_content.side_effect = [
            mock_result1,
            mock_result2
        ]

        # Add process step
        pipeline.process_content()

        # Process batch
        items = ["test.pdf", "https://example.com"]
        context = await pipeline.process_batch(items)

        # Verify context
        assert len(context.results) == 2
        assert context.results["test.pdf"] == mock_result1
        assert context.results["https://example.com"] == mock_result2

    def test_create_default_pipeline(self, services):
        """Test create_default_pipeline class method."""
        # Create default pipeline
        pipeline = Pipeline.create_default_pipeline(services)

        # Verify pipeline
        assert pipeline.name == "default"
        assert len(pipeline.steps) == 4
        assert pipeline.steps[0].name == "process_content"
        assert pipeline.steps[1].name == "extract_text"
        assert pipeline.steps[2].name == "extract_metadata"
        assert pipeline.steps[3].name == "generate_embeddings"
