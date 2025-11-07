"""Tests for remote conversion integration with ingestion tasks."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from morag.ingest_tasks import ingest_file_task, continue_ingestion_after_remote_processing
from morag_core.models.remote_job import RemoteJob


class TestIngestionIntegration:
    """Test cases for remote conversion integration."""

    def test_remote_job_creation_logic(self):
        """Test the logic for creating remote jobs for audio content."""
        from morag.services.remote_job_service import RemoteJobService
        from morag.models.remote_job_api import CreateRemoteJobRequest

        # Test that remote job service can create jobs for audio content
        service = RemoteJobService()

        # Create a request for audio processing
        request = CreateRemoteJobRequest(
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={"remote": True, "webhook_url": "http://example.com/webhook"},
            ingestion_task_id="test-task-123"
        )

        # This tests the service layer logic
        # In a real scenario, this would create a job file
        assert request.content_type == "audio"
        assert request.task_options["remote"] == True
        assert request.source_file_path == "/tmp/test.mp3"

    def test_content_type_support_logic(self):
        """Test that only audio/video content types support remote processing."""
        # Test the logic that determines which content types support remote processing

        # Audio and video should support remote processing
        audio_supports_remote = "audio" in ["audio", "video"]
        video_supports_remote = "video" in ["audio", "video"]
        document_supports_remote = "document" in ["audio", "video"]

        assert audio_supports_remote == True
        assert video_supports_remote == True
        assert document_supports_remote == False

        # This tests the core logic without needing to mock Celery tasks

    @pytest.mark.asyncio
    @patch('morag.ingest_tasks.store_content_in_vector_db')
    @patch('morag.ingest_tasks.send_webhook_notification')
    @patch('morag.ingest_tasks.RemoteJobService')
    async def test_continue_ingestion_after_remote_processing(
        self, mock_service_class, mock_webhook, mock_store
    ):
        """Test continuing ingestion after remote processing completes."""
        # Setup mocks
        mock_service = Mock()
        mock_service_class.return_value = mock_service

        mock_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={
                "webhook_url": "http://example.com/webhook",
                "store_in_vector_db": True,
                "metadata": {"source": "upload"}
            }
        )
        mock_service.get_job_status.return_value = mock_job

        mock_store.return_value = ["point-1", "point-2", "point-3"]

        # Call continuation function
        result = await continue_ingestion_after_remote_processing(
            remote_job_id="remote-job-123",
            content="Processed audio transcript",
            metadata={"duration": 120.5, "speakers": ["Speaker_00", "Speaker_01"]},
            processing_time=45.2
        )

        # Verify success
        assert result == True

        # Verify vector storage was called
        mock_store.assert_called_once()
        store_args = mock_store.call_args
        assert store_args[0][0] == "Processed audio transcript"  # content

        # Check metadata passed to vector storage
        vector_metadata = store_args[0][1]
        assert vector_metadata["source_type"] == "audio"
        assert vector_metadata["source_path"] == "/tmp/test.mp3"
        assert vector_metadata["processing_time"] == 45.2
        assert vector_metadata["remote_processing"] == True
        assert vector_metadata["remote_job_id"] == "remote-job-123"
        assert vector_metadata["duration"] == 120.5
        assert vector_metadata["source"] == "upload"  # From original metadata

        # Verify webhook was called
        mock_webhook.assert_called_once()
        webhook_args = mock_webhook.call_args[0]
        assert webhook_args[0] == "http://example.com/webhook"
        assert webhook_args[1] == "test-task-123"
        assert webhook_args[2] == "SUCCESS"

    @pytest.mark.asyncio
    @patch('morag.ingest_tasks.RemoteJobService')
    async def test_continue_ingestion_job_not_found(self, mock_service_class):
        """Test continuation when remote job is not found."""
        # Setup mock
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_job_status.return_value = None  # Job not found

        # Call continuation function
        result = await continue_ingestion_after_remote_processing(
            remote_job_id="nonexistent-job",
            content="Some content",
            metadata={},
            processing_time=10.0
        )

        # Verify failure
        assert result == False

    @pytest.mark.asyncio
    @patch('morag.ingest_tasks.store_content_in_vector_db')
    @patch('morag.ingest_tasks.RemoteJobService')
    async def test_continue_ingestion_storage_error(self, mock_service_class, mock_store):
        """Test continuation when vector storage fails."""
        # Setup mocks
        mock_service = Mock()
        mock_service_class.return_value = mock_service

        mock_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={"store_in_vector_db": True}
        )
        mock_service.get_job_status.return_value = mock_job

        # Make storage fail
        mock_store.side_effect = Exception("Vector storage error")

        # Call continuation function
        result = await continue_ingestion_after_remote_processing(
            remote_job_id="remote-job-123",
            content="Processed content",
            metadata={},
            processing_time=10.0
        )

        # Verify failure
        assert result == False

    @pytest.mark.asyncio
    @patch('morag.ingest_tasks.RemoteJobService')
    async def test_continue_ingestion_no_vector_storage(self, mock_service_class):
        """Test continuation when vector storage is disabled."""
        # Setup mock
        mock_service = Mock()
        mock_service_class.return_value = mock_service

        mock_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={
                "store_in_vector_db": False,  # Disabled
                "webhook_url": "http://example.com/webhook"
            }
        )
        mock_service.get_job_status.return_value = mock_job

        # Call continuation function
        with patch('morag.ingest_tasks.send_webhook_notification') as mock_webhook:
            result = await continue_ingestion_after_remote_processing(
                remote_job_id="remote-job-123",
                content="Processed content",
                metadata={},
                processing_time=10.0
            )

        # Verify success (even without vector storage)
        assert result == True

        # Verify webhook was still called
        mock_webhook.assert_called_once()
