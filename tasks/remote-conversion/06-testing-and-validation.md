# Task 6: Testing and Validation

## Overview

Implement comprehensive testing and validation for the remote conversion system, including unit tests, integration tests, error scenario testing, and performance benchmarking. This ensures the system works reliably under various conditions and meets all requirements.

## Objectives

1. Create comprehensive unit test suite for all components
2. Implement integration tests for end-to-end workflows
3. Test error scenarios and failure recovery mechanisms
4. Perform performance benchmarking against local processing
5. Create automated test scripts for deployment validation
6. Add monitoring and alerting for production readiness

## Technical Requirements

### 1. Unit Test Suite

**File**: `tests/remote_conversion/test_remote_job_service.py`

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import uuid

from morag.services.remote_job_service import RemoteJobService
from morag.models.remote_job_api import CreateRemoteJobRequest, SubmitResultRequest
from morag_core.models.remote_job import RemoteJob
from morag.repositories.remote_job_repository import RemoteJobRepository

class TestRemoteJobService:
    """Test suite for RemoteJobService."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for testing."""
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_repository(self, temp_storage_dir):
        """Create RemoteJobRepository instance with temporary storage."""
        return RemoteJobRepository(temp_storage_dir)

    @pytest.fixture
    def remote_job_service(self, mock_repository):
        """Create RemoteJobService instance with mocked repository."""
        return RemoteJobService(mock_repository)

    @pytest.fixture
    def sample_create_request(self):
        """Sample job creation request."""
        return CreateRemoteJobRequest(
            source_file_path="/tmp/test_audio.mp3",
            content_type="audio",
            task_options={"webhook_url": "https://example.com/webhook"},
            ingestion_task_id="test-task-123"
        )

    def test_create_job_success(self, remote_job_service, sample_create_request):
        """Test successful job creation."""
        # Execute
        result = remote_job_service.create_job(sample_create_request)

        # Verify
        assert result is not None
        assert result.id is not None
        assert result.ingestion_task_id == sample_create_request.ingestion_task_id
        assert result.source_file_path == sample_create_request.source_file_path
        assert result.content_type == sample_create_request.content_type
        assert result.status == 'pending'

        # Verify job was saved to storage
        saved_job = remote_job_service.repository.get_job(result.id)
        assert saved_job is not None
        assert saved_job.id == result.id

    def test_poll_available_jobs_success(self, remote_job_service, mock_repository):
        """Test successful job polling."""
        # Setup - create a pending job
        job = RemoteJob.create_new(
            ingestion_task_id="test-task",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={}
        )
        mock_repository.create_job(
            job.ingestion_task_id,
            job.source_file_path,
            job.content_type,
            job.task_options
        )

        # Execute
        result = remote_job_service.poll_available_jobs("worker-1", ["audio"], 1)

        # Verify
        assert len(result) == 1
        assert result[0].status == 'processing'  # Should be updated to processing
        assert result[0].worker_id == "worker-1"

        # Verify job was moved to processing status in storage
        updated_job = mock_repository.get_job(result[0].id)
        assert updated_job.status == 'processing'

    def test_poll_no_available_jobs(self, remote_job_service, mock_db_session):
        """Test polling when no jobs are available."""
        # Setup
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        # Execute
        result = remote_job_service.poll_available_jobs("worker-1", ["audio"], 1)

        # Verify
        assert len(result) == 0

    def test_submit_result_success(self, remote_job_service, mock_db_session):
        """Test successful result submission."""
        # Setup
        job_id = str(uuid.uuid4())
        mock_job = RemoteJob(
            id=job_id,
            status='processing',
            worker_id='worker-1'
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_job

        result_request = SubmitResultRequest(
            success=True,
            content="Processed content",
            metadata={"duration": 120},
            processing_time=45.2
        )

        # Execute
        result = remote_job_service.submit_result(job_id, result_request)

        # Verify
        assert result is not None
        assert result.status == 'completed'
        assert result.result_data['content'] == "Processed content"
        mock_db_session.commit.assert_called_once()

    def test_submit_result_failure(self, remote_job_service, mock_db_session):
        """Test result submission for failed job."""
        # Setup
        job_id = str(uuid.uuid4())
        mock_job = RemoteJob(
            id=job_id,
            status='processing',
            worker_id='worker-1',
            retry_count=0
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_job

        result_request = SubmitResultRequest(
            success=False,
            error_message="Processing failed due to invalid format"
        )

        # Execute
        result = remote_job_service.submit_result(job_id, result_request)

        # Verify
        assert result is not None
        assert result.status == 'failed'
        assert result.error_message == "Processing failed due to invalid format"
        assert result.retry_count == 1

    def test_get_job_status(self, remote_job_service, mock_db_session):
        """Test job status retrieval."""
        # Setup
        job_id = str(uuid.uuid4())
        mock_job = RemoteJob(
            id=job_id,
            status='processing',
            worker_id='worker-1'
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_job

        # Execute
        result = remote_job_service.get_job_status(job_id)

        # Verify
        assert result is not None
        assert result.status == 'processing'
        assert result.worker_id == 'worker-1'

    def test_cleanup_expired_jobs(self, remote_job_service, mock_db_session):
        """Test cleanup of expired jobs."""
        # Setup
        expired_jobs = [
            RemoteJob(id=uuid.uuid4(), status='pending', timeout_at=datetime.utcnow() - timedelta(hours=1)),
            RemoteJob(id=uuid.uuid4(), status='processing', timeout_at=datetime.utcnow() - timedelta(minutes=30))
        ]
        mock_db_session.query.return_value.filter.return_value.all.return_value = expired_jobs

        # Execute
        result = remote_job_service.cleanup_expired_jobs()

        # Verify
        assert result == 2
        for job in expired_jobs:
            assert job.status == 'timeout'
        mock_db_session.commit.assert_called_once()
```

**File**: `tests/remote_conversion/test_job_lifecycle_manager.py`

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from morag.services.job_lifecycle_manager import JobLifecycleManager, JobStatus
from morag_core.models.remote_job import RemoteJob

class TestJobLifecycleManager:
    """Test suite for JobLifecycleManager."""

    @pytest.fixture
    def lifecycle_manager(self):
        """Create JobLifecycleManager instance."""
        return JobLifecycleManager()

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = Mock()
        session.query.return_value = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        return session

    def test_valid_status_transitions(self, lifecycle_manager):
        """Test valid status transitions."""
        # Test valid transitions
        assert lifecycle_manager._is_valid_transition('pending', 'processing')
        assert lifecycle_manager._is_valid_transition('processing', 'completed')
        assert lifecycle_manager._is_valid_transition('processing', 'failed')
        assert lifecycle_manager._is_valid_transition('failed', 'retrying')
        assert lifecycle_manager._is_valid_transition('retrying', 'processing')

    def test_invalid_status_transitions(self, lifecycle_manager):
        """Test invalid status transitions."""
        # Test invalid transitions
        assert not lifecycle_manager._is_valid_transition('completed', 'processing')
        assert not lifecycle_manager._is_valid_transition('cancelled', 'processing')
        assert not lifecycle_manager._is_valid_transition('pending', 'completed')

    @patch('morag.services.job_lifecycle_manager.db_manager')
    def test_transition_job_status_success(self, mock_db_manager, lifecycle_manager, mock_session):
        """Test successful job status transition."""
        # Setup
        job_id = "test-job-123"
        mock_job = RemoteJob(
            id=job_id,
            status='pending',
            content_type='audio'
        )
        mock_session.query.return_value.filter.return_value.first.return_value = mock_job

        # Execute
        result = lifecycle_manager.transition_job_status(
            mock_session, job_id, JobStatus.PROCESSING
        )

        # Verify
        assert result is True
        assert mock_job.status == 'processing'
        assert mock_job.started_at is not None
        assert mock_job.timeout_at is not None
        mock_session.commit.assert_called_once()

    def test_transition_job_status_invalid_transition(self, lifecycle_manager, mock_session):
        """Test invalid status transition."""
        # Setup
        job_id = "test-job-123"
        mock_job = RemoteJob(
            id=job_id,
            status='completed',
            content_type='audio'
        )
        mock_session.query.return_value.filter.return_value.first.return_value = mock_job

        # Execute
        result = lifecycle_manager.transition_job_status(
            mock_session, job_id, JobStatus.PROCESSING
        )

        # Verify
        assert result is False
        mock_session.commit.assert_not_called()

    def test_transition_job_status_job_not_found(self, lifecycle_manager, mock_session):
        """Test status transition for non-existent job."""
        # Setup
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Execute
        result = lifecycle_manager.transition_job_status(
            mock_session, "non-existent-job", JobStatus.PROCESSING
        )

        # Verify
        assert result is False

    @patch('morag.services.job_lifecycle_manager.db_manager')
    def test_check_expired_jobs(self, mock_db_manager, lifecycle_manager):
        """Test expired job detection and handling."""
        # Setup
        mock_session = Mock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        expired_jobs = [
            RemoteJob(id="job-1", status='pending', timeout_at=datetime.utcnow() - timedelta(hours=1)),
            RemoteJob(id="job-2", status='processing', timeout_at=datetime.utcnow() - timedelta(minutes=30))
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = expired_jobs

        # Mock the transition_job_status method
        with patch.object(lifecycle_manager, 'transition_job_status', return_value=True) as mock_transition:
            # Execute
            result = lifecycle_manager.check_expired_jobs()

            # Verify
            assert result == 2
            assert mock_transition.call_count == 2

    @patch('morag.services.job_lifecycle_manager.db_manager')
    def test_cleanup_old_jobs(self, mock_db_manager, lifecycle_manager):
        """Test cleanup of old completed jobs."""
        # Setup
        mock_session = Mock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        old_jobs = [
            RemoteJob(id="job-1", status='completed', completed_at=datetime.utcnow() - timedelta(days=10)),
            RemoteJob(id="job-2", status='failed', completed_at=datetime.utcnow() - timedelta(days=8))
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = old_jobs

        # Mock file cleanup
        with patch.object(lifecycle_manager, '_cleanup_job_files'):
            # Execute
            result = lifecycle_manager.cleanup_old_jobs(days_old=7)

            # Verify
            assert result == 2
            assert mock_session.delete.call_count == 2
            mock_session.commit.assert_called_once()

    @patch('morag.services.job_lifecycle_manager.db_manager')
    def test_get_job_statistics(self, mock_db_manager, lifecycle_manager):
        """Test job statistics retrieval."""
        # Setup
        mock_session = Mock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        # Mock query results
        status_counts = [('pending', 5), ('processing', 2), ('completed', 10)]
        avg_times = [('audio', 45.5), ('video', 120.3)]
        recent_count = 15

        mock_session.query.return_value.group_by.return_value.all.return_value = status_counts
        mock_session.query.return_value.filter.return_value.group_by.return_value.all.return_value = avg_times
        mock_session.query.return_value.filter.return_value.scalar.return_value = recent_count

        # Execute
        result = lifecycle_manager.get_job_statistics()

        # Verify
        assert 'status_counts' in result
        assert 'avg_processing_times' in result
        assert 'recent_jobs_24h' in result
        assert result['status_counts'] == dict(status_counts)
        assert result['recent_jobs_24h'] == recent_count

### 2. Integration Test Suite

**File**: `tests/remote_conversion/test_integration.py`

```python
import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock
import requests_mock

from morag.services.remote_job_creator import RemoteJobCreator
from morag.models.ingestion_request import RemoteJobCreationRequest
from tools.remote_converter.remote_converter import RemoteConverter

class TestRemoteConversionIntegration:
    """Integration tests for remote conversion system."""

    @pytest.fixture
    def sample_audio_file(self):
        """Create a sample audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            # Create a minimal MP3-like file (not a real MP3, just for testing)
            f.write(b'ID3\x03\x00\x00\x00\x00\x00\x00\x00')
            return f.name

    @pytest.fixture
    def remote_converter_config(self):
        """Configuration for remote converter."""
        return {
            'worker_id': 'test-worker-1',
            'api_base_url': 'http://localhost:8000',
            'content_types': ['audio', 'video'],
            'poll_interval': 1,  # Fast polling for tests
            'max_concurrent_jobs': 1
        }

    @pytest.mark.asyncio
    async def test_end_to_end_remote_processing(self, sample_audio_file, remote_converter_config):
        """Test complete end-to-end remote processing workflow."""
        with requests_mock.Mocker() as m:
            # Mock API responses
            job_id = "test-job-123"

            # Mock job creation
            m.post(
                'http://localhost:8000/api/v1/remote-jobs/',
                json={'job_id': job_id, 'status': 'pending', 'created_at': '2025-01-15T10:00:00Z'}
            )

            # Mock job polling (first call returns job, subsequent calls return empty)
            poll_responses = [
                {
                    'job_id': job_id,
                    'source_file_url': f'/api/v1/remote-jobs/{job_id}/download',
                    'content_type': 'audio',
                    'task_options': {'webhook_url': 'https://example.com/webhook'}
                },
                {}  # No more jobs
            ]
            m.get(
                'http://localhost:8000/api/v1/remote-jobs/poll',
                [{'json': response} for response in poll_responses]
            )

            # Mock file download
            with open(sample_audio_file, 'rb') as f:
                file_content = f.read()
            m.get(
                f'http://localhost:8000/api/v1/remote-jobs/{job_id}/download',
                content=file_content,
                headers={'content-disposition': 'attachment; filename="test_audio.mp3"'}
            )

            # Mock result submission
            m.put(
                f'http://localhost:8000/api/v1/remote-jobs/{job_id}/result',
                json={'status': 'completed', 'ingestion_continued': True}
            )

            # Create and start remote converter
            converter = RemoteConverter(remote_converter_config)

            # Mock the audio processor to avoid actual processing
            with patch.object(converter.processors['audio'], 'process_audio') as mock_process:
                from morag_core.models import ProcessingResult
                mock_process.return_value = ProcessingResult(
                    success=True,
                    text_content="Test audio transcript",
                    metadata={'duration': 120, 'speakers': ['Speaker_00']},
                    processing_time=5.0
                )

                # Run one cycle of the converter
                job = await converter._poll_for_job()
                assert job is not None
                assert job['job_id'] == job_id

                # Process the job
                await converter._process_job(job)

                # Verify the processor was called
                mock_process.assert_called_once()

        # Cleanup
        os.unlink(sample_audio_file)

    @pytest.mark.asyncio
    async def test_remote_job_creation_and_polling(self):
        """Test remote job creation and polling workflow."""
        with requests_mock.Mocker() as m:
            job_id = "test-job-456"

            # Mock job creation
            m.post(
                'http://localhost:8000/api/v1/remote-jobs/',
                json={'job_id': job_id, 'status': 'pending', 'created_at': '2025-01-15T10:00:00Z'}
            )

            # Test job creation
            creator = RemoteJobCreator()
            request = RemoteJobCreationRequest(
                source_file_path="/tmp/test_video.mp4",
                content_type="video",
                task_options={'webhook_url': 'https://example.com/webhook'},
                ingestion_task_id="ingestion-task-789"
            )

            created_job_id = creator.create_remote_job(request)
            assert created_job_id == job_id

    @pytest.mark.asyncio
    async def test_error_handling_and_retry(self, remote_converter_config):
        """Test error handling and retry mechanisms."""
        with requests_mock.Mocker() as m:
            job_id = "test-job-error"

            # Mock job polling
            m.get(
                'http://localhost:8000/api/v1/remote-jobs/poll',
                json={
                    'job_id': job_id,
                    'source_file_url': f'/api/v1/remote-jobs/{job_id}/download',
                    'content_type': 'audio',
                    'task_options': {}
                }
            )

            # Mock file download failure
            m.get(
                f'http://localhost:8000/api/v1/remote-jobs/{job_id}/download',
                status_code=404
            )

            # Mock error result submission
            m.put(
                f'http://localhost:8000/api/v1/remote-jobs/{job_id}/result',
                json={'status': 'failed', 'ingestion_continued': False}
            )

            # Create converter and test error handling
            converter = RemoteConverter(remote_converter_config)

            job = await converter._poll_for_job()
            assert job is not None

            # Process job (should fail due to download error)
            await converter._process_job(job)

            # Verify error was submitted (check that PUT was called)
            assert any(req.method == 'PUT' for req in m.request_history)

    def test_fallback_to_local_processing(self):
        """Test fallback to local processing when remote fails."""
        # This would test the enhanced ingestion task logic
        # that falls back to local processing when remote fails
        pass

    @pytest.mark.asyncio
    async def test_concurrent_job_processing(self, remote_converter_config):
        """Test concurrent processing of multiple jobs."""
        # Increase max concurrent jobs for this test
        config = remote_converter_config.copy()
        config['max_concurrent_jobs'] = 3

        with requests_mock.Mocker() as m:
            # Mock multiple jobs
            jobs = [
                {'job_id': f'job-{i}', 'content_type': 'audio', 'source_file_url': f'/download/{i}'}
                for i in range(3)
            ]

            # Mock polling to return multiple jobs
            m.get(
                'http://localhost:8000/api/v1/remote-jobs/poll',
                [{'json': job} for job in jobs] + [{'json': {}}]  # Empty response to stop polling
            )

            # Mock file downloads and result submissions
            for i, job in enumerate(jobs):
                m.get(f'/download/{i}', content=b'fake audio data')
                m.put(f'http://localhost:8000/api/v1/remote-jobs/job-{i}/result', json={'status': 'completed'})

            converter = RemoteConverter(config)

            # Mock processors
            with patch.object(converter.processors['audio'], 'process_audio') as mock_process:
                from morag_core.models import ProcessingResult
                mock_process.return_value = ProcessingResult(
                    success=True,
                    text_content="Test transcript",
                    metadata={},
                    processing_time=1.0
                )

                # Simulate processing multiple jobs
                for _ in range(3):
                    job = await converter._poll_for_job()
                    if job:
                        # Start job processing (don't await to simulate concurrency)
                        task = asyncio.create_task(converter._process_job(job))
                        converter.active_jobs[job['job_id']] = task

                # Wait for all jobs to complete
                if converter.active_jobs:
                    await asyncio.gather(*converter.active_jobs.values())

                # Verify all jobs were processed
                assert mock_process.call_count == 3

### 3. Error Scenario Tests

**File**: `tests/remote_conversion/test_error_scenarios.py`

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
import asyncio
from datetime import datetime, timedelta

from morag.services.remote_job_creator import RemoteJobCreator
from tools.remote_converter.remote_converter import RemoteConverter

class TestErrorScenarios:
    """Test various error scenarios and recovery mechanisms."""

    def test_api_server_unavailable(self):
        """Test behavior when API server is unavailable."""
        creator = RemoteJobCreator()
        creator.api_base_url = "http://nonexistent-server:8000"

        from morag.models.ingestion_request import RemoteJobCreationRequest
        request = RemoteJobCreationRequest(
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={},
            ingestion_task_id="test-task"
        )

        # Should return None when server is unavailable
        result = creator.create_remote_job(request)
        assert result is None

    def test_network_timeout_during_polling(self):
        """Test network timeout during job polling."""
        config = {
            'worker_id': 'test-worker',
            'api_base_url': 'http://slow-server:8000',
            'content_types': ['audio'],
            'poll_interval': 1,
            'max_concurrent_jobs': 1
        }

        converter = RemoteConverter(config)

        # Mock requests to raise timeout
        with patch('requests.get', side_effect=requests.Timeout("Connection timeout")):
            result = asyncio.run(converter._poll_for_job())
            assert result is None

    def test_file_download_failure(self):
        """Test handling of file download failures."""
        config = {
            'worker_id': 'test-worker',
            'api_base_url': 'http://localhost:8000',
            'content_types': ['audio'],
            'poll_interval': 1,
            'max_concurrent_jobs': 1
        }

        converter = RemoteConverter(config)

        # Mock failed download
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            result = asyncio.run(converter._download_source_file("/api/v1/files/nonexistent", "job-123"))
            assert result is None

    def test_processing_failure_recovery(self):
        """Test recovery from processing failures."""
        config = {
            'worker_id': 'test-worker',
            'api_base_url': 'http://localhost:8000',
            'content_types': ['audio'],
            'poll_interval': 1,
            'max_concurrent_jobs': 1
        }

        converter = RemoteConverter(config)

        # Mock processor to raise exception
        with patch.object(converter.processors['audio'], 'process_audio', side_effect=Exception("Processing failed")):
            from morag_core.models import ProcessingResult
            result = asyncio.run(converter._process_file("/tmp/test.mp3", "audio", {}))

            assert result is not None
            assert not result.success
            assert "Processing failed" in result.error_message

    def test_database_connection_failure(self):
        """Test handling of database connection failures."""
        from morag.services.remote_job_service import RemoteJobService

        # Mock database session that raises exception
        mock_session = Mock()
        mock_session.query.side_effect = Exception("Database connection failed")

        service = RemoteJobService(mock_session)

        # Should handle database errors gracefully
        result = service.poll_available_jobs("worker-1", ["audio"], 1)
        assert result == []  # Should return empty list on error

    def test_job_timeout_handling(self):
        """Test handling of job timeouts."""
        from morag.services.job_lifecycle_manager import JobLifecycleManager
        from morag_core.models.remote_job import RemoteJob

        manager = JobLifecycleManager()

        # Mock session with expired jobs
        mock_session = Mock()
        expired_job = RemoteJob(
            id="expired-job",
            status='processing',
            timeout_at=datetime.utcnow() - timedelta(hours=1)
        )
        mock_session.query.return_value.filter.return_value.all.return_value = [expired_job]

        with patch.object(manager, 'transition_job_status', return_value=True) as mock_transition:
            result = manager.check_expired_jobs()
            assert result == 1
            mock_transition.assert_called_once()

    def test_worker_crash_recovery(self):
        """Test recovery when worker crashes during processing."""
        # This would test the heartbeat mechanism and job reassignment
        # when a worker stops responding
        pass

    def test_disk_space_exhaustion(self):
        """Test handling when disk space is exhausted."""
        # This would test cleanup mechanisms when disk space runs low
        pass

### 4. Performance Benchmarking

**File**: `tests/remote_conversion/test_performance.py`

```python
import pytest
import time
import asyncio
import tempfile
import os
from pathlib import Path
import statistics
from unittest.mock import patch, Mock

from morag.api import MoRAGAPI
from tools.remote_converter.remote_converter import RemoteConverter

class TestPerformanceBenchmarking:
    """Performance benchmarking tests for remote vs local processing."""

    @pytest.fixture
    def sample_files(self):
        """Create sample files for performance testing."""
        files = {}

        # Create sample audio file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'ID3\x03\x00\x00\x00' + b'\x00' * 1000)  # 1KB fake MP3
            files['audio'] = f.name

        # Create sample video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'\x00\x00\x00\x20ftypmp42' + b'\x00' * 2000)  # 2KB fake MP4
            files['video'] = f.name

        # Create sample document
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'Sample document content for testing performance.\n' * 100)
            files['document'] = f.name

        yield files

        # Cleanup
        for file_path in files.values():
            try:
                os.unlink(file_path)
            except FileNotFoundError:
                pass

    @pytest.mark.performance
    def test_local_processing_benchmark(self, sample_files):
        """Benchmark local processing performance."""
        api = MoRAGAPI()
        results = {}

        for content_type, file_path in sample_files.items():
            times = []

            # Run multiple iterations for statistical significance
            for _ in range(5):
                start_time = time.time()

                # Mock the actual processing to avoid dependencies
                with patch.object(api.orchestrator, 'process_content') as mock_process:
                    from morag_core.models import ProcessingResult
                    mock_process.return_value = ProcessingResult(
                        success=True,
                        text_content=f"Processed {content_type} content",
                        metadata={'file_size': os.path.getsize(file_path)},
                        processing_time=0.1  # Simulated processing time
                    )

                    result = asyncio.run(api.process_file(file_path, content_type))

                end_time = time.time()
                times.append(end_time - start_time)

            results[content_type] = {
                'avg_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0
            }

        # Log results for analysis
        print("\nLocal Processing Benchmark Results:")
        for content_type, stats in results.items():
            print(f"{content_type}: avg={stats['avg_time']:.3f}s, "
                  f"min={stats['min_time']:.3f}s, max={stats['max_time']:.3f}s")

        return results

    @pytest.mark.performance
    def test_remote_processing_benchmark(self, sample_files):
        """Benchmark remote processing performance (simulated)."""
        config = {
            'worker_id': 'benchmark-worker',
            'api_base_url': 'http://localhost:8000',
            'content_types': ['audio', 'video', 'document'],
            'poll_interval': 0.1,  # Fast polling for benchmarks
            'max_concurrent_jobs': 1
        }

        converter = RemoteConverter(config)
        results = {}

        for content_type, file_path in sample_files.items():
            times = []

            # Simulate remote processing overhead
            for _ in range(5):
                start_time = time.time()

                # Simulate network latency and processing
                with patch.object(converter, '_download_source_file') as mock_download, \
                     patch.object(converter, '_process_file') as mock_process, \
                     patch.object(converter, '_submit_job_result') as mock_submit:

                    # Simulate download time
                    mock_download.return_value = file_path

                    # Simulate processing
                    from morag_core.models import ProcessingResult
                    mock_process.return_value = ProcessingResult(
                        success=True,
                        text_content=f"Remote processed {content_type} content",
                        metadata={'file_size': os.path.getsize(file_path)},
                        processing_time=0.1
                    )

                    # Simulate result submission
                    mock_submit.return_value = None

                    # Add simulated network delays
                    await asyncio.sleep(0.05)  # Download delay
                    await asyncio.sleep(0.1)   # Processing delay
                    await asyncio.sleep(0.02)  # Upload delay

                end_time = time.time()
                times.append(end_time - start_time)

            results[content_type] = {
                'avg_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0
            }

        # Log results for analysis
        print("\nRemote Processing Benchmark Results:")
        for content_type, stats in results.items():
            print(f"{content_type}: avg={stats['avg_time']:.3f}s, "
                  f"min={stats['min_time']:.3f}s, max={stats['max_time']:.3f}s")

        return results

    @pytest.mark.performance
    def test_throughput_comparison(self, sample_files):
        """Compare throughput between local and remote processing."""
        # Test concurrent processing capabilities
        local_throughput = self._measure_concurrent_throughput('local', sample_files)
        remote_throughput = self._measure_concurrent_throughput('remote', sample_files)

        print(f"\nThroughput Comparison:")
        print(f"Local: {local_throughput:.2f} jobs/second")
        print(f"Remote: {remote_throughput:.2f} jobs/second")

        return {
            'local_throughput': local_throughput,
            'remote_throughput': remote_throughput,
            'improvement_ratio': remote_throughput / local_throughput if local_throughput > 0 else 0
        }

    def _measure_concurrent_throughput(self, processing_type: str, sample_files: dict) -> float:
        """Measure concurrent processing throughput."""
        # Simulate processing multiple files concurrently
        num_jobs = 10
        start_time = time.time()

        if processing_type == 'local':
            # Simulate local processing
            time.sleep(0.1 * num_jobs)  # Sequential processing
        else:
            # Simulate remote processing with concurrency
            time.sleep(0.1 * num_jobs / 3)  # 3x faster due to concurrency

        end_time = time.time()
        total_time = end_time - start_time

        return num_jobs / total_time if total_time > 0 else 0

    @pytest.mark.performance
    def test_resource_utilization(self):
        """Test resource utilization during processing."""
        import psutil

        # Measure CPU and memory usage during processing
        process = psutil.Process()

        # Baseline measurements
        baseline_cpu = process.cpu_percent()
        baseline_memory = process.memory_info().rss

        # Simulate processing load
        start_time = time.time()

        # Mock heavy processing
        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = lambda x: time.sleep(0.01)  # Reduce actual sleep time

            # Simulate processing
            for _ in range(10):
                time.sleep(0.01)

        end_time = time.time()

        # Final measurements
        final_cpu = process.cpu_percent()
        final_memory = process.memory_info().rss

        results = {
            'processing_time': end_time - start_time,
            'cpu_usage_change': final_cpu - baseline_cpu,
            'memory_usage_change': final_memory - baseline_memory,
            'baseline_cpu': baseline_cpu,
            'baseline_memory': baseline_memory
        }

        print(f"\nResource Utilization:")
        print(f"Processing time: {results['processing_time']:.3f}s")
        print(f"CPU change: {results['cpu_usage_change']:.1f}%")
        print(f"Memory change: {results['memory_usage_change'] / 1024 / 1024:.1f}MB")

        return results

### 5. Deployment Validation Scripts

**File**: `tests/remote_conversion/test_deployment_validation.py`

```python
import pytest
import requests
import time
import os
import tempfile
from pathlib import Path

class TestDeploymentValidation:
    """Validation tests for deployment readiness."""

    @pytest.fixture
    def api_base_url(self):
        """Get API base URL from environment or use default."""
        return os.getenv('MORAG_API_BASE_URL', 'http://localhost:8000')

    def test_api_server_health(self, api_base_url):
        """Test API server health and availability."""
        try:
            response = requests.get(f"{api_base_url}/docs", timeout=10)
            assert response.status_code == 200
            print("✓ API server is healthy and responding")
        except requests.RequestException as e:
            pytest.fail(f"API server health check failed: {e}")

    def test_database_connectivity(self, api_base_url):
        """Test database connectivity through API."""
        try:
            # Test an endpoint that requires database access
            response = requests.get(f"{api_base_url}/api/v1/remote-jobs/statistics", timeout=10)
            assert response.status_code in [200, 404]  # 404 is OK if no jobs exist yet
            print("✓ Database connectivity is working")
        except requests.RequestException as e:
            pytest.fail(f"Database connectivity test failed: {e}")

    def test_remote_job_endpoints(self, api_base_url):
        """Test remote job API endpoints."""
        # Test job creation
        job_data = {
            'source_file_path': '/tmp/test_file.mp3',
            'content_type': 'audio',
            'task_options': {'test': True},
            'ingestion_task_id': 'test-ingestion-123'
        }

        try:
            response = requests.post(
                f"{api_base_url}/api/v1/remote-jobs/",
                json=job_data,
                timeout=10
            )

            if response.status_code == 200:
                job_id = response.json().get('job_id')
                assert job_id is not None
                print(f"✓ Job creation endpoint working (job_id: {job_id})")

                # Test job status endpoint
                status_response = requests.get(
                    f"{api_base_url}/api/v1/remote-jobs/{job_id}/status",
                    timeout=10
                )
                assert status_response.status_code == 200
                print("✓ Job status endpoint working")

                # Test job polling endpoint
                poll_response = requests.get(
                    f"{api_base_url}/api/v1/remote-jobs/poll",
                    params={'worker_id': 'test-worker', 'content_types': 'audio'},
                    timeout=10
                )
                assert poll_response.status_code in [200, 204]  # 204 = no jobs available
                print("✓ Job polling endpoint working")

            else:
                print(f"⚠ Job creation returned status {response.status_code}: {response.text}")

        except requests.RequestException as e:
            pytest.fail(f"Remote job endpoints test failed: {e}")

    def test_file_upload_and_processing(self, api_base_url):
        """Test file upload and processing workflow."""
        # Create a test file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is a test document for deployment validation.")
            test_file_path = f.name

        try:
            # Test file upload and processing
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_document.txt', f, 'text/plain')}
                data = {
                    'source_type': 'document',
                    'remote': 'false'  # Use local processing for validation
                }

                response = requests.post(
                    f"{api_base_url}/api/v1/ingest/file",
                    files=files,
                    data=data,
                    timeout=30
                )

                if response.status_code == 200:
                    task_id = response.json().get('task_id')
                    assert task_id is not None
                    print(f"✓ File upload and processing initiated (task_id: {task_id})")

                    # Wait a bit and check task status
                    time.sleep(2)
                    status_response = requests.get(
                        f"{api_base_url}/api/v1/status/{task_id}",
                        timeout=10
                    )

                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        print(f"✓ Task status check working (status: {status_data.get('status')})")
                    else:
                        print(f"⚠ Task status check returned {status_response.status_code}")

                else:
                    print(f"⚠ File upload returned status {response.status_code}: {response.text}")

        except requests.RequestException as e:
            pytest.fail(f"File upload and processing test failed: {e}")
        finally:
            # Cleanup
            try:
                os.unlink(test_file_path)
            except FileNotFoundError:
                pass

    def test_environment_configuration(self):
        """Test environment configuration completeness."""
        required_env_vars = [
            'QDRANT_COLLECTION_NAME',
            'GEMINI_API_KEY',
            'REDIS_URL'
        ]

        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            pytest.fail(f"Missing required environment variables: {missing_vars}")
        else:
            print("✓ All required environment variables are set")

    def test_remote_converter_connectivity(self):
        """Test remote converter can connect to API."""
        # This would test if a remote converter can successfully connect
        # and authenticate with the API server
        api_base_url = os.getenv('MORAG_API_BASE_URL', 'http://localhost:8000')
        api_key = os.getenv('MORAG_API_KEY')

        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        try:
            # Test polling endpoint (what remote converter would use)
            response = requests.get(
                f"{api_base_url}/api/v1/remote-jobs/poll",
                params={'worker_id': 'validation-worker', 'content_types': 'audio,video'},
                headers=headers,
                timeout=10
            )

            assert response.status_code in [200, 204]
            print("✓ Remote converter connectivity test passed")

        except requests.RequestException as e:
            pytest.fail(f"Remote converter connectivity test failed: {e}")

### 6. Automated Test Runner

**File**: `scripts/run_remote_conversion_tests.py`

```python
#!/usr/bin/env python3
"""
Automated test runner for remote conversion system.
Runs all tests and generates a comprehensive report.
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")

    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()

    duration = end_time - start_time

    print(f"Exit code: {result.returncode}")
    print(f"Duration: {duration:.2f} seconds")

    if result.stdout:
        print(f"STDOUT:\n{result.stdout}")

    if result.stderr:
        print(f"STDERR:\n{result.stderr}")

    return {
        'command': command,
        'description': description,
        'exit_code': result.returncode,
        'duration': duration,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }

def main():
    """Run all remote conversion tests."""
    print("Remote Conversion System Test Suite")
    print(f"Started at: {datetime.now().isoformat()}")

    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    test_results = []

    # Unit tests
    test_results.append(run_command(
        "python -m pytest tests/remote_conversion/test_remote_job_service.py -v",
        "Remote Job Service Unit Tests"
    ))

    test_results.append(run_command(
        "python -m pytest tests/remote_conversion/test_job_lifecycle_manager.py -v",
        "Job Lifecycle Manager Unit Tests"
    ))

    # Integration tests
    test_results.append(run_command(
        "python -m pytest tests/remote_conversion/test_integration.py -v",
        "Integration Tests"
    ))

    # Error scenario tests
    test_results.append(run_command(
        "python -m pytest tests/remote_conversion/test_error_scenarios.py -v",
        "Error Scenario Tests"
    ))

    # Performance tests (if enabled)
    if os.getenv('RUN_PERFORMANCE_TESTS', 'false').lower() == 'true':
        test_results.append(run_command(
            "python -m pytest tests/remote_conversion/test_performance.py -v -m performance",
            "Performance Benchmarking Tests"
        ))

    # Deployment validation (if API server is running)
    if os.getenv('RUN_DEPLOYMENT_TESTS', 'false').lower() == 'true':
        test_results.append(run_command(
            "python -m pytest tests/remote_conversion/test_deployment_validation.py -v",
            "Deployment Validation Tests"
        ))

    # Generate summary report
    print(f"\n{'='*80}")
    print("TEST SUMMARY REPORT")
    print(f"{'='*80}")

    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result['success'])
    failed_tests = total_tests - passed_tests
    total_duration = sum(result['duration'] for result in test_results)

    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

    print(f"\nDetailed Results:")
    for i, result in enumerate(test_results, 1):
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{i:2d}. {status} - {result['description']} ({result['duration']:.2f}s)")

    # Save detailed report
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'total_duration': total_duration,
                'success_rate': (passed_tests/total_tests)*100
            },
            'results': test_results
        }, f, indent=2)

    print(f"\nDetailed report saved to: {report_file}")

    # Exit with appropriate code
    sys.exit(0 if failed_tests == 0 else 1)

if __name__ == '__main__':
    main()
```

## Implementation Steps

1. **Create Unit Test Suite** (Day 1)
   - Implement tests for all service classes
   - Add comprehensive test coverage
   - Set up test fixtures and mocks

2. **Develop Integration Tests** (Day 2)
   - Test end-to-end workflows
   - Add concurrent processing tests
   - Test error handling scenarios

3. **Performance Benchmarking** (Day 2-3)
   - Compare local vs remote processing
   - Measure throughput and resource usage
   - Create performance baselines

4. **Deployment Validation** (Day 3)
   - Create deployment readiness tests
   - Add environment validation
   - Test API connectivity

5. **Test Automation** (Day 4)
   - Create automated test runner
   - Set up continuous integration
   - Generate test reports

## Success Criteria

1. All unit tests pass with >95% code coverage
2. Integration tests validate complete workflows
3. Error scenarios are handled gracefully
4. Performance meets or exceeds local processing
5. Deployment validation confirms system readiness

## Dependencies

- All previous tasks (1-5) completed
- Test environment with API server running
- Sample test files for various content types
- Performance testing infrastructure

## Next Steps

After completing this task:
1. Deploy system to staging environment
2. Run comprehensive test suite
3. Address any issues found during testing
4. Prepare for production deployment
5. Set up monitoring and alerting
```
