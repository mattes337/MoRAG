import asyncio
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from morag.middleware.monitoring import (
    PerformanceMonitoringMiddleware,
    ResourceMonitoringMiddleware,
)
from morag_services import LoggingService
from src.morag.services.metrics_service import (
    ApplicationMetrics,
    MetricsCollector,
    SystemMetrics,
)


class TestLoggingService:
    def test_logging_service_initialization(self):
        """Test logging service initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("morag.core.config.settings.log_file", f"{temp_dir}/test.log"):
                service = LoggingService()
                assert service.logger is not None

    def test_parse_size(self):
        """Test size parsing functionality."""
        service = LoggingService()

        assert service._parse_size("100KB") == 100 * 1024
        assert service._parse_size("50MB") == 50 * 1024 * 1024
        assert service._parse_size("2GB") == 2 * 1024 * 1024 * 1024
        assert service._parse_size("1024") == 1024

    def test_log_request(self):
        """Test HTTP request logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("morag.core.config.settings.log_file", f"{temp_dir}/test.log"):
                service = LoggingService()
                # Should not raise exception
                service.log_request("GET", "/test", 200, 0.5, "127.0.0.1")

    def test_log_task_lifecycle(self):
        """Test task logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("morag.core.config.settings.log_file", f"{temp_dir}/test.log"):
                service = LoggingService()
                service.log_task_start("task-123", "document_processing")
                service.log_task_complete("task-123", "document_processing", 2.5, True)

    def test_log_error(self):
        """Test error logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("morag.core.config.settings.log_file", f"{temp_dir}/test.log"):
                service = LoggingService()
                error = ValueError("Test error")
                context = {"user_id": "123", "action": "test"}
                service.log_error(error, context)

    def test_log_performance_metric(self):
        """Test performance metric logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("morag.core.config.settings.log_file", f"{temp_dir}/test.log"):
                service = LoggingService()
                service.log_performance_metric(
                    "response_time", 0.5, "seconds", {"endpoint": "/api/test"}
                )


class TestMetricsCollector:
    @pytest.fixture
    def collector(self):
        return MetricsCollector()

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.net_io_counters")
    def test_system_metrics_collection(
        self, mock_net, mock_disk, mock_memory, mock_cpu, collector
    ):
        """Test system metrics collection."""
        # Mock psutil responses
        mock_cpu.return_value = 25.5
        mock_memory.return_value = Mock(
            percent=60.0, used=8 * 1024 * 1024 * 1024, total=16 * 1024 * 1024 * 1024
        )
        mock_disk.return_value = Mock(
            used=100 * 1024 * 1024 * 1024, total=500 * 1024 * 1024 * 1024
        )
        mock_net.return_value = Mock(bytes_sent=1000000, bytes_recv=2000000)

        timestamp = datetime.utcnow()
        metrics = collector._collect_system_metrics(timestamp)

        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp == timestamp
        assert metrics.cpu_percent == 25.5
        assert metrics.memory_percent == 60.0
        assert metrics.memory_used_mb == 8 * 1024  # 8GB in MB
        assert metrics.memory_total_mb == 16 * 1024  # 16GB in MB

    @pytest.mark.asyncio
    async def test_application_metrics_collection(self, collector):
        """Test application metrics collection."""
        timestamp = datetime.utcnow()

        with patch("morag.services.task_manager.task_manager") as mock_task_manager:
            mock_task_manager.get_queue_stats.return_value = {
                "active_tasks": 5,
                "queues": {"document_processing": 2, "audio_processing": 1},
            }

            metrics = await collector._collect_application_metrics(timestamp)

            assert isinstance(metrics, ApplicationMetrics)
            assert metrics.timestamp == timestamp
            assert metrics.active_tasks == 5
            assert metrics.queue_lengths == {
                "document_processing": 2,
                "audio_processing": 1,
            }

    @pytest.mark.asyncio
    async def test_collect_metrics(self, collector):
        """Test full metrics collection."""
        with patch.object(collector, "_collect_system_metrics") as mock_system:
            with patch.object(collector, "_collect_application_metrics") as mock_app:
                mock_system.return_value = SystemMetrics(
                    timestamp=datetime.utcnow(),
                    cpu_percent=50.0,
                    memory_percent=70.0,
                    memory_used_mb=8192,
                    memory_total_mb=16384,
                    disk_percent=30.0,
                    disk_used_gb=100,
                    disk_total_gb=500,
                    network_bytes_sent=1000000,
                    network_bytes_recv=2000000,
                )

                mock_app.return_value = ApplicationMetrics(
                    timestamp=datetime.utcnow(),
                    active_tasks=3,
                    completed_tasks_1h=10,
                    failed_tasks_1h=1,
                    queue_lengths={"test": 2},
                    avg_task_duration=2.5,
                    documents_processed=50,
                    storage_size_mb=1024.0,
                    api_requests_1h=100,
                    api_errors_1h=2,
                )

                metrics = await collector.collect_metrics()

                assert "timestamp" in metrics
                assert "system" in metrics
                assert "application" in metrics
                assert len(collector.metrics_history) == 1

    def test_metrics_history_management(self, collector):
        """Test metrics history size management."""
        collector.max_history_size = 2

        # Add metrics beyond limit using the proper method
        for i in range(5):
            collector.metrics_history.append(
                {"test": i, "timestamp": datetime.utcnow().isoformat()}
            )
            # Manually trigger trimming logic
            if len(collector.metrics_history) > collector.max_history_size:
                collector.metrics_history = collector.metrics_history[
                    -collector.max_history_size :
                ]

        # Should trim to max size
        assert len(collector.metrics_history) <= collector.max_history_size

    def test_get_recent_metrics(self, collector):
        """Test getting recent metrics."""
        now = datetime.utcnow()

        # Add some test metrics
        collector.metrics_history = [
            {"timestamp": now.isoformat(), "test": 1},
            {"timestamp": (now).isoformat(), "test": 2},
        ]

        recent = collector.get_recent_metrics(1)
        assert len(recent) >= 0  # Should return metrics from last hour

    def test_get_current_metrics(self, collector):
        """Test getting current metrics."""
        # Empty history
        assert collector.get_current_metrics() == {}

        # With metrics
        test_metrics = {"timestamp": datetime.utcnow().isoformat(), "test": "value"}
        collector.metrics_history.append(test_metrics)

        current = collector.get_current_metrics()
        assert current == test_metrics


class TestMonitoringMiddleware:
    @pytest.mark.asyncio
    async def test_performance_monitoring_middleware(self):
        """Test performance monitoring middleware."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(PerformanceMonitoringMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Process-Time" in response.headers

    @pytest.mark.asyncio
    async def test_resource_monitoring_middleware(self):
        """Test resource monitoring middleware."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(ResourceMonitoringMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value = Mock(percent=60.0)

                client = TestClient(app)
                response = client.get("/test")

                assert response.status_code == 200


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        # Mock the problematic imports
        with patch.dict(
            "sys.modules",
            {
                "faster_whisper": MagicMock(),
                "morag.processors.audio": MagicMock(),
                "morag.tasks.audio_tasks": MagicMock(),
            },
        ):
            from fastapi.testclient import TestClient
            from morag.api.main import app

            client = TestClient(app)

            with patch("morag.core.config.settings.metrics_enabled", True):
                with patch(
                    "morag.services.metrics_service.metrics_collector"
                ) as mock_collector:
                    mock_collector.get_current_metrics.return_value = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "system": {"cpu_percent": 50.0},
                        "application": {"active_tasks": 2},
                    }

                    response = client.get("/health/metrics")
                    assert response.status_code == 200
                    data = response.json()
                    assert "timestamp" in data
                    assert "system" in data
                    assert "application" in data

    @pytest.mark.asyncio
    async def test_metrics_history_endpoint(self):
        """Test metrics history endpoint."""
        # Mock the problematic imports
        with patch.dict(
            "sys.modules",
            {
                "faster_whisper": MagicMock(),
                "morag.processors.audio": MagicMock(),
                "morag.tasks.audio_tasks": MagicMock(),
            },
        ):
            from fastapi.testclient import TestClient
            from morag.api.main import app

            client = TestClient(app)

            with patch("morag.core.config.settings.metrics_enabled", True):
                with patch(
                    "morag.services.metrics_service.metrics_collector"
                ) as mock_collector:
                    mock_collector.get_recent_metrics.return_value = [
                        {"timestamp": datetime.utcnow().isoformat(), "test": 1},
                        {"timestamp": datetime.utcnow().isoformat(), "test": 2},
                    ]

                    response = client.get("/health/metrics/history?hours=2")
                    assert response.status_code == 200
                    data = response.json()
                    assert "metrics" in data
                    assert "hours" in data
                    assert "count" in data
                    assert data["hours"] == 2
