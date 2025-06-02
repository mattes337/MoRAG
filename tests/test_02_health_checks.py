import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from morag.api.main import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

class TestHealthChecks:
    """Test health check functionality."""

    def test_basic_health_check(self, client):
        """Test basic health endpoint returns correct structure."""
        response = client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["status", "version", "services"]
        for field in required_fields:
            assert field in data

        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"

    def test_readiness_check_structure(self, client):
        """Test readiness check returns correct structure."""
        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "version" in data

        # Check that all expected services are present
        expected_services = ["redis", "qdrant", "gemini"]
        for service in expected_services:
            assert service in data["services"]

    def test_health_check_response_format(self, client):
        """Test that health check response has correct format."""
        response = client.get("/health/")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        data = response.json()
        assert isinstance(data["services"], dict)
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)

    def test_readiness_check_response_format(self, client):
        """Test that readiness check response has correct format."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        data = response.json()
        assert isinstance(data["services"], dict)
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)

    def test_health_check_services_empty(self, client):
        """Test that basic health check has empty services dict."""
        response = client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        assert data["services"] == {}

    def test_readiness_check_service_status_values(self, client):
        """Test that readiness check service statuses are valid."""
        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        valid_statuses = ["healthy", "unhealthy", "not_connected"]

        for service, status in data["services"].items():
            assert status in valid_statuses

    def test_readiness_check_overall_status(self, client):
        """Test that overall status is calculated correctly."""
        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        valid_overall_statuses = ["healthy", "degraded"]
        assert data["status"] in valid_overall_statuses
