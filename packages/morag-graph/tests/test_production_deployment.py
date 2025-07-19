"""Tests for production deployment readiness."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from morag_graph.graphiti.production.config import ProductionConfigManager
from morag_graph.graphiti.production.monitoring import GraphitiMonitoringService, HealthCheckResult
from morag_graph.graphiti.production.cleanup import LegacyCleanupManager


class TestProductionConfig:
    """Test production configuration."""
    
    @patch.dict('os.environ', {
        'MORAG_ENVIRONMENT': 'production',
        'MORAG_MAX_CONCURRENT_REQUESTS': '200',
        'GRAPHITI_NEO4J_PASSWORD': 'test_password',
        'OPENAI_API_KEY': 'test_api_key'
    })
    def test_production_config_loading(self):
        """Test production configuration loading."""
        config_manager = ProductionConfigManager()
        
        assert config_manager.config.environment == 'production'
        assert config_manager.config.max_concurrent_requests == 200
        assert config_manager.graphiti_config.neo4j_password == 'test_password'
    
    @patch.dict('os.environ', {
        'GRAPHITI_NEO4J_PASSWORD': 'test_password',
        'OPENAI_API_KEY': 'test_api_key'
    })
    def test_configuration_validation_success(self):
        """Test successful configuration validation."""
        config_manager = ProductionConfigManager()
        validation = config_manager.validate_configuration()
        
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
    
    def test_configuration_validation_failure(self):
        """Test configuration validation with missing required vars."""
        config_manager = ProductionConfigManager()
        validation = config_manager.validate_configuration()
        
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0


class TestGraphitiMonitoring:
    """Test monitoring functionality."""
    
    @pytest.fixture
    def mock_graphiti_service(self):
        """Create mock Graphiti service."""
        service = Mock()
        service.get_graphiti_status = Mock(return_value={"available": True})
        service.search_graphiti = AsyncMock(return_value={"success": True, "backend": "graphiti"})
        return service
    
    @pytest.fixture
    def monitoring_service(self, mock_graphiti_service):
        """Create monitoring service."""
        from morag_graph.graphiti.production.config import ProductionConfig
        config = ProductionConfig()
        return GraphitiMonitoringService(mock_graphiti_service, config)
    
    @pytest.mark.asyncio
    async def test_graphiti_health_check(self, monitoring_service):
        """Test Graphiti health check."""
        health_result = await monitoring_service._check_graphiti_health()
        
        assert isinstance(health_result, HealthCheckResult)
        assert health_result.service == "graphiti"
        assert health_result.status == "healthy"
        assert health_result.response_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_search_health_check(self, monitoring_service):
        """Test search health check."""
        health_result = await monitoring_service._check_search_health()
        
        assert isinstance(health_result, HealthCheckResult)
        assert health_result.service == "search"
        assert health_result.status == "healthy"
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_checks(self, monitoring_service):
        """Test comprehensive health checks."""
        health_results = await monitoring_service.perform_health_checks()
        
        assert len(health_results) >= 4  # At least 4 services checked
        assert all(isinstance(result, HealthCheckResult) for result in health_results)
        
        services_checked = {result.service for result in health_results}
        expected_services = {"graphiti", "neo4j", "openai", "search", "ingestion"}
        assert services_checked.intersection(expected_services) == expected_services
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, monitoring_service):
        """Test system metrics collection."""
        metrics = await monitoring_service.collect_system_metrics()
        
        assert metrics.cpu_usage_percent >= 0
        assert metrics.memory_usage_mb >= 0
        assert metrics.disk_usage_gb >= 0
    
    def test_alert_condition_checking(self, monitoring_service):
        """Test alert condition checking."""
        from morag_graph.graphiti.production.monitoring import SystemMetrics
        
        # Create metrics that should trigger alerts
        high_cpu_metrics = SystemMetrics(
            cpu_usage_percent=95.0,  # Above critical threshold
            memory_usage_mb=1024.0,
            disk_usage_gb=10.0,
            active_connections=50,
            requests_per_minute=100.0,
            average_response_time_ms=500.0,
            error_rate_percent=2.0
        )
        
        alerts = monitoring_service.check_alert_conditions(high_cpu_metrics)
        
        assert len(alerts) > 0
        cpu_alerts = [alert for alert in alerts if alert["metric"] == "cpu_usage"]
        assert len(cpu_alerts) == 1
        assert cpu_alerts[0]["severity"] == "critical"


class TestLegacyCleanup:
    """Test legacy cleanup functionality."""
    
    @pytest.fixture
    def cleanup_manager(self):
        """Create cleanup manager."""
        return LegacyCleanupManager()
    
    @pytest.mark.asyncio
    async def test_cleanup_plan_creation(self, cleanup_manager):
        """Test cleanup plan creation."""
        plan = await cleanup_manager.create_cleanup_plan()
        
        assert "phases" in plan
        assert len(plan["phases"]) == 3
        assert plan["estimated_duration_hours"] > 0
        assert "prerequisites" in plan
        assert "risks" in plan
        
        # Check phase structure
        for phase in plan["phases"]:
            assert "phase" in phase
            assert "name" in phase
            assert "steps" in phase
            assert "estimated_hours" in phase
    
    @pytest.mark.asyncio
    async def test_migration_validation(self, cleanup_manager):
        """Test migration completeness validation."""
        validation = await cleanup_manager.validate_migration_completeness()
        
        assert "complete" in validation
        assert "checks" in validation
        assert "recommendations" in validation
        assert isinstance(validation["complete"], bool)
    
    @pytest.mark.asyncio
    async def test_cleanup_phase_execution_dry_run(self, cleanup_manager):
        """Test cleanup phase execution in dry run mode."""
        result = await cleanup_manager.execute_cleanup_phase(1, dry_run=True)
        
        assert result["phase"] == 1
        assert result["dry_run"] is True
        assert "completed_steps" in result
        assert "failed_steps" in result
        assert isinstance(result["success"], bool)


class TestProductionDeployment:
    """Integration tests for production deployment."""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_cycle(self):
        """Test complete monitoring cycle."""
        # This would test with actual services
        # For now, we'll test the workflow with mocks
        
        mock_graphiti_service = Mock()
        mock_graphiti_service.get_graphiti_status = Mock(return_value={"available": True})
        mock_graphiti_service.search_graphiti = AsyncMock(return_value={"success": True})
        
        from morag_graph.graphiti.production.config import ProductionConfig
        config = ProductionConfig()
        
        monitoring_service = GraphitiMonitoringService(mock_graphiti_service, config)
        
        # Test health checks
        health_results = await monitoring_service.perform_health_checks()
        assert len(health_results) > 0
        
        # Test metrics collection
        metrics = await monitoring_service.collect_system_metrics()
        assert metrics is not None
        
        # Test alert checking
        alerts = monitoring_service.check_alert_conditions(metrics)
        assert isinstance(alerts, list)
        
        # Test system status
        status = monitoring_service.get_system_status()
        assert "overall_status" in status
    
    @pytest.mark.asyncio
    async def test_production_configuration_validation(self):
        """Test production configuration validation."""
        config_manager = ProductionConfigManager()
        validation = config_manager.validate_configuration()
        
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
        assert isinstance(validation["valid"], bool)
    
    @pytest.mark.asyncio
    async def test_cleanup_workflow(self):
        """Test cleanup workflow."""
        cleanup_manager = LegacyCleanupManager()
        
        # Test plan creation
        plan = await cleanup_manager.create_cleanup_plan()
        assert len(plan["phases"]) == 3
        
        # Test validation
        validation = await cleanup_manager.validate_migration_completeness()
        assert "complete" in validation
        
        # Test phase execution (dry run)
        for phase_num in [1, 2, 3]:
            result = await cleanup_manager.execute_cleanup_phase(phase_num, dry_run=True)
            assert result["dry_run"] is True
            assert result["phase"] == phase_num
