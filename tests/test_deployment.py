import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

class TestDockerConfiguration:
    """Test Docker configuration files."""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and is valid."""
        dockerfile_path = Path("Dockerfile")
        assert dockerfile_path.exists(), "Dockerfile not found"

        content = dockerfile_path.read_text()
        assert "FROM python:3.11-slim" in content
        assert "WORKDIR /app" in content
        assert "HEALTHCHECK" in content
        assert "CMD" in content

    def test_worker_dockerfile_exists(self):
        """Test that worker Dockerfile exists and is valid."""
        dockerfile_path = Path("Dockerfile.worker")
        assert dockerfile_path.exists(), "Dockerfile.worker not found"

        content = dockerfile_path.read_text()
        assert "FROM python:3.11-slim" in content
        assert "celery" in content
        assert "worker" in content

    def test_docker_compose_prod_exists(self):
        """Test that production docker-compose file exists and is valid."""
        compose_path = Path("docker-compose.prod.yml")
        assert compose_path.exists(), "docker-compose.prod.yml not found"

        with open(compose_path) as f:
            compose_config = yaml.safe_load(f)

        # Check required services
        services = compose_config.get('services', {})
        required_services = ['redis', 'qdrant', 'api', 'worker-documents', 'worker-media', 'worker-web', 'nginx', 'flower']

        for service in required_services:
            assert service in services, f"Service {service} not found in docker-compose"

        # Check volumes
        volumes = compose_config.get('volumes', {})
        assert 'redis_data' in volumes
        assert 'qdrant_data' in volumes

        # Check networks
        networks = compose_config.get('networks', {})
        assert 'morag-network' in networks

class TestNginxConfiguration:
    """Test Nginx configuration."""

    def test_nginx_config_exists(self):
        """Test that nginx configuration exists."""
        nginx_path = Path("nginx/nginx.conf")
        assert nginx_path.exists(), "nginx/nginx.conf not found"

        content = nginx_path.read_text()
        assert "upstream api" in content
        assert "limit_req_zone" in content
        assert "proxy_pass" in content
        assert "gzip on" in content

class TestEnvironmentConfiguration:
    """Test environment configuration."""

    def test_env_prod_example_exists(self):
        """Test that production environment example exists."""
        env_path = Path(".env.prod.example")
        assert env_path.exists(), ".env.prod.example not found"

        content = env_path.read_text()
        assert "GEMINI_API_KEY" in content
        assert "REDIS_URL" in content
        assert "QDRANT_HOST" in content
        assert "LOG_LEVEL" in content

    def test_env_variables_complete(self):
        """Test that all required environment variables are documented."""
        env_path = Path(".env.prod.example")
        content = env_path.read_text()

        required_vars = [
            "GEMINI_API_KEY",
            "REDIS_URL",
            "QDRANT_HOST",
            "QDRANT_PORT",
            "LOG_LEVEL",
            "METRICS_ENABLED",
            "ENVIRONMENT"
        ]

        for var in required_vars:
            assert var in content, f"Required environment variable {var} not found"

class TestDeploymentScripts:
    """Test deployment scripts."""

    def test_deploy_script_exists(self):
        """Test that deployment script exists and is executable."""
        script_path = Path("scripts/deploy.sh")
        assert script_path.exists(), "scripts/deploy.sh not found"

        content = script_path.read_text(encoding='utf-8')
        assert "#!/bin/bash" in content
        assert "docker-compose" in content
        assert ".env.prod" in content

    def test_backup_script_exists(self):
        """Test that backup script exists."""
        script_path = Path("scripts/backup.sh")
        assert script_path.exists(), "scripts/backup.sh not found"

        content = script_path.read_text(encoding='utf-8')
        assert "#!/bin/bash" in content
        assert "BACKUP_DIR" in content
        assert "tar" in content

    def test_monitor_script_exists(self):
        """Test that monitoring script exists."""
        script_path = Path("scripts/monitor.sh")
        assert script_path.exists(), "scripts/monitor.sh not found"

        content = script_path.read_text(encoding='utf-8')
        assert "#!/bin/bash" in content
        assert "docker stats" in content
        assert "curl" in content

    def test_init_db_script_exists(self):
        """Test that database initialization script exists."""
        script_path = Path("scripts/init_db.py")
        assert script_path.exists(), "scripts/init_db.py not found"

        content = script_path.read_text()
        assert "#!/usr/bin/env python3" in content
        assert "qdrant_service" in content
        assert "asyncio" in content

class TestKubernetesConfiguration:
    """Test Kubernetes configuration."""

    def test_k8s_deployment_exists(self):
        """Test that Kubernetes deployment file exists."""
        k8s_path = Path("k8s/deployment.yaml")
        assert k8s_path.exists(), "k8s/deployment.yaml not found"

        with open(k8s_path) as f:
            k8s_configs = list(yaml.safe_load_all(f))

        # Check for required Kubernetes resources
        resource_kinds = [config.get('kind') for config in k8s_configs if config]
        assert 'Deployment' in resource_kinds
        assert 'Service' in resource_kinds
        assert 'PersistentVolumeClaim' in resource_kinds
        assert 'Secret' in resource_kinds

class TestProductionReadiness:
    """Test production readiness aspects."""

    def test_health_check_configuration(self):
        """Test that health checks are properly configured."""
        compose_path = Path("docker-compose.prod.yml")
        with open(compose_path) as f:
            compose_config = yaml.safe_load(f)

        services = compose_config.get('services', {})

        # Check that critical services have health checks
        critical_services = ['redis', 'qdrant', 'api']
        for service in critical_services:
            service_config = services.get(service, {})
            assert 'healthcheck' in service_config, f"Service {service} missing health check"

    def test_security_configuration(self):
        """Test security configurations."""
        nginx_path = Path("nginx/nginx.conf")
        content = nginx_path.read_text()

        # Check security headers
        assert "X-Frame-Options" in content
        assert "X-Content-Type-Options" in content
        assert "X-XSS-Protection" in content

        # Check rate limiting
        assert "limit_req_zone" in content
        assert "limit_req" in content

    def test_resource_limits(self):
        """Test that resource limits are configured."""
        k8s_path = Path("k8s/deployment.yaml")
        with open(k8s_path) as f:
            k8s_configs = list(yaml.safe_load_all(f))

        for config in k8s_configs:
            if config and config.get('kind') == 'Deployment':
                containers = config['spec']['template']['spec']['containers']
                for container in containers:
                    resources = container.get('resources', {})
                    assert 'requests' in resources, "Resource requests not configured"
                    assert 'limits' in resources, "Resource limits not configured"

    def test_persistent_storage(self):
        """Test that persistent storage is configured."""
        k8s_path = Path("k8s/deployment.yaml")
        with open(k8s_path) as f:
            k8s_configs = list(yaml.safe_load_all(f))

        pvc_found = False
        for config in k8s_configs:
            if config and config.get('kind') == 'PersistentVolumeClaim':
                pvc_found = True
                break

        assert pvc_found, "No PersistentVolumeClaim found"

class TestDeploymentIntegration:
    """Integration tests for deployment (requires Docker)."""

    @pytest.mark.integration
    def test_docker_build(self):
        """Test that Docker images can be built."""
        try:
            import docker
            client = docker.from_env()

            # Test main Dockerfile
            image, logs = client.images.build(
                path=".",
                dockerfile="Dockerfile",
                tag="morag-test:latest",
                rm=True
            )
            assert image is not None

            # Test worker Dockerfile
            worker_image, worker_logs = client.images.build(
                path=".",
                dockerfile="Dockerfile.worker",
                tag="morag-worker-test:latest",
                rm=True
            )
            assert worker_image is not None

        except (ImportError, Exception) as e:
            pytest.skip(f"Docker not available: {e}")

    @pytest.mark.integration
    def test_compose_validation(self):
        """Test that docker-compose configuration is valid."""
        try:
            result = subprocess.run(
                ["docker-compose", "-f", "docker-compose.prod.yml", "config"],
                capture_output=True,
                text=True,
                check=True
            )
            assert result.returncode == 0

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            pytest.skip(f"Docker Compose not available: {e}")

class TestDocumentation:
    """Test deployment documentation."""

    def test_deployment_task_documentation(self):
        """Test that deployment task documentation exists."""
        task_path = Path("tasks/22-deployment-config.md")
        assert task_path.exists(), "Deployment task documentation not found"

        content = task_path.read_text(encoding='utf-8')
        assert "Docker" in content
        assert "production" in content
        assert "nginx" in content
        assert "Kubernetes" in content

    def test_readme_deployment_section(self):
        """Test that README contains deployment information."""
        readme_path = Path("README.md")
        if readme_path.exists():
            content = readme_path.read_text()
            # Check for deployment-related content
            deployment_keywords = ["docker", "deployment", "production"]
            has_deployment_info = any(keyword in content.lower() for keyword in deployment_keywords)
            assert has_deployment_info, "README missing deployment information"
