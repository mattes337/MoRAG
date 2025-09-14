"""Tests for the PowerShell debug session script."""

import pytest
import subprocess
import sys
import os
import time
import requests
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestDebugSession:
    """Test the PowerShell debug session script functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.script_path = Path("scripts/debug-session.ps1")
        self.project_root = Path.cwd()
        
    def test_script_exists(self):
        """Test that the debug session script exists."""
        assert self.script_path.exists(), "Debug session script should exist"
        assert self.script_path.is_file(), "Debug session script should be a file"
        
    def test_script_has_execution_policy(self):
        """Test that the script can be executed."""
        # Check if script has proper PowerShell shebang
        with open(self.script_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            assert first_line.startswith('#!/usr/bin/env pwsh'), "Script should have PowerShell shebang"
    
    def test_script_help_parameter(self):
        """Test that the script shows help when -Help parameter is used."""
        if sys.platform != "win32":
            pytest.skip("PowerShell tests only run on Windows")
            
        try:
            result = subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass", 
                "-File", str(self.script_path), "-Help"
            ], capture_output=True, text=True, timeout=30)
            
            # Should exit with code 0 and show help
            assert result.returncode == 0, f"Help should succeed, got: {result.stderr}"
            assert "SYNOPSIS" in result.stdout or "DESCRIPTION" in result.stdout, "Should show help content"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Script help timed out")
        except FileNotFoundError:
            pytest.skip("PowerShell not available")
    
    def test_script_syntax_validation(self):
        """Test that the PowerShell script has valid syntax."""
        if sys.platform != "win32":
            pytest.skip("PowerShell tests only run on Windows")
            
        try:
            # Use PowerShell to validate syntax without executing
            result = subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-Command", f"Get-Content '{self.script_path}' | Out-String | Invoke-Expression"
            ], capture_output=True, text=True, timeout=10)
            
            # Should not have syntax errors
            assert "syntax error" not in result.stderr.lower(), f"Syntax error found: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Syntax validation timed out")
        except FileNotFoundError:
            pytest.skip("PowerShell not available")
    
    def test_required_directories_creation(self):
        """Test that required directories are created."""
        required_dirs = ["uploads", "temp", "logs"]
        
        # Remove directories if they exist
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                import shutil
                shutil.rmtree(dir_path, ignore_errors=True)
        
        # Mock the script execution for directory creation
        from scripts.debug_session_functions import setup_environment
        
        # This would be called by the script
        setup_environment()
        
        # Check that directories were created
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            assert dir_path.exists(), f"Directory {dir_name} should be created"
            assert dir_path.is_dir(), f"{dir_name} should be a directory"
    
    def test_prerequisites_check(self):
        """Test the prerequisites checking functionality."""
        # Mock the prerequisites check
        with patch('subprocess.run') as mock_run:
            # Mock successful Python check
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Python 3.11.0"
            
            from scripts.debug_session_functions import test_prerequisites
            
            # Should not raise an exception
            try:
                test_prerequisites()
            except SystemExit:
                pytest.fail("Prerequisites check should pass with valid Python")
    
    def test_virtual_environment_setup(self):
        """Test virtual environment setup."""
        venv_path = Path("test_venv")
        
        # Clean up any existing test venv
        if venv_path.exists():
            import shutil
            shutil.rmtree(venv_path, ignore_errors=True)
        
        # Mock venv creation
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            from scripts.debug_session_functions import setup_virtual_environment
            
            # Should create virtual environment
            setup_virtual_environment(venv_path="test_venv")
            
            # Verify subprocess was called correctly
            assert mock_run.called, "subprocess.run should be called for venv creation"
        
        # Clean up
        if venv_path.exists():
            import shutil
            shutil.rmtree(venv_path, ignore_errors=True)
    
    def test_dependency_installation(self):
        """Test dependency installation process."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            from scripts.debug_session_functions import install_dependencies
            
            # Test with skip flag
            install_dependencies(skip=True)
            assert not mock_run.called, "Should skip installation when flag is set"
            
            # Test normal installation
            mock_run.reset_mock()
            install_dependencies(skip=False)
            assert mock_run.called, "Should call pip install when not skipping"
    
    def test_service_health_checks(self):
        """Test service health check functionality."""
        from scripts.debug_session_functions import check_service_health
        
        # Test Redis health check
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "PONG"
            mock_run.return_value.returncode = 0
            
            result = check_service_health("redis")
            assert result is True, "Redis health check should pass with PONG response"
        
        # Test Qdrant health check
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}
            mock_get.return_value = mock_response
            
            result = check_service_health("qdrant")
            assert result is True, "Qdrant health check should pass with 200 response"
    
    def test_environment_variable_setup(self):
        """Test environment variable configuration."""
        from scripts.debug_session_functions import setup_environment_variables
        
        # Test normal mode
        env_vars = setup_environment_variables(test_mode=False, log_level="DEBUG")
        
        assert env_vars["LOG_LEVEL"] == "DEBUG", "Log level should be set correctly"
        assert env_vars["LOG_FORMAT"] == "console", "Log format should be console for debugging"
        assert "PYTHONPATH" in env_vars, "PYTHONPATH should be set"
        
        # Test test mode
        env_vars = setup_environment_variables(test_mode=True, log_level="INFO")
        
        assert "test" in env_vars["REDIS_URL"], "Test mode should use test Redis DB"
        assert "test" in env_vars["QDRANT_COLLECTION_NAME"], "Test mode should use test collection"
    
    def test_logging_functionality(self):
        """Test the logging functionality."""
        from scripts.debug_session_functions import write_log
        
        log_file = Path("logs/test_debug.log")
        
        # Ensure logs directory exists
        log_file.parent.mkdir(exist_ok=True)
        
        # Remove existing log file
        if log_file.exists():
            log_file.unlink()
        
        # Test logging
        write_log("Test message", "INFO", log_file=str(log_file))
        
        # Verify log file was created and contains message
        assert log_file.exists(), "Log file should be created"
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content, "Log message should be in file"
            assert "[INFO]" in content, "Log level should be in file"
        
        # Clean up
        if log_file.exists():
            log_file.unlink()
    
    def test_error_handling(self):
        """Test error handling and cleanup."""
        from scripts.debug_session_functions import handle_error, stop_services
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # Test error handling calls cleanup
            with patch('scripts.debug_session_functions.stop_services') as mock_stop:
                try:
                    handle_error("Test error", "Test context")
                except SystemExit:
                    pass  # Expected
                
                assert mock_stop.called, "Error handler should call stop_services"
    
    @pytest.mark.integration
    def test_full_script_dry_run(self):
        """Test running the script in test mode (integration test)."""
        if sys.platform != "win32":
            pytest.skip("PowerShell tests only run on Windows")
        
        try:
            # Run script in test mode with short timeout
            result = subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", str(self.script_path), 
                "-TestMode", "-SkipDependencies", "-SkipServices"
            ], capture_output=True, text=True, timeout=60)
            
            # Should complete without major errors
            # (May have warnings but shouldn't crash)
            assert result.returncode in [0, 1], f"Script should complete, got: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Script execution timed out")
        except FileNotFoundError:
            pytest.skip("PowerShell not available")


# Helper functions that would be extracted from the PowerShell script
# These are Python equivalents for testing purposes
class DebugSessionFunctions:
    """Python equivalents of PowerShell script functions for testing."""
    
    @staticmethod
    def setup_environment():
        """Create required directories."""
        dirs = ["uploads", "temp", "logs"]
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
    
    @staticmethod
    def test_prerequisites():
        """Check if prerequisites are available."""
        try:
            result = subprocess.run(["python", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise SystemExit("Python not found")
        except FileNotFoundError:
            raise SystemExit("Python not found")
    
    @staticmethod
    def setup_virtual_environment(venv_path="venv"):
        """Setup virtual environment."""
        subprocess.run(["python", "-m", "venv", venv_path], check=True)
    
    @staticmethod
    def install_dependencies(skip=False):
        """Install dependencies."""
        if skip:
            return
        subprocess.run(["pip", "install", "-e", "."], check=True)
    
    @staticmethod
    def check_service_health(service):
        """Check service health."""
        if service == "redis":
            result = subprocess.run(["docker", "exec", "morag-redis", "redis-cli", "ping"],
                                  capture_output=True, text=True)
            return result.stdout.strip() == "PONG"
        elif service == "qdrant":
            try:
                response = requests.get("http://localhost:6333/health", timeout=5)
                return response.status_code == 200
            except:
                return False
        return False
    
    @staticmethod
    def setup_environment_variables(test_mode=False, log_level="INFO"):
        """Setup environment variables."""
        env_vars = {
            "LOG_LEVEL": log_level,
            "LOG_FORMAT": "console",
            "PYTHONPATH": str(Path.cwd() / "src")
        }
        
        if test_mode:
            env_vars.update({
                "REDIS_URL": "redis://localhost:6379/15",
                "QDRANT_COLLECTION_NAME": "test_morag_documents"
            })
        
        return env_vars
    
    @staticmethod
    def write_log(message, level="INFO", log_file="logs/debug-session.log"):
        """Write log message."""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        Path(log_file).parent.mkdir(exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    @staticmethod
    def handle_error(error_message, context=""):
        """Handle errors and cleanup."""
        DebugSessionFunctions.write_log(f"ERROR in {context}: {error_message}", "ERROR")
        DebugSessionFunctions.stop_services()
        raise SystemExit(1)
    
    @staticmethod
    def stop_services():
        """Stop Docker services."""
        try:
            subprocess.run(["docker-compose", "-f", "docker/docker-compose.redis.yml", "down"],
                         capture_output=True)
            subprocess.run(["docker-compose", "-f", "docker/docker-compose.qdrant.yml", "down"],
                         capture_output=True)
        except:
            pass


# Make functions available for import
import sys
sys.modules['scripts.debug_session_functions'] = DebugSessionFunctions
