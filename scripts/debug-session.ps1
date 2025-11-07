#!/usr/bin/env pwsh
<#
.SYNOPSIS
    MoRAG Debug Session Startup Script

.DESCRIPTION
    This script sets up a complete debugging environment for the MoRAG project.
    It handles dependency installation, service startup, environment configuration,
    and launches the application in debug mode with comprehensive logging.

.PARAMETER SkipDependencies
    Skip dependency installation if already installed

.PARAMETER SkipServices
    Skip starting Docker services (Redis, Qdrant)

.PARAMETER TestMode
    Run in test mode with mock services

.PARAMETER LogLevel
    Set logging level (DEBUG, INFO, WARNING, ERROR)

.EXAMPLE
    .\debug-session.ps1

.EXAMPLE
    .\debug-session.ps1 -SkipDependencies -LogLevel DEBUG

.EXAMPLE
    .\debug-session.ps1 -TestMode
#>

param(
    [switch]$SkipDependencies,
    [switch]$SkipServices,
    [switch]$TestMode,
    [ValidateSet("DEBUG", "INFO", "WARNING", "ERROR")]
    [string]$LogLevel = "DEBUG",
    [switch]$Help
)

# Show help if requested
if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

# Script configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for output
$Colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Debug = "Magenta"
}

# Logging function
function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("SUCCESS", "WARNING", "ERROR", "INFO", "DEBUG")]
        [string]$Level = "INFO"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = $Colors[$Level]

    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $color

    # Also log to file
    $logFile = "logs/debug-session.log"
    if (!(Test-Path "logs")) {
        New-Item -ItemType Directory -Path "logs" -Force | Out-Null
    }
    "[$timestamp] [$Level] $Message" | Out-File -FilePath $logFile -Append -Encoding UTF8
}

# Error handling
function Handle-Error {
    param([string]$ErrorMessage, [string]$Context = "")

    Write-Log "ERROR in $Context`: $ErrorMessage" -Level "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" -Level "DEBUG"

    # Cleanup on error
    Write-Log "Performing cleanup..." -Level "INFO"
    Stop-Services

    exit 1
}

# Trap errors
trap {
    Handle-Error -ErrorMessage $_.Exception.Message -Context $_.InvocationInfo.ScriptName
}

# Check prerequisites
function Test-Prerequisites {
    Write-Log "Checking prerequisites..." -Level "INFO"

    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Python not found"
        }
        Write-Log "Found Python: $pythonVersion" -Level "SUCCESS"
    }
    catch {
        Write-Log "Python is required but not found. Please install Python 3.9+" -Level "ERROR"
        exit 1
    }

    # Check Docker
    if (!$SkipServices -and !$TestMode) {
        try {
            $dockerVersion = docker --version 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw "Docker not found"
            }
            Write-Log "Found Docker: $dockerVersion" -Level "SUCCESS"
        }
        catch {
            Write-Log "Docker is required for services. Use -SkipServices or -TestMode to bypass" -Level "ERROR"
            exit 1
        }
    }

    # Check if we're in the right directory
    if (!(Test-Path "pyproject.toml")) {
        Write-Log "Not in MoRAG project directory. Please run from project root." -Level "ERROR"
        exit 1
    }

    Write-Log "Prerequisites check completed" -Level "SUCCESS"
}

# Setup virtual environment
function Setup-VirtualEnvironment {
    Write-Log "Setting up Python virtual environment..." -Level "INFO"

    if (!(Test-Path "venv")) {
        Write-Log "Creating virtual environment..." -Level "INFO"
        python -m venv venv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment"
        }
    }

    # Activate virtual environment
    Write-Log "Activating virtual environment..." -Level "INFO"
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        & "venv\Scripts\Activate.ps1"
    } else {
        & "venv/bin/Activate.ps1"
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate virtual environment"
    }

    Write-Log "Virtual environment activated" -Level "SUCCESS"
}

# Install dependencies
function Install-Dependencies {
    if ($SkipDependencies) {
        Write-Log "Skipping dependency installation" -Level "INFO"
        return
    }

    Write-Log "Installing dependencies..." -Level "INFO"

    # Upgrade pip first
    Write-Log "Upgrading pip..." -Level "INFO"
    python -m pip install --upgrade pip

    # Install main dependencies
    Write-Log "Installing main dependencies..." -Level "INFO"
    pip install -e .

    # Install development dependencies
    Write-Log "Installing development dependencies..." -Level "INFO"
    pip install -e ".[dev]"

    # Install optional dependencies based on mode
    if (!$TestMode) {
        Write-Log "Installing optional dependencies..." -Level "INFO"
        pip install -e ".[audio,video,image,docling,morphik,milvus]"
    }

    Write-Log "Dependencies installed successfully" -Level "SUCCESS"
}

# Start services
function Start-Services {
    if ($SkipServices -or $TestMode) {
        Write-Log "Skipping service startup" -Level "INFO"
        return
    }

    Write-Log "Starting required services..." -Level "INFO"

    # Start Redis
    Write-Log "Starting Redis..." -Level "INFO"
    docker-compose -f docker/docker-compose.redis.yml up -d
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start Redis"
    }

    # Start Qdrant
    Write-Log "Starting Qdrant..." -Level "INFO"
    docker-compose -f docker/docker-compose.qdrant.yml up -d
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start Qdrant"
    }

    # Wait for services to be ready
    Write-Log "Waiting for services to be ready..." -Level "INFO"
    Start-Sleep -Seconds 10

    # Check Redis
    $redisCheck = docker exec morag-redis redis-cli ping 2>&1
    if ($redisCheck -ne "PONG") {
        Write-Log "Redis health check failed: $redisCheck" -Level "WARNING"
    } else {
        Write-Log "Redis is ready" -Level "SUCCESS"
    }

    # Check Qdrant
    try {
        $qdrantCheck = Invoke-RestMethod -Uri "http://localhost:6333/health" -Method GET -TimeoutSec 5
        Write-Log "Qdrant is ready" -Level "SUCCESS"
    }
    catch {
        Write-Log "Qdrant health check failed: $($_.Exception.Message)" -Level "WARNING"
    }
}

# Stop services
function Stop-Services {
    Write-Log "Stopping services..." -Level "INFO"

    try {
        docker-compose -f docker/docker-compose.redis.yml down 2>&1 | Out-Null
        docker-compose -f docker/docker-compose.qdrant.yml down 2>&1 | Out-Null
        Write-Log "Services stopped" -Level "SUCCESS"
    }
    catch {
        Write-Log "Error stopping services: $($_.Exception.Message)" -Level "WARNING"
    }
}

# Setup environment
function Setup-Environment {
    Write-Log "Setting up environment..." -Level "INFO"

    # Create required directories
    $dirs = @("uploads", "temp", "logs")
    foreach ($dir in $dirs) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Log "Created directory: $dir" -Level "DEBUG"
        }
    }

    # Set environment variables
    $env:LOG_LEVEL = $LogLevel
    $env:LOG_FORMAT = "console"  # Better for debugging
    $env:PYTHONPATH = "$PWD/src"

    if ($TestMode) {
        $env:REDIS_URL = "redis://localhost:6379/15"
        $env:QDRANT_COLLECTION_NAME = "test_morag_documents"
        Write-Log "Test mode environment configured" -Level "INFO"
    }

    Write-Log "Environment setup completed" -Level "SUCCESS"
}

# Initialize database
function Initialize-Database {
    if ($TestMode) {
        Write-Log "Skipping database initialization in test mode" -Level "INFO"
        return
    }

    Write-Log "Initializing database..." -Level "INFO"

    try {
        python scripts/init_db.py
        if ($LASTEXITCODE -ne 0) {
            throw "Database initialization failed"
        }
        Write-Log "Database initialized successfully" -Level "SUCCESS"
    }
    catch {
        Write-Log "Database initialization failed: $($_.Exception.Message)" -Level "WARNING"
        Write-Log "Continuing without database initialization..." -Level "INFO"
    }
}

# Run tests
function Run-Tests {
    Write-Log "Running tests to verify setup..." -Level "INFO"

    try {
        # Run basic configuration tests
        python -m pytest tests/test_01_config_validation.py -v
        if ($LASTEXITCODE -ne 0) {
            throw "Configuration tests failed"
        }

        # Run health check tests
        python -m pytest tests/test_02_health_checks.py -v
        if ($LASTEXITCODE -ne 0) {
            throw "Health check tests failed"
        }

        Write-Log "Basic tests passed" -Level "SUCCESS"
    }
    catch {
        Write-Log "Some tests failed: $($_.Exception.Message)" -Level "WARNING"
        Write-Log "Continuing with debug session..." -Level "INFO"
    }
}

# Start Celery worker
function Start-CeleryWorker {
    if ($TestMode) {
        Write-Log "Skipping Celery worker in test mode" -Level "INFO"
        return
    }

    Write-Log "Starting Celery worker..." -Level "INFO"

    # Start worker in background
    Start-Process -FilePath "python" -ArgumentList @(
        "-m", "celery",
        "worker",
        "-A", "morag.core.celery_app:app",
        "--loglevel=$($LogLevel.ToLower())",
        "--concurrency=2"
    ) -WindowStyle Hidden

    Write-Log "Celery worker started in background" -Level "SUCCESS"
}

# Start main application
function Start-Application {
    Write-Log "Starting MoRAG application in debug mode..." -Level "INFO"

    # Set debug environment variables
    $env:FASTAPI_ENV = "development"
    $env:DEBUG = "true"

    Write-Log "Application will be available at: http://localhost:8000" -Level "INFO"
    Write-Log "API documentation: http://localhost:8000/docs" -Level "INFO"
    Write-Log "Health check: http://localhost:8000/health" -Level "INFO"
    Write-Log "" -Level "INFO"
    Write-Log "Press Ctrl+C to stop the application" -Level "INFO"
    Write-Log "Logs will be written to: logs/morag.log" -Level "INFO"
    Write-Log "" -Level "INFO"

    try {
        # Start the application
        python -m uvicorn morag.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level $LogLevel.ToLower()
    }
    catch {
        Write-Log "Application stopped: $($_.Exception.Message)" -Level "INFO"
    }
    finally {
        Write-Log "Cleaning up..." -Level "INFO"
        Stop-Services
    }
}

# Show system information
function Show-SystemInfo {
    Write-Log "=== MoRAG Debug Session Information ===" -Level "INFO"
    Write-Log "Python Version: $(python --version)" -Level "INFO"
    Write-Log "Working Directory: $PWD" -Level "INFO"
    Write-Log "Log Level: $LogLevel" -Level "INFO"
    Write-Log "Test Mode: $TestMode" -Level "INFO"
    Write-Log "Skip Dependencies: $SkipDependencies" -Level "INFO"
    Write-Log "Skip Services: $SkipServices" -Level "INFO"
    Write-Log "=========================================" -Level "INFO"
}

# Main execution
function Main {
    try {
        Write-Log "Starting MoRAG Debug Session..." -Level "SUCCESS"

        Show-SystemInfo
        Test-Prerequisites
        Setup-VirtualEnvironment
        Install-Dependencies
        Setup-Environment
        Start-Services
        Initialize-Database
        Run-Tests
        Start-CeleryWorker
        Start-Application
    }
    catch {
        Handle-Error -ErrorMessage $_.Exception.Message -Context "Main"
    }
}

# Cleanup on script exit
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    Write-Log "Script exiting, cleaning up..." -Level "INFO"
    Stop-Services
}

# Run main function
Main
