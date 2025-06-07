@echo off
REM GPU Worker Startup Script for Windows

echo 🚀 Starting MoRAG GPU Worker
echo ================================

REM Configuration
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set CONFIG_FILE=%1
if "%CONFIG_FILE%"=="" set CONFIG_FILE=%PROJECT_ROOT%\configs\gpu-worker.env

REM Check if config file exists
if not exist "%CONFIG_FILE%" (
    echo ❌ Configuration file not found: %CONFIG_FILE%
    echo Please copy configs\gpu-worker.env and configure it for your environment
    exit /b 1
)

REM Load configuration (simplified - user should set environment variables)
echo 📋 Please ensure environment variables are set from: %CONFIG_FILE%
echo See the .env file for required variables

REM Check GPU availability
echo 🔍 Checking GPU availability...
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
if %errorlevel% equ 0 (
    echo ✅ GPU detected
) else (
    echo ⚠️  nvidia-smi failed. GPU acceleration may not be available.
)

REM Set default values
if "%WORKER_QUEUES%"=="" set WORKER_QUEUES=gpu-tasks
if "%WORKER_CONCURRENCY%"=="" set WORKER_CONCURRENCY=2
if "%WORKER_NAME%"=="" set WORKER_NAME=gpu-worker-%COMPUTERNAME%

REM Change to project directory
cd /d "%PROJECT_ROOT%"

REM Start the GPU worker
echo 🎯 Starting Celery worker...
echo Worker Name: %WORKER_NAME%
echo Queues: %WORKER_QUEUES%
echo Concurrency: %WORKER_CONCURRENCY%

celery -A morag.worker worker ^
    --hostname="%WORKER_NAME%@%%h" ^
    --queues="%WORKER_QUEUES%" ^
    --concurrency="%WORKER_CONCURRENCY%" ^
    --loglevel=info ^
    --time-limit=7800 ^
    --soft-time-limit=7200
