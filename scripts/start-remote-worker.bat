@echo off
REM Remote Worker Startup Script for Windows

echo 🚀 Starting MoRAG Remote Worker
echo ================================

REM Configuration
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set CONFIG_FILE=%1
if "%CONFIG_FILE%"=="" set CONFIG_FILE=%PROJECT_ROOT%\configs\remote-worker.env

REM Check if config file exists
if not exist "%CONFIG_FILE%" (
    echo ❌ Configuration file not found: %CONFIG_FILE%
    echo Please copy configs\gpu-worker.env.example to configs\remote-worker.env and configure it
    exit /b 1
)

REM Load configuration (simplified - user should set environment variables)
echo 📋 Please ensure environment variables are set from: %CONFIG_FILE%
echo See the .env file for required variables

REM Check required environment variables
if "%MORAG_API_KEY%"=="" (
    echo ❌ Required environment variable not set: MORAG_API_KEY
    exit /b 1
)
if "%USER_ID%"=="" (
    echo ❌ Required environment variable not set: USER_ID
    exit /b 1
)
if "%REDIS_URL%"=="" (
    echo ❌ Required environment variable not set: REDIS_URL
    exit /b 1
)
if "%MAIN_SERVER_URL%"=="" (
    echo ❌ Required environment variable not set: MAIN_SERVER_URL
    exit /b 1
)

REM Check GPU availability
echo 🔍 Checking GPU availability...
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
if %errorlevel% equ 0 (
    echo ✅ GPU detected
) else (
    echo ⚠️  nvidia-smi failed. GPU acceleration may not be available.
)

REM Set default values
if "%WORKER_CONCURRENCY%"=="" set WORKER_CONCURRENCY=2
if "%WORKER_NAME%"=="" set WORKER_NAME=remote-worker-%USER_ID%-%COMPUTERNAME%

REM Calculate user-specific queue name
set USER_QUEUE=gpu-tasks-%USER_ID%
set WORKER_QUEUES=%USER_QUEUE%

REM Create temp directory
if not "%TEMP_DIR%"=="" (
    mkdir "%TEMP_DIR%" 2>nul
    echo ✅ Temp directory ready: %TEMP_DIR%
)

REM Change to project directory
cd /d "%PROJECT_ROOT%"

REM Start the remote worker
echo 🎯 Starting Remote Celery Worker...
echo User ID: %USER_ID%
echo Worker Name: %WORKER_NAME%
echo Queue: %WORKER_QUEUES%
echo Concurrency: %WORKER_CONCURRENCY%
echo Redis URL: %REDIS_URL%
echo Server URL: %MAIN_SERVER_URL%
echo.
echo ⚠️  IMPORTANT: This worker will ONLY process tasks for user: %USER_ID%
echo ⚠️  External services (Qdrant, Gemini) are handled by the main server
echo.

celery -A morag.worker worker ^
    --hostname="%WORKER_NAME%@%%h" ^
    --queues="%WORKER_QUEUES%" ^
    --concurrency="%WORKER_CONCURRENCY%" ^
    --loglevel=info ^
    --time-limit=7800 ^
    --soft-time-limit=7200
