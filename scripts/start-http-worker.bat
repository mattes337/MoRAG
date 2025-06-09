@echo off
REM Start HTTP Remote Worker - No Redis Required
REM This script starts a remote worker that connects directly to MoRAG server via HTTP

setlocal enabledelayedexpansion

REM Default values
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "ENV_FILE=%PROJECT_ROOT%\.env"
set "WORKER_TYPE=gpu"
set "POLL_INTERVAL=5"
set "MAX_CONCURRENT=1"

REM Function to show usage
if "%1"=="--help" goto :show_usage
if "%1"=="-h" goto :show_usage
if "%1"=="/?" goto :show_usage

REM Parse command line arguments
:parse_args
if "%1"=="" goto :end_parse
if "%1"=="-s" (
    set "SERVER_URL=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--server-url" (
    set "SERVER_URL=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="-k" (
    set "API_KEY=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--api-key" (
    set "API_KEY=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="-t" (
    set "WORKER_TYPE=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--worker-type" (
    set "WORKER_TYPE=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="-i" (
    set "POLL_INTERVAL=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--poll-interval" (
    set "POLL_INTERVAL=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="-c" (
    set "MAX_CONCURRENT=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--max-concurrent" (
    set "MAX_CONCURRENT=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="-e" (
    set "ENV_FILE=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--env-file" (
    set "ENV_FILE=%2"
    shift
    shift
    goto :parse_args
)
echo ❌ Unknown option: %1
goto :show_usage

:end_parse

REM Load environment file if it exists
if exist "%ENV_FILE%" (
    echo ℹ️  Loading environment from: %ENV_FILE%
    for /f "usebackq tokens=1,2 delims==" %%a in ("%ENV_FILE%") do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" (
            set "%%a=%%b"
        )
    )
)

REM Use environment variables if command line options not provided
if not defined SERVER_URL set "SERVER_URL=%MORAG_SERVER_URL%"
if not defined API_KEY set "API_KEY=%MORAG_API_KEY%"
if not defined WORKER_TYPE set "WORKER_TYPE=%WORKER_TYPE%"
if not defined POLL_INTERVAL set "POLL_INTERVAL=%POLL_INTERVAL%"
if not defined MAX_CONCURRENT set "MAX_CONCURRENT=%MAX_CONCURRENT_TASKS%"

REM Validate required parameters
if not defined SERVER_URL (
    echo ❌ Server URL is required. Use --server-url or set MORAG_SERVER_URL
    goto :show_usage
)

if not defined API_KEY (
    echo ❌ API key is required. Use --api-key or set MORAG_API_KEY
    goto :show_usage
)

REM Validate worker type
if not "%WORKER_TYPE%"=="gpu" if not "%WORKER_TYPE%"=="cpu" (
    echo ❌ Invalid worker type: %WORKER_TYPE%. Must be 'gpu' or 'cpu'
    exit /b 1
)

REM Check if Python script exists
set "PYTHON_SCRIPT=%SCRIPT_DIR%start_http_remote_worker.py"
if not exist "%PYTHON_SCRIPT%" (
    echo ❌ Python script not found: %PYTHON_SCRIPT%
    exit /b 1
)

REM Check Python
echo ℹ️  Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    exit /b 1
)

REM Generate worker ID
for /f "tokens=2 delims==" %%a in ('wmic computersystem get name /value ^| find "Name"') do set "HOSTNAME=%%a"
for /f "tokens=1-3 delims=:." %%a in ("%time%") do set "TIMESTAMP=%%a%%b%%c"
set "WORKER_ID=http-worker-%HOSTNAME%-%TIMESTAMP%"

REM Display configuration
echo ℹ️  HTTP Remote Worker Configuration:
echo   Server URL: %SERVER_URL%
echo   Worker Type: %WORKER_TYPE%
echo   Worker ID: %WORKER_ID%
echo   Poll Interval: %POLL_INTERVAL%s
echo   Max Concurrent: %MAX_CONCURRENT%
echo   API Key: %API_KEY:~0,8%...
echo.

REM Test server connectivity
echo ℹ️  Testing server connectivity...
curl -s --connect-timeout 10 "%SERVER_URL%/health" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Cannot reach server at %SERVER_URL%
    echo ⚠️  Worker will continue trying to connect...
) else (
    echo ✅ Server is reachable
)

REM Change to project directory
cd /d "%PROJECT_ROOT%"

REM Start the worker
echo ✅ Starting HTTP Remote Worker...
echo ℹ️  Press Ctrl+C to stop
echo.

python "%PYTHON_SCRIPT%" ^
    --server-url "%SERVER_URL%" ^
    --api-key "%API_KEY%" ^
    --worker-id "%WORKER_ID%" ^
    --worker-type "%WORKER_TYPE%" ^
    --poll-interval "%POLL_INTERVAL%" ^
    --max-concurrent "%MAX_CONCURRENT%"

goto :eof

:show_usage
echo Usage: %0 [OPTIONS]
echo.
echo Start HTTP Remote Worker for MoRAG (No Redis Required)
echo.
echo OPTIONS:
echo     -s, --server-url URL     Main server URL (required)
echo     -k, --api-key KEY        API key for authentication (required)
echo     -t, --worker-type TYPE   Worker type: gpu, cpu (default: gpu)
echo     -i, --poll-interval SEC  Polling interval in seconds (default: 5)
echo     -c, --max-concurrent N   Max concurrent tasks (default: 1)
echo     -e, --env-file FILE      Environment file path (default: .env)
echo     -h, --help               Show this help message
echo.
echo EXAMPLES:
echo     REM Start GPU worker
echo     %0 --server-url http://main-server:8000 --api-key your-key
echo.
echo     REM Start CPU worker with custom settings
echo     %0 -s http://main-server:8000 -k your-key -t cpu -i 10 -c 2
echo.
echo     REM Use environment file
echo     %0 --env-file configs\http-worker.env
echo.
echo ENVIRONMENT VARIABLES:
echo     MORAG_SERVER_URL         Main server URL
echo     MORAG_API_KEY           API key for authentication
echo     WORKER_TYPE             Worker type (gpu/cpu)
echo     POLL_INTERVAL           Polling interval in seconds
echo     MAX_CONCURRENT_TASKS    Maximum concurrent tasks
echo.
exit /b 0
