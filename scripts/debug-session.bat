@echo off
REM MoRAG Debug Session Startup Script (Windows Batch)
REM This is a wrapper for the PowerShell script to handle execution policy issues

setlocal enabledelayedexpansion

echo ========================================
echo MoRAG Debug Session Startup
echo ========================================
echo.

REM Check if PowerShell is available
powershell -Command "Write-Host 'PowerShell is available'" >nul 2>&1
if errorlevel 1 (
    echo ERROR: PowerShell is not available or not in PATH
    echo Please install PowerShell or use the manual setup instructions
    pause
    exit /b 1
)

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%debug-session.ps1"

REM Check if PowerShell script exists
if not exist "%PS_SCRIPT%" (
    echo ERROR: PowerShell script not found at: %PS_SCRIPT%
    echo Please ensure the debug-session.ps1 file exists in the scripts directory
    pause
    exit /b 1
)

echo Found PowerShell script: %PS_SCRIPT%
echo.

REM Parse command line arguments
set "PS_ARGS="
set "SHOW_HELP=false"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="-help" set "SHOW_HELP=true"
if /i "%~1"=="--help" set "SHOW_HELP=true"
if /i "%~1"=="/?" set "SHOW_HELP=true"
if /i "%~1"=="-h" set "SHOW_HELP=true"

REM Pass all arguments to PowerShell
set "PS_ARGS=%PS_ARGS% %~1"
shift
goto :parse_args

:args_done

REM Show help if requested
if "%SHOW_HELP%"=="true" (
    echo.
    echo Usage: debug-session.bat [OPTIONS]
    echo.
    echo Options:
    echo   -SkipDependencies    Skip dependency installation
    echo   -SkipServices        Skip starting Docker services
    echo   -TestMode           Run in test mode with mock services
    echo   -LogLevel LEVEL     Set logging level (DEBUG, INFO, WARNING, ERROR)
    echo   -Help               Show this help message
    echo.
    echo Examples:
    echo   debug-session.bat
    echo   debug-session.bat -SkipDependencies -LogLevel DEBUG
    echo   debug-session.bat -TestMode
    echo.
    pause
    exit /b 0
)

echo Starting MoRAG debug session...
echo Arguments: %PS_ARGS%
echo.

REM Try to run PowerShell script with bypass execution policy
echo Attempting to run PowerShell script...
powershell -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %PS_ARGS%
set "EXIT_CODE=%errorlevel%"

echo.
echo PowerShell script completed with exit code: %EXIT_CODE%

if %EXIT_CODE% neq 0 (
    echo.
    echo ========================================
    echo DEBUG SESSION FAILED
    echo ========================================
    echo.
    echo The debug session encountered an error.
    echo.
    echo Common solutions:
    echo 1. Ensure Python 3.9+ is installed and in PATH
    echo 2. Ensure Docker is installed and running
    echo 3. Check that you're in the MoRAG project root directory
    echo 4. Try running with -TestMode to skip external services
    echo 5. Check the logs in the logs/ directory for more details
    echo.
    echo For more help, run: debug-session.bat -Help
    echo.
) else (
    echo.
    echo ========================================
    echo DEBUG SESSION COMPLETED SUCCESSFULLY
    echo ========================================
    echo.
)

pause
exit /b %EXIT_CODE%
