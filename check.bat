@echo off
REM Windows batch script for running build checks

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="check" goto check
if "%1"=="syntax" goto syntax
if "%1"=="imports" goto imports
if "%1"=="build" goto build
if "%1"=="test" goto test
if "%1"=="clean" goto clean

:help
echo Available commands:
echo   check    - Run all checks (syntax, imports, build)
echo   syntax   - Check Python syntax
echo   imports  - Check for import issues
echo   build    - Run comprehensive build checks
echo   test     - Run tests
echo   clean    - Clean build artifacts
echo.
echo Usage: check.bat [command]
goto end

:check
echo Running all checks...
call :syntax
if errorlevel 1 goto error
call :imports
if errorlevel 1 goto error
call :build
if errorlevel 1 goto error
echo ✅ All checks passed!
goto end

:syntax
echo 🔍 Checking Python syntax...
python scripts/build_test.py packages --syntax-only
goto end

:imports
echo 📦 Checking imports...
python scripts/check_imports.py packages --errors-only
goto end

:build
echo 🚀 Running build checks...
python scripts/build_check.py --skip-tests
goto end

:test
echo 🧪 Running tests...
cd packages\morag-stages
python -m pytest tests\ -v
cd ..\..
goto end

:clean
echo 🧹 Cleaning build artifacts...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (build) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (dist) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
echo Cleanup completed.
goto end

:error
echo ❌ Checks failed!
exit /b 1

:end
