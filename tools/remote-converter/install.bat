@echo off
REM MoRAG Remote Converter Installation Script for Windows

echo 🚀 MoRAG Remote Converter Installation
echo ================================================

REM Check if we're in the right directory
if not exist "..\..\packages\morag-core\pyproject.toml" (
    echo ❌ Error: Please run this script from the tools\remote-converter directory
    echo Expected directory structure: MoRAG\tools\remote-converter\
    pause
    exit /b 1
)

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found. Please install Python 3.9+ and add it to PATH
    pause
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment
echo 📦 Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists, removing...
    rmdir /s /q venv
)

python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install MoRAG packages
echo 📦 Installing MoRAG packages...
pip install -e ..\..\packages\morag-core
pip install -e ..\..\packages\morag-audio
pip install -e ..\..\packages\morag-video
pip install -e ..\..\packages\morag-document
pip install -e ..\..\packages\morag-image
pip install -e ..\..\packages\morag-web
pip install -e ..\..\packages\morag-youtube

REM Install additional dependencies
echo 📦 Installing additional dependencies...
pip install requests pyyaml python-dotenv structlog

REM Create configuration file
echo ⚙️ Creating sample configuration...
python cli.py --create-config

REM Create startup script
echo 📝 Creating startup script...
echo @echo off > start.bat
echo REM MoRAG Remote Converter Startup Script >> start.bat
echo. >> start.bat
echo cd /d "%%~dp0" >> start.bat
echo. >> start.bat
echo REM Activate virtual environment >> start.bat
echo call venv\Scripts\activate.bat >> start.bat
echo. >> start.bat
echo REM Start the remote converter >> start.bat
echo python cli.py %%* >> start.bat

REM Create environment file template
echo 📝 Creating environment file template...
echo # MoRAG Remote Converter Environment Configuration > .env.example
echo. >> .env.example
echo # Worker Configuration >> .env.example
echo MORAG_WORKER_ID=gpu-worker-01 >> .env.example
echo MORAG_API_BASE_URL=http://localhost:8000 >> .env.example
echo MORAG_WORKER_CONTENT_TYPES=audio,video >> .env.example
echo MORAG_WORKER_POLL_INTERVAL=10 >> .env.example
echo MORAG_WORKER_MAX_CONCURRENT_JOBS=2 >> .env.example
echo. >> .env.example
echo # Optional API Authentication >> .env.example
echo # MORAG_API_KEY=your-api-key-here >> .env.example
echo. >> .env.example
echo # Logging and Temp Directory >> .env.example
echo MORAG_LOG_LEVEL=INFO >> .env.example
echo MORAG_TEMP_DIR=/tmp/morag_remote >> .env.example
echo. >> .env.example
echo # Audio Processing Configuration >> .env.example
echo # Override Whisper model size (tiny, base, small, medium, large-v2, large-v3) >> .env.example
echo WHISPER_MODEL_SIZE=large-v3 >> .env.example
echo # Alternative variable name (both are supported) >> .env.example
echo # MORAG_WHISPER_MODEL_SIZE=large-v3 >> .env.example
echo. >> .env.example
echo # Audio language (optional, auto-detect if not set) >> .env.example
echo # MORAG_AUDIO_LANGUAGE=en >> .env.example
echo. >> .env.example
echo # Audio device (auto, cpu, cuda) >> .env.example
echo # MORAG_AUDIO_DEVICE=auto >> .env.example
echo. >> .env.example
echo # Enable/disable features >> .env.example
echo # MORAG_ENABLE_SPEAKER_DIARIZATION=true >> .env.example
echo # MORAG_ENABLE_TOPIC_SEGMENTATION=true >> .env.example
echo. >> .env.example
echo # SpaCy model for topic segmentation (en_core_web_sm, de_core_news_sm, etc.) >> .env.example
echo # MORAG_SPACY_MODEL=de_core_news_sm >> .env.example
echo MORAG_TEMP_DIR=C:\temp\morag_remote >> .env.example

echo.
echo ✅ Installation complete!
echo.
echo 📋 Next steps:
echo 1. Configure the remote converter:
echo    copy remote_converter_config.yaml.example remote_converter_config.yaml
echo    REM Edit the configuration file with your settings
echo.
echo 2. (Optional) Set up environment variables:
echo    copy .env.example .env
echo    REM Edit the .env file with your settings
echo.
echo 3. Test the connection:
echo    start.bat --test-connection
echo.
echo 4. Start the remote converter:
echo    start.bat
echo.
echo 🔧 Windows Service installation (optional):
echo    Use NSSM or similar tool to create a Windows service
echo.
echo 🎉 Happy processing!
pause
