# MoRAG Essential Scripts

This directory contains essential scripts for development, debugging, deployment, and maintenance of the MoRAG project.

## Script Categories

- **Development & Debugging**: Scripts for setting up development environments
- **Database Management**: Scripts for initializing and managing the database
- **Worker Management**: Scripts for managing Celery workers
- **Production Deployment**: Scripts for production deployment and monitoring
- **System Maintenance**: Scripts for backup and monitoring

**Note**: Test scripts have been moved to `tests/manual/` and demo scripts to `examples/`.

## Debug Session Scripts

### `debug-session.ps1` (PowerShell)

The main debugging script that sets up a complete development environment.

**Features:**
- Automatic virtual environment setup
- Dependency installation
- Docker service management (Redis, Qdrant)
- Database initialization
- Test execution
- Celery worker startup
- FastAPI application launch with hot reload
- Comprehensive logging and error handling

**Usage:**
```powershell
# Basic usage
.\debug-session.ps1

# Skip dependency installation (if already installed)
.\debug-session.ps1 -SkipDependencies

# Skip Docker services (use external services)
.\debug-session.ps1 -SkipServices

# Run in test mode (mock services)
.\debug-session.ps1 -TestMode

# Set log level
.\debug-session.ps1 -LogLevel DEBUG

# Show help
.\debug-session.ps1 -Help
```

**Requirements:**
- PowerShell 5.1+ or PowerShell Core 6+
- Python 3.9+
- Docker (unless using -SkipServices or -TestMode)

### `debug-session.bat` (Windows Batch)

A Windows batch file wrapper for the PowerShell script that handles execution policy issues.

**Usage:**
```cmd
# Basic usage
debug-session.bat

# With options (same as PowerShell script)
debug-session.bat -TestMode -LogLevel DEBUG

# Show help
debug-session.bat -Help
```

## Other Scripts

### `init_db.py`

Initializes the database and creates necessary collections.

### `start_worker.py`

Starts a Celery worker for background task processing.

### Production Scripts

Essential scripts for production deployment and management:

#### `backup.sh`
Creates timestamped backups of all system data including:
- Qdrant vector database
- Redis data
- Uploaded files
- Configuration files

**Usage:**
```bash
./scripts/backup.sh
```

#### `deploy.sh`
Production deployment script that:
- Creates necessary directories
- Builds and starts Docker services
- Initializes the database
- Performs health checks

**Usage:**
```bash
./scripts/deploy.sh
```

#### `monitor.sh`
System monitoring script that displays:
- Service status
- Resource usage
- Queue statistics
- Health status
- Recent logs

**Usage:**
```bash
./scripts/monitor.sh
```

## Environment Setup

### Prerequisites

1. **Python 3.9+**
   ```bash
   python --version
   ```

2. **Docker** (for services)
   ```bash
   docker --version
   docker-compose --version
   ```

3. **Git** (for version control)
   ```bash
   git --version
   ```

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (defaults provided)
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379/0
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Manual Setup (Alternative)

If you prefer manual setup or the debug scripts don't work:

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   pip install -e ".[dev,audio,video,image,docling]"
   ```

3. **Start services:**
   ```bash
   docker-compose -f docker/docker-compose.redis.yml up -d
   docker-compose -f docker/docker-compose.qdrant.yml up -d
   ```

4. **Initialize database:**
   ```bash
   python scripts/init_db.py
   ```

5. **Start Celery worker:**
   ```bash
   celery worker -A morag.core.celery_app:app --loglevel=info
   ```

6. **Start application:**
   ```bash
   uvicorn morag.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Troubleshooting

### Common Issues

1. **PowerShell Execution Policy Error**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   Or use the batch file wrapper: `debug-session.bat`

2. **Python Not Found**
   - Ensure Python 3.9+ is installed
   - Add Python to your PATH
   - Try `python3` instead of `python`

3. **Docker Services Not Starting**
   - Ensure Docker is running
   - Check if ports 6379 (Redis) and 6333 (Qdrant) are available
   - Use `-SkipServices` flag to use external services

4. **Permission Errors**
   - Run as administrator (Windows)
   - Check file permissions
   - Ensure you have write access to the project directory

5. **Dependency Installation Fails**
   - Update pip: `python -m pip install --upgrade pip`
   - Clear pip cache: `pip cache purge`
   - Use `-SkipDependencies` if already installed

### Debug Logs

Logs are written to:
- `logs/debug-session.log` - Debug script logs
- `logs/morag.log` - Application logs

### Test Mode

Use `-TestMode` to run without external dependencies:
- Uses mock services instead of Redis/Qdrant
- Skips database initialization
- Suitable for development without Docker

### Getting Help

1. **Script Help:**
   ```powershell
   .\debug-session.ps1 -Help
   ```

2. **Application Help:**
   - Visit http://localhost:8000/docs for API documentation
   - Check logs in the `logs/` directory
   - Run tests: `pytest tests/`

3. **Community:**
   - Check the project README.md
   - Review the TASKS.md for implementation status
   - Look at existing test files for examples

## Development Workflow

1. **Start debug session:**
   ```bash
   ./scripts/debug-session.ps1
   ```

2. **Make changes to code**

3. **Test changes:**
   - Application auto-reloads (FastAPI with --reload)
   - Run specific tests: `pytest tests/test_specific.py`
   - Check API at http://localhost:8000/docs

4. **Debug issues:**
   - Check logs in `logs/` directory
   - Use debug endpoints: http://localhost:8000/health
   - Monitor Celery tasks

5. **Stop session:**
   - Press Ctrl+C to stop the application
   - Services are automatically cleaned up

## Performance Tips

1. **Skip dependencies** if already installed: `-SkipDependencies`
2. **Use test mode** for faster startup: `-TestMode`
3. **Adjust log level** for less verbose output: `-LogLevel WARNING`
4. **Use external services** if available: `-SkipServices`

## Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive configuration
- The debug scripts are for development only
- Production deployments should use proper configuration management
