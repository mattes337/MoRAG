#!/usr/bin/env python3
"""Installation script for MoRAG modular system."""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoRAGInstaller:
    """Handles installation of MoRAG modular system."""

    def __init__(self, install_dir: Path, install_type: str = "full"):
        self.install_dir = Path(install_dir)
        self.install_type = install_type
        self.packages = {
            "core": "packages/morag-core",
            "services": "packages/morag-services",
            "web": "packages/morag-web",
            "youtube": "packages/morag-youtube",
            "main": "packages/morag",
        }

    def check_system_requirements(self) -> bool:
        """Check if system meets requirements."""
        logger.info("Checking system requirements")

        # Check Python version
        if sys.version_info < (3, 11):
            logger.error("Python 3.11 or higher is required")
            return False

        # Check for required system tools
        required_tools = ["git", "curl"]
        for tool in required_tools:
            if not shutil.which(tool):
                logger.error(f"Required tool not found: {tool}")
                return False

        # Check for optional tools based on install type
        if self.install_type in ["full", "web"]:
            if not shutil.which("chromium") and not shutil.which("google-chrome"):
                logger.warning(
                    "Chromium/Chrome not found - web scraping may be limited"
                )

        if self.install_type in ["full", "youtube"]:
            if not shutil.which("ffmpeg"):
                logger.warning("FFmpeg not found - video processing may be limited")

        logger.info("System requirements check completed")
        return True

    def create_virtual_environment(self) -> None:
        """Create Python virtual environment."""
        logger.info("Creating virtual environment")

        venv_path = self.install_dir / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

        # Activate virtual environment
        if os.name == "nt":  # Windows
            self.python_exe = venv_path / "Scripts" / "python.exe"
            self.pip_exe = venv_path / "Scripts" / "pip.exe"
        else:  # Unix-like
            self.python_exe = venv_path / "bin" / "python"
            self.pip_exe = venv_path / "bin" / "pip"

        # Upgrade pip
        subprocess.run(
            [str(self.pip_exe), "install", "--upgrade", "pip", "setuptools", "wheel"],
            check=True,
        )

        logger.info("Virtual environment created")

    def install_packages(self) -> None:
        """Install MoRAG packages."""
        logger.info(f"Installing MoRAG packages for {self.install_type} installation")

        # Determine which packages to install
        if self.install_type == "core":
            packages_to_install = ["core", "services"]
        elif self.install_type == "web":
            packages_to_install = ["core", "services", "web"]
        elif self.install_type == "youtube":
            packages_to_install = ["core", "services", "youtube"]
        elif self.install_type == "full":
            packages_to_install = ["core", "services", "web", "youtube", "main"]
        else:
            raise ValueError(f"Unknown install type: {self.install_type}")

        # Install packages in dependency order
        for package_name in packages_to_install:
            package_path = self.install_dir / self.packages[package_name]
            logger.info(f"Installing {package_name} package")
            subprocess.run(
                [str(self.pip_exe), "install", "-e", str(package_path)], check=True
            )

    def setup_configuration(self) -> None:
        """Set up configuration files."""
        logger.info("Setting up configuration")

        config_dir = self.install_dir / "config"
        config_dir.mkdir(exist_ok=True)

        # Create default configuration
        default_config = {
            "gemini_api_key": "",
            "qdrant_host": "localhost",
            "qdrant_port": 6333,
            "qdrant_collection_name": "morag_vectors",
            "redis_url": "redis://localhost:6379/0",
            "max_workers": 4,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "log_level": "INFO",
        }

        config_path = config_dir / "morag_config.json"
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)

        # Create environment file
        env_content = """# MoRAG Environment Configuration
# Copy this file to .env and update with your values

# Required: Gemini API Key for embeddings and AI services
GEMINI_API_KEY=your-gemini-api-key-here

# Vector Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=morag_documents

# Optional: Qdrant Cloud Configuration
# QDRANT_API_KEY=your-qdrant-cloud-api-key
# QDRANT_HOST=your-cluster-url.qdrant.io

# Task Queue Configuration
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO

# Optional: Custom settings
# MAX_WORKERS=4
# CHUNK_SIZE=1000
# CHUNK_OVERLAP=200
"""

        with open(config_dir / "env.example", "w") as f:
            f.write(env_content)

        logger.info(f"Configuration files created in {config_dir}")

    def setup_docker(self) -> None:
        """Set up Docker configuration."""
        logger.info("Setting up Docker configuration")

        # Copy docker-compose.yml to install directory
        docker_compose_source = Path("docker-compose.yml")
        if docker_compose_source.exists():
            shutil.copy(docker_compose_source, self.install_dir / "docker-compose.yml")

        # Create .env file for Docker
        env_content = """GEMINI_API_KEY=your-gemini-api-key-here
QDRANT_HOST=qdrant
QDRANT_PORT=6333
REDIS_URL=redis://redis:6379/0
LOG_LEVEL=INFO
"""

        with open(self.install_dir / ".env", "w") as f:
            f.write(env_content)

        logger.info("Docker configuration created")

    def create_startup_scripts(self) -> None:
        """Create startup scripts."""
        logger.info("Creating startup scripts")

        scripts_dir = self.install_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # Create start script
        if os.name == "nt":  # Windows
            start_script = """@echo off
echo Starting MoRAG services...
call venv\\Scripts\\activate
start "MoRAG API" morag-server --host 0.0.0.0 --port 8000
start "MoRAG Worker" morag-worker --concurrency 2
echo MoRAG services started
pause
"""
            with open(scripts_dir / "start.bat", "w") as f:
                f.write(start_script)
        else:  # Unix-like
            start_script = """#!/bin/bash
echo "Starting MoRAG services..."
source venv/bin/activate

# Start API server in background
morag-server --host 0.0.0.0 --port 8000 &
API_PID=$!

# Start worker in background
morag-worker --concurrency 2 &
WORKER_PID=$!

echo "MoRAG API started (PID: $API_PID)"
echo "MoRAG Worker started (PID: $WORKER_PID)"
echo "API available at: http://localhost:8000"
echo "Press Ctrl+C to stop services"

# Wait for interrupt
trap "kill $API_PID $WORKER_PID; exit" INT
wait
"""
            script_path = scripts_dir / "start.sh"
            with open(script_path, "w") as f:
                f.write(start_script)
            script_path.chmod(0o755)

        logger.info("Startup scripts created")

    def create_readme(self) -> None:
        """Create installation-specific README."""
        logger.info("Creating README")

        readme_content = f"""# MoRAG Installation

This is a {self.install_type} installation of MoRAG.

## Quick Start

### 1. Configure API Keys

Edit `config/morag_config.json` and add your API keys:
- Gemini API key for AI services
- Qdrant API key (if using cloud)

### 2. Start Services

#### Option A: Using Python directly
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows

# Start API server
morag-server --host 0.0.0.0 --port 8000

# In another terminal, start worker
morag-worker --concurrency 2
```

#### Option B: Using startup script
```bash
./scripts/start.sh  # Linux/Mac
# or
scripts\\start.bat  # Windows
```

#### Option C: Using Docker
```bash
docker-compose up -d
```

### 3. Test Installation

```bash
# Check health
morag health

# Process a web page
morag process-url https://example.com

# Access API documentation
# Open http://localhost:8000/docs in your browser
```

## Configuration

- Main config: `config/morag_config.json`
- Environment variables: Copy `config/env.example` to `.env`
- Docker environment: `.env` file in root directory

## Installed Packages

"""

        if self.install_type == "full":
            readme_content += """- morag-core: Core interfaces and models
- morag-services: AI and storage services
- morag-web: Web content processing
- morag-youtube: YouTube video processing
- morag: Main integration package
"""
        else:
            readme_content += f"- Packages for {self.install_type} installation\n"

        readme_content += """
## Next Steps

1. Configure your API keys in the configuration files
2. Start the required services (Redis, Qdrant)
3. Run the MoRAG services
4. Check the documentation for API usage

## Support

- Documentation: https://morag.readthedocs.io
- Issues: https://github.com/yourusername/morag/issues
"""

        with open(self.install_dir / "README.md", "w") as f:
            f.write(readme_content)

        logger.info("README created")

    def run_installation(self) -> None:
        """Run the complete installation process."""
        logger.info(f"Starting MoRAG {self.install_type} installation")

        try:
            if not self.check_system_requirements():
                raise RuntimeError("System requirements not met")

            self.install_dir.mkdir(parents=True, exist_ok=True)

            self.create_virtual_environment()
            self.install_packages()
            self.setup_configuration()
            self.setup_docker()
            self.create_startup_scripts()
            self.create_readme()

            logger.info("Installation completed successfully!")
            logger.info(f"MoRAG installed in: {self.install_dir}")
            logger.info("Please review the README.md file for next steps")

        except Exception as e:
            logger.error(f"Installation failed: {str(e)}")
            raise


def main():
    """Main entry point for installation script."""
    parser = argparse.ArgumentParser(description="Install MoRAG modular system")
    parser.add_argument(
        "--install-dir", "-d", default="./morag_install", help="Installation directory"
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["core", "web", "youtube", "full"],
        default="full",
        help="Installation type",
    )
    parser.add_argument(
        "--no-venv", action="store_true", help="Skip virtual environment creation"
    )

    args = parser.parse_args()

    install_dir = Path(args.install_dir).resolve()

    print(f"Installing MoRAG ({args.type}) to: {install_dir}")

    if install_dir.exists() and any(install_dir.iterdir()):
        confirm = (
            input("Directory exists and is not empty. Continue? (yes/no): ")
            .lower()
            .strip()
        )
        if confirm not in ["yes", "y"]:
            print("Installation cancelled")
            return

    installer = MoRAGInstaller(install_dir, args.type)
    installer.run_installation()


if __name__ == "__main__":
    import shutil

    main()
