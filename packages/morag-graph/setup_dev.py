#!/usr/bin/env python3
"""Development setup script for morag-graph package.

This script helps set up the development environment for the morag-graph package,
including installing dependencies, setting up pre-commit hooks, and running initial tests.

Usage:
    python setup_dev.py [--api-key YOUR_GEMINI_API_KEY] [--skip-tests]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result.
    
    Args:
        cmd: Command to run as list of strings
        cwd: Working directory for the command
        check: Whether to raise exception on non-zero exit code
        
    Returns:
        CompletedProcess result
    """
    print(f"🔧 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if check:
            raise
        return e


def check_python_version() -> bool:
    """Check if Python version is compatible.
    
    Returns:
        True if Python version is compatible, False otherwise
    """
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ is required, but you have {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_pip() -> bool:
    """Check if pip is available and up to date.
    
    Returns:
        True if pip is available, False otherwise
    """
    print("📦 Checking pip...")
    
    try:
        result = run_command(["python", "-m", "pip", "--version"])
        print("✅ pip is available")
        
        # Try to upgrade pip
        print("🔄 Upgrading pip...")
        run_command(["python", "-m", "pip", "install", "--upgrade", "pip"], check=False)
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pip is not available")
        return False


def install_package_in_development_mode() -> bool:
    """Install the package in development mode.
    
    Returns:
        True if installation succeeded, False otherwise
    """
    print("📦 Installing morag-graph in development mode...")
    
    try:
        # Install in editable mode with development dependencies
        run_command(["python", "-m", "pip", "install", "-e", ".[dev]"])
        print("✅ Package installed in development mode")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install package in development mode")
        
        # Try installing dependencies manually
        print("🔄 Trying to install dependencies manually...")
        try:
            # Core dependencies
            core_deps = [
                "pydantic>=2.0.0",
                "structlog>=23.0.0",
                "python-dotenv>=1.0.0",
                "neo4j>=5.0.0",
                "google-generativeai>=0.3.0",
                "httpx>=0.24.0",
                "aiofiles>=23.0.0"
            ]
            
            # Development dependencies
            dev_deps = [
                "pytest>=7.0.0",
                "pytest-asyncio>=0.21.0",
                "black>=23.0.0",
                "isort>=5.12.0",
                "flake8>=6.0.0",
                "mypy>=1.0.0"
            ]
            
            all_deps = core_deps + dev_deps
            
            run_command(["python", "-m", "pip", "install"] + all_deps)
            print("✅ Dependencies installed manually")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False


def setup_environment_file(api_key: Optional[str] = None) -> bool:
    """Set up .env file for development.
    
    Args:
        api_key: Optional Gemini API key
        
    Returns:
        True if .env file was set up, False otherwise
    """
    print("🔧 Setting up environment file...")
    
    env_file = Path(".env")
    
    # Check if .env already exists
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    # Create .env file
    env_content = [
        "# Environment variables for morag-graph development",
        "",
        "# Gemini API Key (required for LLM-based extraction)",
        f"GEMINI_API_KEY={api_key or 'your_gemini_api_key_here'}",
        "",
        "# Neo4j Configuration (optional, for Neo4j storage backend)",
        "NEO4J_URI=bolt://localhost:7687",
        "NEO4J_USER=neo4j",
        "NEO4J_PASSWORD=password",
        "",
        "# Development settings",
        "LOG_LEVEL=DEBUG",
        "PYTHONPATH=src"
    ]
    
    try:
        with open(env_file, "w") as f:
            f.write("\n".join(env_content))
        
        print(f"✅ Created .env file: {env_file.absolute()}")
        if not api_key:
            print("⚠️  Please edit .env file and add your Gemini API key")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def run_code_quality_checks() -> bool:
    """Run code quality checks.
    
    Returns:
        True if all checks passed, False otherwise
    """
    print("🔍 Running code quality checks...")
    
    checks_passed = True
    
    # Check if tools are available
    tools = ["black", "isort", "flake8"]
    available_tools = []
    
    for tool in tools:
        try:
            run_command(["python", "-m", tool, "--version"])
            available_tools.append(tool)
        except subprocess.CalledProcessError:
            print(f"⚠️  {tool} is not available")
    
    if not available_tools:
        print("⚠️  No code quality tools available, skipping checks")
        return True
    
    # Run black (code formatting)
    if "black" in available_tools:
        try:
            print("🎨 Running black (code formatting)...")
            run_command(["python", "-m", "black", "--check", "src", "tests", "examples"])
            print("✅ Black formatting check passed")
        except subprocess.CalledProcessError:
            print("⚠️  Black formatting issues found (run 'black src tests examples' to fix)")
            checks_passed = False
    
    # Run isort (import sorting)
    if "isort" in available_tools:
        try:
            print("📚 Running isort (import sorting)...")
            run_command(["python", "-m", "isort", "--check-only", "src", "tests", "examples"])
            print("✅ isort check passed")
        except subprocess.CalledProcessError:
            print("⚠️  Import sorting issues found (run 'isort src tests examples' to fix)")
            checks_passed = False
    
    # Run flake8 (linting)
    if "flake8" in available_tools:
        try:
            print("🔍 Running flake8 (linting)...")
            run_command(["python", "-m", "flake8", "src", "tests", "examples"])
            print("✅ flake8 linting passed")
        except subprocess.CalledProcessError:
            print("⚠️  Linting issues found")
            checks_passed = False
    
    return checks_passed


def run_basic_tests(api_key: Optional[str] = None) -> bool:
    """Run basic tests to verify setup.
    
    Args:
        api_key: Optional OpenAI API key for tests
        
    Returns:
        True if tests passed, False otherwise
    """
    print("🧪 Running basic tests...")
    
    # Set API key if provided
    env = os.environ.copy()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    
    try:
        # Check if pytest is available
        run_command(["python", "-m", "pytest", "--version"])
        
        # Run import tests (quick validation)
        print("📦 Testing package imports...")
        import_test_code = """
import sys
sys.path.insert(0, 'src')

try:
    from morag_graph.models import Entity, Relation, Graph
    from morag_graph.extraction import EntityExtractor, RelationExtractor
    from morag_graph.storage import JsonStorage
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
"""
        
        result = subprocess.run(
            ["python", "-c", import_test_code],
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ Import test failed: {result.stderr}")
            return False
        
        # Run quick pytest if API key is available
        if api_key:
            print("🧪 Running quick pytest validation...")
            try:
                run_command(
                    ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-x", "--maxfail=1"],
                    check=False  # Don't fail setup if tests fail
                )
            except subprocess.CalledProcessError:
                print("⚠️  Some tests failed, but setup can continue")
        else:
            print("⚠️  Skipping pytest (no API key provided)")
        
        return True
        
    except subprocess.CalledProcessError:
        print("❌ pytest is not available")
        return False


def create_development_scripts() -> bool:
    """Create helpful development scripts.
    
    Returns:
        True if scripts were created successfully, False otherwise
    """
    print("📝 Creating development scripts...")
    
    scripts = {
        "format_code.py": """
#!/usr/bin/env python3
\"\"\"Format code using black and isort.\"\"\"
import subprocess
import sys

def main():
    print("🎨 Formatting code with black...")
    subprocess.run(["python", "-m", "black", "src", "tests", "examples"])
    
    print("📚 Sorting imports with isort...")
    subprocess.run(["python", "-m", "isort", "src", "tests", "examples"])
    
    print("✅ Code formatting complete")

if __name__ == "__main__":
    main()
""",
        "lint_code.py": """
#!/usr/bin/env python3
\"\"\"Run linting checks.\"\"\"
import subprocess
import sys

def main():
    print("🔍 Running flake8 linting...")
    result1 = subprocess.run(["python", "-m", "flake8", "src", "tests", "examples"])
    
    print("🔍 Running mypy type checking...")
    result2 = subprocess.run(["python", "-m", "mypy", "src"], check=False)
    
    if result1.returncode == 0 and result2.returncode == 0:
        print("✅ All linting checks passed")
        return 0
    else:
        print("❌ Some linting checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    }
    
    try:
        for script_name, script_content in scripts.items():
            script_path = Path(script_name)
            if not script_path.exists():
                with open(script_path, "w") as f:
                    f.write(script_content)
                print(f"✅ Created {script_name}")
            else:
                print(f"✅ {script_name} already exists")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create development scripts: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Set up development environment for morag-graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will:
1. Check Python version compatibility
2. Install the package in development mode
3. Set up environment file (.env)
4. Run code quality checks
5. Run basic tests (if API key provided)
6. Create helpful development scripts

Example:
  python setup_dev.py --api-key your-gemini-api-key
"""
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key for testing (can also be set via GEMINI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests during setup"
    )
    
    parser.add_argument(
        "--skip-quality-checks",
        action="store_true",
        help="Skip code quality checks"
    )
    
    args = parser.parse_args()
    
    print("🚀 MoRAG Graph - Development Environment Setup")
    print("" + "="*50)
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    
    success = True
    
    # Step 1: Check Python version
    if not check_python_version():
        return 1
    
    # Step 2: Check pip
    if not check_pip():
        return 1
    
    # Step 3: Install package
    if not install_package_in_development_mode():
        success = False
    
    # Step 4: Setup environment
    if not setup_environment_file(api_key):
        success = False
    
    # Step 5: Create development scripts
    if not create_development_scripts():
        success = False
    
    # Step 6: Run code quality checks
    if not args.skip_quality_checks:
        if not run_code_quality_checks():
            print("⚠️  Code quality checks failed, but setup can continue")
    
    # Step 7: Run basic tests
    if not args.skip_tests:
        if not run_basic_tests(api_key):
            print("⚠️  Basic tests failed, but setup can continue")
    
    print("" + "="*50)
    
    if success:
        print("✅ Development environment setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Edit .env file and add your Gemini API key (if not already set)")
        print("   2. Run tests: python run_tests.py")
        print("   3. Run demo: python examples/extraction_demo.py --api-key YOUR_KEY")
        print("   4. Format code: python format_code.py")
        print("   5. Run linting: python lint_code.py")
        return 0
    else:
        print("⚠️  Development environment setup completed with some issues")
        print("   Please review the output above and fix any issues")
        return 1


if __name__ == "__main__":
    exit(main())