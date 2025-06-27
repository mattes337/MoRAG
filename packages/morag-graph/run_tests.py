#!/usr/bin/env python3
"""Test runner script for morag-graph package.

This script provides an easy way to run the entity and relation extraction tests.
It handles environment setup and provides clear output about test results.

Usage:
    python run_tests.py --api-key YOUR_GEMINI_API_KEY
    
Or set GEMINI_API_KEY environment variable and run:
    python run_tests.py
    
Options:
    --entity-only    Run only entity extraction tests
    --relation-only  Run only relation extraction tests
    --quick          Run a subset of tests for quick validation
    --verbose        Show detailed test output
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_dependencies() -> bool:
    """Check if required dependencies are installed.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    # Map package names to their import names
    required_packages = {
        "pytest": "pytest",
        "pytest-asyncio": "pytest_asyncio",
        "google-generativeai": "google.generativeai",
        "httpx": "httpx",
        "pydantic": "pydantic",
        "python-dotenv": "dotenv",
        "aiofiles": "aiofiles"
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def setup_environment(api_key: Optional[str] = None) -> bool:
    """Setup environment for testing.
    
    Args:
        api_key: Optional Gemini API key
        
    Returns:
        True if environment is properly set up, False otherwise
    """
    # Check API key
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Gemini API key is required for testing.")
        print("   Set it via --api-key argument or GEMINI_API_KEY environment variable.")
        return False
    
    # Set environment variable for tests
    os.environ["GEMINI_API_KEY"] = api_key
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "src" / "morag_graph").exists():
        print("❌ Please run this script from the morag-graph package root directory.")
        return False
    
    return True


def run_pytest(test_files: List[str], verbose: bool = False, quick: bool = False) -> int:
    """Run pytest with the specified test files.
    
    Args:
        test_files: List of test file paths
        verbose: Whether to show verbose output
        quick: Whether to run quick tests only
        
    Returns:
        Exit code from pytest
    """
    cmd = ["python", "-m", "pytest"]
    
    # Add test files
    cmd.extend(test_files)
    
    # Add pytest options
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add asyncio support
    cmd.extend(["--asyncio-mode=auto"])
    
    # Add quick test markers if requested
    if quick:
        cmd.extend(["-m", "not slow"])
    
    # Show output in real-time
    cmd.extend(["-s"])
    
    print(f"🧪 Running tests: {' '.join(cmd)}")
    print("" + "="*60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Please install pytest:")
        print("   pip install pytest pytest-asyncio")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 1


def run_entity_tests(verbose: bool = False, quick: bool = False) -> int:
    """Run entity extraction tests.
    
    Args:
        verbose: Whether to show verbose output
        quick: Whether to run quick tests only
        
    Returns:
        Exit code from tests
    """
    test_files = ["tests/test_entity_extraction.py"]
    
    if quick:
        # Run only basic tests for quick validation
        test_files = ["tests/test_entity_extraction.py::test_entity_extraction_basic"]
    
    return run_pytest(test_files, verbose, quick)


def run_relation_tests(verbose: bool = False, quick: bool = False) -> int:
    """Run relation extraction tests.
    
    Args:
        verbose: Whether to show verbose output
        quick: Whether to run quick tests only
        
    Returns:
        Exit code from tests
    """
    test_files = ["tests/test_relation_extraction.py"]
    
    if quick:
        # Run only basic tests for quick validation
        test_files = ["tests/test_relation_extraction.py::test_relation_extraction_basic"]
    
    return run_pytest(test_files, verbose, quick)


def run_all_tests(verbose: bool = False, quick: bool = False) -> int:
    """Run all extraction tests.
    
    Args:
        verbose: Whether to show verbose output
        quick: Whether to run quick tests only
        
    Returns:
        Exit code from tests
    """
    test_files = [
        "tests/test_entity_extraction.py",
        "tests/test_relation_extraction.py"
    ]
    
    if quick:
        # Run only basic tests for quick validation
        test_files = [
            "tests/test_entity_extraction.py::test_entity_extraction_basic",
            "tests/test_relation_extraction.py::test_relation_extraction_basic"
        ]
    
    return run_pytest(test_files, verbose, quick)


def run_demo(api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> int:
    """Run the extraction demo.
    
    Args:
        api_key: Optional OpenAI API key
        model: LLM model to use
        
    Returns:
        Exit code from demo
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OpenAI API key is required for demo.")
        return 1
    
    cmd = ["python", "examples/extraction_demo.py", "--api-key", api_key, "--model", model]
    
    print(f"🚀 Running demo: {' '.join(cmd[:-2])} --api-key *** --model {model}")
    print("" + "="*60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("❌ Demo script not found.")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        return 1


def main():
    """Main function to run tests or demo."""
    parser = argparse.ArgumentParser(
        description="Test runner for morag-graph entity and relation extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --entity-only      # Run only entity tests
  python run_tests.py --relation-only    # Run only relation tests
  python run_tests.py --quick            # Run quick validation tests
  python run_tests.py --demo             # Run extraction demo
  python run_tests.py --verbose          # Show detailed output
"""
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (can also be set via GEMINI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--entity-only",
        action="store_true",
        help="Run only entity extraction tests"
    )
    
    parser.add_argument(
        "--relation-only",
        action="store_true",
        help="Run only relation extraction tests"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a subset of tests for quick validation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed test output"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the extraction demo instead of tests"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="LLM model to use for demo (default: gemini-1.5-flash)"
    )
    
    args = parser.parse_args()
    
    print("🧪 MoRAG Graph - Entity and Relation Extraction Test Runner")
    print("" + "="*60)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Setup environment
    if not setup_environment(args.api_key):
        return 1
    
    # Run demo if requested
    if args.demo:
        return run_demo(args.api_key, args.model)
    
    # Determine which tests to run
    if args.entity_only and args.relation_only:
        print("❌ Cannot specify both --entity-only and --relation-only")
        return 1
    
    try:
        if args.entity_only:
            print("🔍 Running entity extraction tests...")
            exit_code = run_entity_tests(args.verbose, args.quick)
        elif args.relation_only:
            print("🔗 Running relation extraction tests...")
            exit_code = run_relation_tests(args.verbose, args.quick)
        else:
            print("🧪 Running all extraction tests...")
            exit_code = run_all_tests(args.verbose, args.quick)
        
        print("" + "="*60)
        if exit_code == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ Tests failed with exit code {exit_code}")
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⏹️  Test run interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test run failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())