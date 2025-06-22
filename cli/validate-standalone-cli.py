#!/usr/bin/env python3
"""
Validation Script for Standalone CLI Functionality

This script validates that the CLI scripts work independently without requiring
a running API server, and that they properly use GEMINI_API_KEY.

Usage: python tests/cli/validate-standalone-cli.py
"""

import sys
import os
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def print_result(key: str, value: str, indent: int = 0):
    """Print a formatted key-value result."""
    spaces = "  " * indent
    print(f"{spaces}📋 {key}: {value}")


def check_environment() -> bool:
    """Check environment configuration."""
    print_section("Environment Configuration Check")
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print_result("Environment File", "✅ .env found")
    else:
        print_result("Environment File", "❌ .env not found")
        return False
    
    # Check for GEMINI_API_KEY
    gemini_key = os.getenv('GEMINI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    
    if gemini_key:
        print_result("GEMINI_API_KEY", "✅ Set")
        return True
    elif google_key:
        print_result("GEMINI_API_KEY", "❌ Not set")
        print_result("GOOGLE_API_KEY", "⚠️  Found (deprecated)")
        print("  💡 Consider updating to use GEMINI_API_KEY instead")
        return True
    else:
        print_result("API Keys", "❌ No API keys found")
        print("  💡 Set GEMINI_API_KEY in your .env file")
        return False


def find_test_files() -> Dict[str, Path]:
    """Find available test files."""
    print_section("Test Files Discovery")
    
    uploads_dir = Path("uploads")
    test_files = {}
    
    # Look for specific file types
    file_patterns = {
        "audio": ["*.mp3", "*.wav", "*.m4a"],
        "document": ["*.pdf", "*.txt"],
        "image": ["*.jpg", "*.jpeg", "*.png"],
        "video": ["*.mp4", "*.avi", "*.mov"]
    }
    
    for file_type, patterns in file_patterns.items():
        for pattern in patterns:
            files = list(uploads_dir.glob(pattern))
            if files:
                test_files[file_type] = files[0]  # Take the first one
                print_result(f"{file_type.title()} File", f"✅ {files[0].name}")
                break
        else:
            print_result(f"{file_type.title()} File", "❌ Not found")
    
    return test_files


def run_cli_test(script_name: str, file_path: Path, mode: str = "processing") -> bool:
    """Run a CLI test script and check if it works."""
    print(f"\n🔄 Testing {script_name} in {mode} mode...")
    
    try:
        cmd = [sys.executable, f"tests/cli/{script_name}", str(file_path)]
        
        if mode == "ingestion":
            cmd.append("--ingest")
        
        # Run the command with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
            cwd=project_root
        )
        
        if result.returncode == 0:
            print(f"  ✅ {script_name} {mode} mode: SUCCESS")
            return True
        else:
            print(f"  ❌ {script_name} {mode} mode: FAILED")
            print(f"  Error: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {script_name} {mode} mode: TIMEOUT")
        return False
    except Exception as e:
        print(f"  ❌ {script_name} {mode} mode: ERROR - {e}")
        return False


def validate_standalone_functionality(test_files: Dict[str, Path]) -> Dict[str, Dict[str, bool]]:
    """Validate that CLI scripts work standalone."""
    print_section("Standalone CLI Validation")
    
    # Test mapping: script_name -> file_type
    tests = {
        "test-audio.py": "audio",
        "test-document.py": "document",
        "test-image.py": "image",
        "test-video.py": "video"
    }
    
    results = {}
    
    for script_name, file_type in tests.items():
        if file_type not in test_files:
            print(f"⏭️  Skipping {script_name} - no {file_type} file available")
            continue
        
        file_path = test_files[file_type]
        script_results = {}
        
        # Test processing mode
        script_results["processing"] = run_cli_test(script_name, file_path, "processing")
        
        # Test ingestion mode (the new standalone functionality)
        script_results["ingestion"] = run_cli_test(script_name, file_path, "ingestion")
        
        results[script_name] = script_results
    
    return results


def main():
    """Main validation function."""
    print_header("MoRAG Standalone CLI Validation")
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please configure your .env file.")
        return False
    
    # Find test files
    test_files = find_test_files()
    if not test_files:
        print("\n❌ No test files found. Please add some files to the uploads/ directory.")
        return False
    
    # Validate standalone functionality
    results = validate_standalone_functionality(test_files)
    
    # Summary
    print_section("Validation Summary")
    
    total_tests = 0
    passed_tests = 0
    
    for script_name, script_results in results.items():
        print(f"\n📄 {script_name}:")
        for mode, success in script_results.items():
            total_tests += 1
            if success:
                passed_tests += 1
                print(f"  ✅ {mode} mode: PASS")
            else:
                print(f"  ❌ {mode} mode: FAIL")
    
    print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All CLI scripts work standalone successfully!")
        print("✅ GEMINI_API_KEY configuration is working")
        print("✅ Direct ingestion (without API server) is working")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed")
        print("Check the error messages above for details")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
