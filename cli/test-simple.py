#!/usr/bin/env python3
"""
Simple MoRAG System Test Script

Usage: python test-simple.py

This script runs basic tests to verify MoRAG components can be imported and initialized.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


def test_imports():
    """Test that all packages can be imported."""
    print_section("Testing Package Imports")
    
    packages = [
        ("morag_core.config", "Settings"),
        ("morag_core.models", "ProcessingConfig"),
        ("morag_services", "ServiceConfig, ContentType"),
        ("morag_audio", "AudioProcessor"),
        ("morag_document", "DocumentProcessor"),
        ("morag_video", "VideoProcessor"),
        ("morag_image", "ImageProcessor"),
        ("morag_web", "WebProcessor"),
        ("morag_youtube", "YouTubeProcessor"),
    ]
    
    results = {}
    
    for package, items in packages:
        try:
            exec(f"from {package} import {items}")
            print(f"✅ {package}: {items}")
            results[package] = True
        except ImportError as e:
            print(f"❌ {package}: {e}")
            results[package] = False
        except Exception as e:
            print(f"⚠️  {package}: {e}")
            results[package] = False
    
    return results


def test_basic_functionality():
    """Test basic functionality."""
    print_section("Testing Basic Functionality")
    
    try:
        from morag_services import ServiceConfig, ContentType
        from morag_core.models import ProcessingConfig
        
        # Test ServiceConfig creation
        config = ServiceConfig()
        print("✅ ServiceConfig creation successful")
        
        # Test ProcessingConfig creation
        proc_config = ProcessingConfig()
        print("✅ ProcessingConfig creation successful")
        
        # Test ContentType enum
        content_types = [
            ContentType.DOCUMENT,
            ContentType.AUDIO,
            ContentType.VIDEO,
            ContentType.IMAGE,
            ContentType.WEB,
            ContentType.YOUTUBE
        ]
        print(f"✅ ContentType enum has {len(content_types)} types")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_sample_files():
    """Test for sample files."""
    print_section("Checking Sample Files")
    
    sample_files = [
        "uploads/Sprache.mp3",
        "uploads/recording.m4a",
        "uploads/super saftiger Schoko Schmand Kuchen fur ein ganzes Blech - 2014-02-08.pdf",
        "uploads/test_video.mp4",
    ]
    
    found_files = []
    
    for file_path in sample_files:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"✅ {path.name} ({size_mb:.2f} MB)")
            found_files.append(file_path)
        else:
            print(f"❌ {file_path} not found")
    
    return found_files


def test_docker_files():
    """Test for Docker configuration files."""
    print_section("Checking Docker Configuration")
    
    docker_files = [
        "docker-compose.yml",
        "docker-compose.dev.yml",
        "docker-compose.microservices.yml",
        "Dockerfile",
        ".env.example",
    ]
    
    for file_path in docker_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} not found")


def test_environment():
    """Test environment configuration."""
    print_section("Checking Environment Configuration")

    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("❌ .env file not found (copy from .env.example)")

    # Check for required environment variables
    required_vars = ["GEMINI_API_KEY"]
    optional_vars = ["QDRANT_HOST", "QDRANT_PORT", "REDIS_URL"]

    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var} is set")
        else:
            # Check for backward compatibility
            if var == "GEMINI_API_KEY" and os.getenv("GOOGLE_API_KEY"):
                print(f"⚠️  {var} not set, but GOOGLE_API_KEY found (consider updating)")
            else:
                print(f"❌ {var} is not set")

    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var} is set ({value})")
        else:
            print(f"ℹ️  {var} not set (using defaults)")


def test_documentation():
    """Test for documentation files."""
    print_section("Checking Documentation")

    doc_files = [
        "README.md",
        "docs/DOCKER_DEPLOYMENT.md",
        "docs/ARCHITECTURE.md",
        "docs/DEVELOPMENT_GUIDE.md",
        "TASKS.md",
        "COMPLETION_SUMMARY.md",
    ]

    for file_path in doc_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} not found")


def main():
    """Main test function."""
    print_header("MoRAG Simple System Test")
    
    # Test imports
    import_results = test_imports()
    
    # Test basic functionality
    basic_test = test_basic_functionality()
    
    # Check sample files
    sample_files = test_sample_files()

    # Check environment
    test_environment()

    # Check Docker files
    test_docker_files()

    # Check documentation
    test_documentation()
    
    # Summary
    print_section("Test Summary")
    
    total_packages = len(import_results)
    successful_imports = sum(import_results.values())
    
    print(f"📦 Package Imports: {successful_imports}/{total_packages} successful")
    print(f"⚙️  Basic Functionality: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"📁 Sample Files: {len(sample_files)} found")
    
    if successful_imports >= total_packages * 0.8 and basic_test:
        print("\n🎉 MoRAG system appears to be working correctly!")
        print("\nNext steps:")
        print("1. Try individual test scripts:")
        print("   python tests/cli/test-audio.py uploads/Sprache.mp3")
        print("   python tests/cli/test-document.py uploads/your-document.pdf")
        print("2. Start the system with Docker:")
        print("   docker-compose up -d")
        print("3. Check the API documentation:")
        print("   http://localhost:8000/docs")
        return True
    else:
        print("\n⚠️  MoRAG system has some issues.")
        print("Check the error messages above and ensure all packages are installed.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
