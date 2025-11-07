#!/usr/bin/env python3
"""
Test script to verify Docker build and module imports.
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def run_command(cmd, capture_output=True, timeout=300):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Error running command: {e}")
        return None

def test_docker_build():
    """Test Docker build process."""
    print("🔨 Testing Docker build...")

    # Build the production image
    result = run_command([
        "docker", "build",
        "--target", "production",
        "--tag", "morag:test",
        "."
    ], timeout=600)

    if result is None or result.returncode != 0:
        print("❌ Docker build failed!")
        if result:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        return False

    print("✅ Docker build successful!")
    return True

def test_module_import():
    """Test that the morag module can be imported in the container."""
    print("🐍 Testing module import...")

    # Test basic import
    result = run_command([
        "docker", "run", "--rm",
        "morag:test",
        "python", "-c", "import morag; print('morag module imported successfully')"
    ])

    if result is None or result.returncode != 0:
        print("❌ Module import failed!")
        if result:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        return False

    print("✅ Module import successful!")
    return True

def test_worker_module():
    """Test that the worker module can be imported."""
    print("👷 Testing worker module...")

    result = run_command([
        "docker", "run", "--rm",
        "morag:test",
        "python", "-c", "import morag.worker; print('worker module imported successfully')"
    ])

    if result is None or result.returncode != 0:
        print("❌ Worker module import failed!")
        if result:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        return False

    print("✅ Worker module import successful!")
    return True

def test_celery_app():
    """Test that Celery app can be created."""
    print("🌿 Testing Celery app...")

    result = run_command([
        "docker", "run", "--rm",
        "morag:test",
        "python", "-c",
        "from morag.worker import celery_app; print(f'Celery app: {celery_app.main}')"
    ])

    if result is None or result.returncode != 0:
        print("❌ Celery app test failed!")
        if result:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        return False

    print("✅ Celery app test successful!")
    return True

def test_package_versions():
    """Test that all packages are properly installed."""
    print("📦 Testing package versions...")

    packages = [
        "morag",
        "morag-core",
        "morag-services",
        "morag-audio",
        "morag-video",
        "morag-document",
        "morag-image",
        "morag-web",
        "morag-youtube",
        "morag-embedding"
    ]

    for package in packages:
        result = run_command([
            "docker", "run", "--rm",
            "morag:test",
            "python", "-c",
            f"import pkg_resources; print(f'{package}: {{pkg_resources.get_distribution(\"{package}\").version}}')"
        ])

        if result is None or result.returncode != 0:
            print(f"❌ Package {package} not found!")
            if result:
                print("STDERR:", result.stderr)
            return False
        else:
            print(f"✅ {result.stdout.strip()}")

    return True

def cleanup():
    """Clean up test resources."""
    print("🧹 Cleaning up...")
    run_command(["docker", "rmi", "morag:test"], capture_output=False)

def main():
    """Main test function."""
    print("🚀 Starting Docker build and import tests...")

    tests = [
        ("Docker Build", test_docker_build),
        ("Module Import", test_module_import),
        ("Worker Module", test_worker_module),
        ("Celery App", test_celery_app),
        ("Package Versions", test_package_versions),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)

        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results[test_name] = False

    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)

    all_passed = True
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed!")
        cleanup()
        return 0
    else:
        print("\n💥 Some tests failed!")
        print("Docker image 'morag:test' left for debugging.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
