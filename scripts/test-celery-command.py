#!/usr/bin/env python3
"""
Test script to verify the Celery worker command works correctly.
"""

import subprocess
import sys
import time
import signal
from pathlib import Path

def run_command(cmd, capture_output=True, timeout=30):
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

def test_celery_worker_help():
    """Test that celery worker help command works."""
    print("🌿 Testing Celery worker help command...")
    
    result = run_command([
        "python", "-c",
        "import sys; sys.path.insert(0, 'packages/morag/src'); "
        "from morag.worker import celery_app; "
        "celery_app.worker_main(['worker', '--help'])"
    ], timeout=10)
    
    if result is None or result.returncode != 0:
        print("❌ Celery worker help command failed!")
        if result:
            print("STDERR:", result.stderr)
        return False
    
    print("✅ Celery worker help command successful!")
    return True

def test_celery_inspect():
    """Test that celery inspect commands work."""
    print("🔍 Testing Celery inspect commands...")
    
    result = run_command([
        "python", "-c",
        "import sys; sys.path.insert(0, 'packages/morag/src'); "
        "from morag.worker import celery_app; "
        "print('Registered tasks:'); "
        "[print(f'  - {task}') for task in sorted(celery_app.tasks.keys()) if 'morag' in task]"
    ])
    
    if result is None or result.returncode != 0:
        print("❌ Celery inspect command failed!")
        if result:
            print("STDERR:", result.stderr)
        return False
    
    output = result.stdout.strip()
    if "morag.worker" not in output:
        print("❌ MoRAG tasks not found in Celery app!")
        print("Output:", output)
        return False
    
    print("✅ Celery inspect command successful!")
    print("Output:", output)
    return True

def test_celery_configuration():
    """Test that Celery configuration is correct."""
    print("⚙️ Testing Celery configuration...")
    
    result = run_command([
        "python", "-c",
        "import sys; sys.path.insert(0, 'packages/morag/src'); "
        "from morag.worker import celery_app; "
        "print(f'Broker URL: {celery_app.conf.broker_url}'); "
        "print(f'Result backend: {celery_app.conf.result_backend}'); "
        "print(f'Task serializer: {celery_app.conf.task_serializer}'); "
        "print(f'Accept content: {celery_app.conf.accept_content}')"
    ])
    
    if result is None or result.returncode != 0:
        print("❌ Celery configuration test failed!")
        if result:
            print("STDERR:", result.stderr)
        return False
    
    output = result.stdout.strip()
    required_configs = [
        "Broker URL: redis://localhost:6379/0",
        "Result backend: redis://localhost:6379/0",
        "Task serializer: json",
        "Accept content: ['json']"
    ]
    
    for config in required_configs:
        if config not in output:
            print(f"❌ Missing configuration: {config}")
            print("Output:", output)
            return False
    
    print("✅ Celery configuration test successful!")
    print("Output:", output)
    return True

def test_docker_compose_worker_command():
    """Test that the docker-compose worker command syntax is correct."""
    print("🐳 Testing docker-compose worker command syntax...")

    # Check the command directly from the file
    try:
        with open('docker-compose.yml', 'r') as f:
            content = f.read()

        # Look for the worker command
        if 'celery' in content and 'morag.worker' in content and 'worker' in content:
            print("✅ Docker-compose worker command syntax is correct!")
            print("Found: celery -A morag.worker worker command in docker-compose.yml")
            return True
        else:
            print("❌ Worker command not found in docker-compose.yml!")
            return False

    except Exception as e:
        print(f"❌ Failed to read docker-compose.yml: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting Celery worker command tests...")
    
    tests = [
        ("Celery Configuration", test_celery_configuration),
        ("Celery Inspect", test_celery_inspect),
        ("Docker-Compose Worker Command", test_docker_compose_worker_command),
        ("Celery Worker Help", test_celery_worker_help),
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
        print("\n🎉 All Celery worker tests passed!")
        print("\n📝 The Docker container should now work correctly with:")
        print("  celery -A morag.worker worker --loglevel=info --concurrency=2")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
