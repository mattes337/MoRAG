#!/usr/bin/env python3
"""Validation script for GPU workers implementation."""

import sys
import importlib.util
from pathlib import Path

def validate_python_file(file_path):
    """Validate that a Python file can be imported without syntax errors."""
    try:
        spec = importlib.util.spec_from_file_location("module", file_path)
        if spec is None:
            return False, f"Could not create spec for {file_path}"
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, "OK"
    except Exception as e:
        return False, str(e)

def main():
    """Validate all GPU workers implementation files."""
    print("🔍 Validating GPU Workers Implementation")
    print("=" * 50)
    
    # Files to validate
    files_to_check = [
        "packages/morag/src/morag/worker.py",
        "packages/morag/src/morag/ingest_tasks.py", 
        "packages/morag/src/morag/server.py",
        "packages/morag/src/morag/services/task_router.py",
        "packages/morag/src/morag/services/file_transfer.py",
        "tests/test-gpu-workers.py",
        "tests/test-gpu-setup.py",
        "scripts/test-gpu-worker-config.py"
    ]
    
    project_root = Path(__file__).parent.parent
    
    passed = 0
    total = len(files_to_check)
    
    for file_path in files_to_check:
        full_path = project_root / file_path
        
        if not full_path.exists():
            print(f"❌ {file_path}: File not found")
            continue
            
        success, message = validate_python_file(full_path)
        
        if success:
            print(f"✅ {file_path}: {message}")
            passed += 1
        else:
            print(f"❌ {file_path}: {message}")
    
    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {passed}/{total} files passed")
    
    if passed == total:
        print("✅ All GPU workers implementation files are valid!")
        
        # Check for key features
        print("\n🔍 Checking Key Features:")
        
        # Check if GPU task variants exist
        try:
            sys.path.insert(0, str(project_root / "packages/morag/src"))
            from morag.worker import process_file_task_gpu, process_url_task_gpu
            print("✅ GPU task variants found")
        except ImportError as e:
            print(f"❌ GPU task variants missing: {e}")
        
        # Check if task router exists
        try:
            from morag.services.task_router import TaskRouter, get_task_router
            print("✅ Task router service found")
        except ImportError as e:
            print(f"❌ Task router missing: {e}")
        
        # Check if file transfer service exists
        try:
            from morag.services.file_transfer import FileTransferService
            print("✅ File transfer service found")
        except ImportError as e:
            print(f"❌ File transfer service missing: {e}")
        
        print("\n🎉 GPU Workers implementation validation complete!")
        return True
    else:
        print("❌ Some files have validation errors!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
