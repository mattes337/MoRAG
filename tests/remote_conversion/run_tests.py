#!/usr/bin/env python3
"""Test runner for remote conversion system."""

import os
import subprocess
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-services" / "src"))


def run_tests():
    """Run all remote conversion tests."""
    test_dir = Path(__file__).parent

    print("🧪 Running Remote Conversion System Tests")
    print("=" * 50)

    # Test files to run
    test_files = [
        "test_remote_job_model.py",
        "test_remote_job_repository.py",
        "test_remote_job_service.py",
        "test_remote_job_api.py",
        "test_ingestion_integration.py",
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_file in test_files:
        test_path = test_dir / test_file
        if not test_path.exists():
            print(f"❌ Test file not found: {test_file}")
            continue

        print(f"\n📋 Running {test_file}...")
        print("-" * 30)

        try:
            # Run pytest on the specific file
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            if result.returncode == 0:
                print(f"✅ {test_file} - All tests passed")
                # Count passed tests from output
                lines = result.stdout.split("\n")
                for line in lines:
                    if "passed" in line and "failed" not in line:
                        try:
                            count = int(line.split()[0])
                            passed_tests += count
                            total_tests += count
                        except:
                            pass
            else:
                print(f"❌ {test_file} - Some tests failed")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                failed_tests += 1

        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            failed_tests += 1

    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    print(f"Total test files: {len(test_files)}")
    print(f"Passed tests: {passed_tests}")
    print(f"Failed tests: {failed_tests}")
    print(f"Total tests: {total_tests}")

    if failed_tests == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
