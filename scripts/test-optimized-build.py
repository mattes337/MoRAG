#!/usr/bin/env python3
"""
Test script to measure Docker build times for optimized Dockerfiles.
This script helps verify that the build optimization is working correctly.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and measure execution time."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ SUCCESS - Duration: {duration:.2f} seconds")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        return duration, True
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"❌ FAILED - Duration: {duration:.2f} seconds")
        print("STDERR:", e.stderr)
        if e.stdout:
            print("STDOUT:", e.stdout)
        return duration, False


def test_build_optimization():
    """Test the optimized Docker build process."""

    # Change to repository root
    repo_root = Path(__file__).parent.parent
    print(f"Working directory: {repo_root}")

    # Test commands
    tests = [
        {
            "cmd": "docker build --target dependencies -t morag:deps .",
            "description": "Build dependencies layer (should be fast on rebuild)",
        },
        {
            "cmd": "docker build --target development -t morag:dev .",
            "description": "Build development image",
        },
        {
            "cmd": "docker build --target production -t morag:prod .",
            "description": "Build production image",
        },
        {
            "cmd": "docker build -f Dockerfile.worker --target production -t morag:worker .",
            "description": "Build worker image",
        },
    ]

    results = []

    print("🚀 Testing optimized Docker build process...")
    print(f"Repository root: {repo_root.absolute()}")

    # Change to repo root for Docker commands
    import os

    os.chdir(repo_root)

    for test in tests:
        duration, success = run_command(test["cmd"], test["description"])
        results.append(
            {"test": test["description"], "duration": duration, "success": success}
        )

    # Print summary
    print(f"\n{'='*60}")
    print("BUILD TEST SUMMARY")
    print(f"{'='*60}")

    total_time = 0
    successful_builds = 0

    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {result['test']}: {result['duration']:.2f}s")
        total_time += result["duration"]
        if result["success"]:
            successful_builds += 1

    print(f"\nTotal build time: {total_time:.2f} seconds")
    print(f"Successful builds: {successful_builds}/{len(results)}")

    if successful_builds == len(results):
        print("\n🎉 All builds completed successfully!")
        print("\n💡 To test rebuild optimization:")
        print("   1. Make a small change to application code")
        print("   2. Run the same build commands again")
        print("   3. Dependencies layer should be cached and build much faster")
    else:
        print(f"\n⚠️  {len(results) - successful_builds} builds failed")
        return False

    return True


def test_rebuild_optimization():
    """Test that rebuilds are faster by making a small change."""
    print("\n🔄 Testing rebuild optimization...")

    # Create a small test change
    test_file = Path("temp_test_change.txt")
    test_file.write_text(f"Test change at {time.time()}")

    try:
        # Test rebuild
        duration, success = run_command(
            "docker build --target development -t morag:dev-test .",
            "Rebuild after small change (should be fast)",
        )

        if success and duration < 60:  # Should be much faster than initial build
            print(f"✅ Rebuild optimization working! Build took only {duration:.2f}s")
        elif success:
            print(
                f"⚠️  Rebuild completed but took {duration:.2f}s (may not be optimized)"
            )
        else:
            print("❌ Rebuild failed")

    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    print("Docker Build Optimization Test")
    print("=" * 60)

    # Test initial builds
    if test_build_optimization():
        # Test rebuild optimization
        test_rebuild_optimization()

    print("\n🏁 Test completed!")
