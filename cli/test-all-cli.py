#!/usr/bin/env python3
"""
Comprehensive CLI test script to verify all CLI scripts are working.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_cli_test(script_name, description, timeout=30):
    """Run a CLI script with --help and check if it works."""
    print(f"\n{'='*60}")
    print(f"Testing: {script_name}")
    print(f"Description: {description}")
    print(f"{'='*60}")
    
    try:
        # Run the script with --help
        result = subprocess.run(
            [sys.executable, f"cli/{script_name}", "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print(f"[OK] {script_name} - SUCCESS")
            print(f"   Help output length: {len(result.stdout)} characters")
            return True
        else:
            print(f"[FAIL] {script_name} - FAILED")
            print(f"   Return code: {result.returncode}")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} - TIMEOUT (>{timeout}s)")
        return False
    except Exception as e:
        print(f"[ERROR] {script_name} - EXCEPTION: {e}")
        return False

def main():
    """Test all CLI scripts."""
    print("🚀 MoRAG CLI Test Suite")
    print("=" * 60)
    
    # Define all CLI scripts to test
    cli_scripts = [
        ("test-document.py", "Document processing test"),
        ("test-audio.py", "Audio processing test"),
        ("test-video.py", "Video processing test"),
        ("test-image.py", "Image processing test"),
        ("test-web.py", "Web scraping test"),
        ("test-youtube.py", "YouTube processing test"),
        ("test-query.py", "Query testing"),
        ("test-folder.py", "Folder processing test"),
        ("test-simple.py", "Basic system test"),
        ("test-embedding.py", "Embedding test"),
        ("test-cli-fix.py", "CLI fix verification"),
        ("test-openie.py", "OpenIE relation extraction test"),
        ("validate-openie-config.py", "OpenIE configuration validation"),
    ]
    
    results = []
    start_time = time.time()
    
    for script_name, description in cli_scripts:
        success = run_cli_test(script_name, description)
        results.append((script_name, success))
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"🎯 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"[OK] Successful: {successful}/{total}")
    print(f"[FAIL] Failed: {total - successful}/{total}")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"{'='*60}")
    
    # Detailed results
    print(f"\n📊 DETAILED RESULTS:")
    for script_name, success in results:
        status = "[OK] PASS" if success else "[FAIL] FAIL"
        print(f"   {status} - {script_name}")
    
    if successful == total:
        print(f"\n[SUCCESS] ALL CLI SCRIPTS ARE WORKING! [SUCCESS]")
        return 0
    else:
        print(f"\n[WARN]  Some CLI scripts need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
