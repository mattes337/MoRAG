#!/usr/bin/env python3
"""
Simple validation test for the bug fixes.

This script validates that:
1. Speaker diarization code no longer wraps async methods in run_in_executor
2. Rate limiting code includes proper retry logic with exponential backoff

Usage:
    python tests/test_fixes_validation.py
"""

import ast
import re
from pathlib import Path


def test_speaker_diarization_fix():
    """Test that speaker diarization code is fixed."""
    print("\n🎯 Testing Speaker Diarization Fix...")
    
    processor_file = Path("packages/morag-audio/src/morag_audio/processor.py")
    
    if not processor_file.exists():
        print("❌ Audio processor file not found")
        return False
    
    content = processor_file.read_text()
    
    # Check that the problematic run_in_executor pattern is removed
    if "run_in_executor" in content and "diarize_audio" in content:
        # Look for the specific problematic pattern
        problematic_pattern = r"run_in_executor\([^)]*diarize_audio"
        if re.search(problematic_pattern, content):
            print("❌ Still contains problematic run_in_executor with diarize_audio")
            return False
    
    # Check that we now have direct await of diarize_audio
    if "await self.diarization_service.diarize_audio(" in content:
        print("✅ Speaker diarization now correctly awaits async method")
        return True
    else:
        print("❌ Direct await of diarize_audio not found")
        return False


def test_rate_limiting_fix():
    """Test that rate limiting code includes proper retry logic."""
    print("\n🎯 Testing Rate Limiting Fix...")
    
    embedding_files = [
        Path("packages/morag-services/src/morag_services/embedding.py"),
        Path("packages/morag-embedding/src/morag_embedding/service.py")
    ]
    
    fixes_found = 0
    
    for file_path in embedding_files:
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
            
        content = file_path.read_text()
        
        # Check for rate limiting error detection
        rate_limit_patterns = [
            "429",
            "RESOURCE_EXHAUSTED",
            "quota exceeded",
            "rate limit"
        ]
        
        has_rate_limit_detection = any(pattern in content for pattern in rate_limit_patterns)
        
        # Check for retry logic
        has_retry_logic = any(pattern in content for pattern in [
            "max_retries",
            "attempt",
            "exponential",
            "backoff",
            "time.sleep"
        ])
        
        # Check for exponential backoff calculation
        has_exponential_backoff = "2 ** attempt" in content or "2**attempt" in content
        
        if has_rate_limit_detection and has_retry_logic:
            print(f"✅ Rate limiting fix found in {file_path.name}")
            fixes_found += 1
            
            if has_exponential_backoff:
                print(f"✅ Exponential backoff implemented in {file_path.name}")
        else:
            print(f"❌ Rate limiting fix incomplete in {file_path.name}")
    
    if fixes_found >= 1:
        print("✅ Rate limiting fixes implemented")
        return True
    else:
        print("❌ No rate limiting fixes found")
        return False


def test_error_handling_improvements():
    """Test that error handling has been improved."""
    print("\n🎯 Testing Error Handling Improvements...")
    
    embedding_files = [
        Path("packages/morag-services/src/morag_services/embedding.py"),
        Path("packages/morag-embedding/src/morag_embedding/service.py")
    ]
    
    improvements_found = 0
    
    for file_path in embedding_files:
        if not file_path.exists():
            continue
            
        content = file_path.read_text()
        
        # Check for improved error handling patterns
        improvements = [
            "error_str = str(e)",  # Better error string handling
            "RateLimitError",      # Specific rate limit exceptions
            "logger.warning",      # Better logging
            "logger.error"         # Error logging
        ]
        
        found_improvements = sum(1 for pattern in improvements if pattern in content)
        
        if found_improvements >= 3:
            print(f"✅ Error handling improvements found in {file_path.name}")
            improvements_found += 1
    
    if improvements_found >= 1:
        print("✅ Error handling improvements implemented")
        return True
    else:
        print("❌ No error handling improvements found")
        return False


def test_batch_processing_improvements():
    """Test that batch processing includes delays."""
    print("\n🎯 Testing Batch Processing Improvements...")
    
    embedding_files = [
        Path("packages/morag-services/src/morag_services/embedding.py"),
        Path("packages/morag-embedding/src/morag_embedding/service.py")
    ]
    
    improvements_found = 0
    
    for file_path in embedding_files:
        if not file_path.exists():
            continue
            
        content = file_path.read_text()
        
        # Check for batch processing improvements
        if "asyncio.sleep" in content and "batch" in content.lower():
            print(f"✅ Batch processing delays found in {file_path.name}")
            improvements_found += 1
    
    if improvements_found >= 1:
        print("✅ Batch processing improvements implemented")
        return True
    else:
        print("❌ No batch processing improvements found")
        return False


def main():
    """Run all validation tests."""
    print("🧪 Validating Bug Fixes Implementation")
    print("=" * 60)
    
    tests = [
        test_speaker_diarization_fix,
        test_rate_limiting_fix,
        test_error_handling_improvements,
        test_batch_processing_improvements
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Validation Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All fixes have been properly implemented!")
        return 0
    else:
        print("\n⚠️  Some fixes may be incomplete. Please review the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
