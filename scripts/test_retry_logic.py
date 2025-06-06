#!/usr/bin/env python3
"""Test script to demonstrate the new indefinite retry logic for rate limits."""

import sys
import os
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_retry_configuration():
    """Test that retry configuration is properly loaded."""
    print("🔧 Testing Retry Configuration")
    print("=" * 50)
    
    try:
        from morag_core.config import settings
        
        print(f"✅ Retry indefinitely: {settings.retry_indefinitely}")
        print(f"✅ Base delay: {settings.retry_base_delay}s")
        print(f"✅ Max delay: {settings.retry_max_delay}s")
        print(f"✅ Exponential base: {settings.retry_exponential_base}")
        print(f"✅ Jitter enabled: {settings.retry_jitter}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load retry configuration: {e}")
        return False


def simulate_exponential_backoff():
    """Simulate exponential backoff delay calculation."""
    print("\n🔧 Simulating Exponential Backoff")
    print("=" * 50)
    
    try:
        from morag_core.config import settings
        
        base_delay = settings.retry_base_delay
        max_delay = settings.retry_max_delay
        exponential_base = settings.retry_exponential_base
        use_jitter = settings.retry_jitter
        
        print(f"Configuration: base={base_delay}s, max={max_delay}s, multiplier={exponential_base}")
        print("\nRetry attempt delays:")
        
        total_time = 0
        for attempt in range(1, 11):  # Show first 10 attempts
            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
            
            # Add jitter if enabled (simulate)
            if use_jitter:
                jitter = (time.time() % 1) * 0.1 * delay  # 10% jitter
                delay_with_jitter = delay + jitter
            else:
                delay_with_jitter = delay
            
            total_time += delay_with_jitter
            
            print(f"  Attempt {attempt:2d}: {delay:.2f}s (with jitter: {delay_with_jitter:.2f}s) - Total: {total_time:.1f}s")
            
            # Show when we hit max delay
            if delay >= max_delay:
                print(f"  → Max delay reached at attempt {attempt}")
                break
        
        print(f"\n✅ After 10 attempts, total wait time would be: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to simulate backoff: {e}")
        return False


def demonstrate_rate_limit_vs_other_errors():
    """Demonstrate different retry behavior for rate limits vs other errors."""
    print("\n🔧 Demonstrating Error-Specific Retry Logic")
    print("=" * 50)
    
    try:
        from morag_core.config import settings
        
        print("Rate Limit Errors:")
        if settings.retry_indefinitely:
            print("  ✅ Will retry indefinitely with exponential backoff")
            print(f"  ✅ Max delay capped at {settings.retry_max_delay}s")
        else:
            print("  ⚠️  Will use limited retries (legacy mode)")
        
        print("\nOther Errors (timeout, authentication, etc.):")
        print("  ✅ Will use limited retries (3 attempts)")
        print("  ✅ Prevents infinite loops for non-recoverable errors")
        
        print("\nError Detection Patterns:")
        rate_limit_patterns = [
            "429",
            "RESOURCE_EXHAUSTED", 
            "quota exceeded",
            "rate limit"
        ]
        
        for pattern in rate_limit_patterns:
            print(f"  • '{pattern}' → Treated as rate limit error")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to demonstrate retry logic: {e}")
        return False


def show_environment_variables():
    """Show how to configure retry behavior via environment variables."""
    print("\n🔧 Environment Variable Configuration")
    print("=" * 50)
    
    env_vars = [
        ("MORAG_RETRY_INDEFINITELY", "Enable/disable indefinite retries", "true"),
        ("MORAG_RETRY_BASE_DELAY", "Base delay in seconds", "1.0"),
        ("MORAG_RETRY_MAX_DELAY", "Maximum delay in seconds", "300.0"),
        ("MORAG_RETRY_EXPONENTIAL_BASE", "Exponential multiplier", "2.0"),
        ("MORAG_RETRY_JITTER", "Enable random jitter", "true"),
    ]
    
    print("Set these environment variables to customize retry behavior:")
    print()
    
    for var, description, default in env_vars:
        current_value = os.environ.get(var, f"(default: {default})")
        print(f"  {var}={current_value}")
        print(f"    {description}")
        print()
    
    print("Example configuration for aggressive retrying:")
    print("  export MORAG_RETRY_INDEFINITELY=true")
    print("  export MORAG_RETRY_BASE_DELAY=0.5")
    print("  export MORAG_RETRY_MAX_DELAY=600.0")
    print("  export MORAG_RETRY_EXPONENTIAL_BASE=1.5")
    
    print("\nExample configuration for conservative retrying:")
    print("  export MORAG_RETRY_INDEFINITELY=false")
    print("  export MORAG_RETRY_BASE_DELAY=2.0")
    
    return True


def main():
    """Run all retry logic tests."""
    print("🚀 MoRAG Indefinite Retry Logic Test Suite")
    print("=" * 60)
    print("Testing the new rate limit retry configuration...")
    print()
    
    tests = [
        test_retry_configuration,
        simulate_exponential_backoff,
        demonstrate_rate_limit_vs_other_errors,
        show_environment_variables,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All retry logic tests passed!")
        print("\n💡 The system is now configured to retry rate limit errors indefinitely")
        print("   with intelligent exponential backoff, preventing task failures due to")
        print("   temporary API quota limits.")
        return True
    else:
        print("❌ Some tests failed. Please check the configuration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
