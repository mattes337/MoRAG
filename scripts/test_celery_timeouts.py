#!/usr/bin/env python3
"""Test script to verify Celery timeout configuration."""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_timeout_configuration():
    """Test that Celery timeout configuration is properly loaded."""
    print("🔧 Testing Celery Timeout Configuration")
    print("=" * 50)
    
    try:
        from morag_core.config import settings
        
        # Display current configuration
        soft_limit_hours = settings.celery_task_soft_time_limit / 3600
        hard_limit_hours = settings.celery_task_time_limit / 3600
        
        print(f"✅ Soft time limit: {settings.celery_task_soft_time_limit}s ({soft_limit_hours:.1f} hours)")
        print(f"✅ Hard time limit: {settings.celery_task_time_limit}s ({hard_limit_hours:.1f} hours)")
        print(f"✅ Worker prefetch multiplier: {settings.celery_worker_prefetch_multiplier}")
        print(f"✅ Worker max tasks per child: {settings.celery_worker_max_tasks_per_child}")
        
        # Validate configuration
        if settings.celery_task_soft_time_limit >= settings.celery_task_time_limit:
            print("⚠️  WARNING: Soft limit should be less than hard limit")
            return False
        
        if settings.celery_task_soft_time_limit < 3600:  # Less than 1 hour
            print("⚠️  WARNING: Soft limit might be too short for large documents with retries")
        
        if settings.celery_task_time_limit < 3600:  # Less than 1 hour
            print("⚠️  WARNING: Hard limit might be too short for large documents with retries")
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load timeout configuration: {e}")
        return False


def test_environment_variable_override():
    """Test environment variable override functionality."""
    print("\n🔧 Testing Environment Variable Override")
    print("=" * 50)
    
    try:
        from unittest.mock import patch
        import os
        
        # Test with custom environment variables
        test_env = {
            'MORAG_CELERY_TASK_SOFT_TIME_LIMIT': '14400',  # 4 hours
            'MORAG_CELERY_TASK_TIME_LIMIT': '18000',       # 5 hours
            'MORAG_CELERY_WORKER_PREFETCH_MULTIPLIER': '2',
            'MORAG_CELERY_WORKER_MAX_TASKS_PER_CHILD': '500'
        }
        
        with patch.dict(os.environ, test_env):
            from morag_core.config import Settings
            test_settings = Settings()
            
            assert test_settings.celery_task_soft_time_limit == 14400, f"Expected 14400, got {test_settings.celery_task_soft_time_limit}"
            assert test_settings.celery_task_time_limit == 18000, f"Expected 18000, got {test_settings.celery_task_time_limit}"
            assert test_settings.celery_worker_prefetch_multiplier == 2, f"Expected 2, got {test_settings.celery_worker_prefetch_multiplier}"
            assert test_settings.celery_worker_max_tasks_per_child == 500, f"Expected 500, got {test_settings.celery_worker_max_tasks_per_child}"
            
            print("✅ Environment variable override works correctly")
            print(f"   Soft limit: {test_settings.celery_task_soft_time_limit}s ({test_settings.celery_task_soft_time_limit/3600:.1f}h)")
            print(f"   Hard limit: {test_settings.celery_task_time_limit}s ({test_settings.celery_task_time_limit/3600:.1f}h)")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment variable override test failed: {e}")
        return False


def show_timeout_scenarios():
    """Show different timeout scenarios and their implications."""
    print("\n🔧 Timeout Scenarios and Implications")
    print("=" * 50)
    
    try:
        from morag_core.config import settings
        
        scenarios = [
            {
                "name": "Large PDF with Rate Limits",
                "description": "100MB PDF with docling + embedding generation hitting rate limits",
                "estimated_time": "30-90 minutes",
                "retry_time": "Additional 30-60 minutes for retries"
            },
            {
                "name": "High-Resolution Video",
                "description": "4K video processing with audio transcription and frame extraction",
                "estimated_time": "45-120 minutes",
                "retry_time": "Additional 15-30 minutes for retries"
            },
            {
                "name": "Large Batch Processing",
                "description": "100 documents with embedding generation",
                "estimated_time": "60-180 minutes",
                "retry_time": "Additional 60-120 minutes for retries"
            },
            {
                "name": "Web Scraping with JS Rendering",
                "description": "Complex web pages with dynamic content",
                "estimated_time": "5-15 minutes",
                "retry_time": "Additional 5-10 minutes for retries"
            }
        ]
        
        current_soft_limit_hours = settings.celery_task_soft_time_limit / 3600
        current_hard_limit_hours = settings.celery_task_time_limit / 3600
        
        print(f"Current Configuration: {current_soft_limit_hours:.1f}h soft / {current_hard_limit_hours:.1f}h hard")
        print()
        
        for scenario in scenarios:
            print(f"📋 {scenario['name']}")
            print(f"   Description: {scenario['description']}")
            print(f"   Estimated time: {scenario['estimated_time']}")
            print(f"   With retries: {scenario['retry_time']}")
            
            # Rough estimate if this would fit in current limits
            max_time_estimate = 180  # minutes (3 hours) - worst case
            if current_soft_limit_hours * 60 >= max_time_estimate:
                print("   ✅ Should fit within current limits")
            else:
                print("   ⚠️  Might exceed current limits")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to show timeout scenarios: {e}")
        return False


def show_configuration_recommendations():
    """Show configuration recommendations for different use cases."""
    print("\n🔧 Configuration Recommendations")
    print("=" * 50)
    
    recommendations = [
        {
            "use_case": "Development/Testing",
            "soft_limit": "1800",  # 30 minutes
            "hard_limit": "3600",  # 1 hour
            "description": "Short timeouts for quick feedback during development"
        },
        {
            "use_case": "Production (Light Workload)",
            "soft_limit": "3600",  # 1 hour
            "hard_limit": "5400",  # 1.5 hours
            "description": "Moderate timeouts for typical document processing"
        },
        {
            "use_case": "Production (Heavy Workload)",
            "soft_limit": "7200",  # 2 hours (default)
            "hard_limit": "9000",  # 2.5 hours (default)
            "description": "Extended timeouts for large documents and rate limit handling"
        },
        {
            "use_case": "Production (Batch Processing)",
            "soft_limit": "14400",  # 4 hours
            "hard_limit": "18000",  # 5 hours
            "description": "Very long timeouts for large batch operations"
        }
    ]
    
    print("Environment variable configurations for different use cases:")
    print()
    
    for rec in recommendations:
        soft_hours = int(rec["soft_limit"]) / 3600
        hard_hours = int(rec["hard_limit"]) / 3600
        
        print(f"🎯 {rec['use_case']}")
        print(f"   {rec['description']}")
        print(f"   MORAG_CELERY_TASK_SOFT_TIME_LIMIT={rec['soft_limit']}  # {soft_hours:.1f} hours")
        print(f"   MORAG_CELERY_TASK_TIME_LIMIT={rec['hard_limit']}       # {hard_hours:.1f} hours")
        print()
    
    return True


def main():
    """Run all Celery timeout tests."""
    print("🚀 MoRAG Celery Timeout Configuration Test Suite")
    print("=" * 60)
    print("Testing configurable Celery task timeouts...")
    print()
    
    tests = [
        test_timeout_configuration,
        test_environment_variable_override,
        show_timeout_scenarios,
        show_configuration_recommendations,
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
        print("🎉 All Celery timeout configuration tests passed!")
        print("\n💡 The system now supports configurable task timeouts with defaults")
        print("   suitable for long-running tasks with indefinite retries.")
        return True
    else:
        print("❌ Some tests failed. Please check the configuration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
