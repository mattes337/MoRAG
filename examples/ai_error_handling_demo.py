"""Demonstration of AI error handling and resilience features."""

import asyncio
import time
from typing import Dict, Any
import structlog

from morag.core.ai_error_handlers import (
    execute_with_ai_resilience, 
    get_ai_service_health,
    universal_ai_handler
)
from morag.services.embedding import gemini_service
from morag.core.exceptions import RateLimitError, ExternalServiceError

# Configure logging for demo
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def demo_successful_operations():
    """Demonstrate successful operations with resilience tracking."""
    print("\n=== Demo: Successful Operations ===")
    
    try:
        # Test embedding generation
        print("Testing embedding generation...")
        result = await gemini_service.generate_embedding("Hello, world!")
        print(f"✓ Embedding generated: {len(result.embedding)} dimensions")
        
        # Test text generation
        print("Testing text generation...")
        summary = await gemini_service.generate_summary(
            "This is a test document that needs to be summarized. "
            "It contains important information that should be condensed.",
            max_length=30
        )
        print(f"✓ Summary generated: {summary.summary[:50]}...")
        
    except Exception as e:
        print(f"✗ Operation failed: {e}")
    
    # Show health metrics
    health = get_ai_service_health("gemini")
    print(f"Health metrics: {health}")


async def demo_retry_mechanism():
    """Demonstrate retry mechanism with simulated failures."""
    print("\n=== Demo: Retry Mechanism ===")
    
    call_count = 0
    
    async def flaky_operation():
        nonlocal call_count
        call_count += 1
        print(f"  Attempt {call_count}")
        
        if call_count < 3:
            raise Exception("503 Service temporarily unavailable")
        return f"Success after {call_count} attempts"
    
    try:
        result = await execute_with_ai_resilience(
            "demo_service",
            flaky_operation,
            timeout=5.0
        )
        print(f"✓ Operation succeeded: {result}")
        
    except Exception as e:
        print(f"✗ Operation failed after retries: {e}")
    
    # Show retry metrics
    health = get_ai_service_health("demo_service")
    print(f"Retry metrics: {health}")


async def demo_circuit_breaker():
    """Demonstrate circuit breaker functionality."""
    print("\n=== Demo: Circuit Breaker ===")
    
    async def always_failing_operation():
        raise Exception("500 Internal Server Error")
    
    print("Triggering circuit breaker with repeated failures...")
    
    # Trigger circuit breaker
    for i in range(6):
        try:
            await execute_with_ai_resilience(
                "circuit_demo",
                always_failing_operation,
                timeout=1.0
            )
        except Exception as e:
            print(f"  Attempt {i+1}: {type(e).__name__}")
    
    # Check circuit breaker status
    health = get_ai_service_health("circuit_demo")
    if isinstance(health, dict) and "circuit_breaker" in health:
        cb_state = health["circuit_breaker"]
        print(f"Circuit breaker state: {cb_state['state']}")
        print(f"Failure count: {cb_state['failure_count']}")
    
    # Try operation with open circuit
    print("\nTrying operation with open circuit breaker...")
    try:
        await execute_with_ai_resilience(
            "circuit_demo",
            always_failing_operation,
            timeout=1.0
        )
    except Exception as e:
        print(f"✓ Circuit breaker blocked call: {type(e).__name__}")


async def demo_fallback_mechanism():
    """Demonstrate fallback mechanism."""
    print("\n=== Demo: Fallback Mechanism ===")
    
    async def primary_operation():
        raise Exception("401 Unauthorized - Primary service unavailable")
    
    async def fallback_operation():
        return "Fallback result - using alternative service"
    
    try:
        result = await execute_with_ai_resilience(
            "fallback_demo",
            primary_operation,
            fallback=fallback_operation,
            timeout=5.0
        )
        print(f"✓ Fallback succeeded: {result}")
        
    except Exception as e:
        print(f"✗ Both primary and fallback failed: {e}")


async def demo_error_classification():
    """Demonstrate error classification."""
    print("\n=== Demo: Error Classification ===")
    
    error_scenarios = [
        ("Rate limit", lambda: (_ for _ in ()).throw(Exception("429 Too Many Requests"))),
        ("Quota exceeded", lambda: (_ for _ in ()).throw(Exception("quota exceeded"))),
        ("Authentication", lambda: (_ for _ in ()).throw(Exception("401 Unauthorized"))),
        ("Timeout", lambda: (_ for _ in ()).throw(Exception("timeout occurred"))),
        ("Content policy", lambda: (_ for _ in ()).throw(Exception("safety filter triggered"))),
    ]
    
    for error_name, error_operation in error_scenarios:
        try:
            await execute_with_ai_resilience(
                "classification_demo",
                error_operation,
                timeout=1.0
            )
        except Exception as e:
            print(f"  {error_name}: {type(e).__name__} - {str(e)[:50]}...")
    
    # Show error distribution
    health = get_ai_service_health("classification_demo")
    if isinstance(health, dict) and "error_distribution" in health:
        print(f"Error distribution: {health['error_distribution']}")


async def demo_health_monitoring():
    """Demonstrate health monitoring and metrics."""
    print("\n=== Demo: Health Monitoring ===")
    
    # Generate some activity
    for i in range(10):
        try:
            if i % 3 == 0:
                # Simulate failure
                await execute_with_ai_resilience(
                    "health_demo",
                    lambda: (_ for _ in ()).throw(Exception("Simulated failure")),
                    timeout=1.0
                )
            else:
                # Simulate success
                await execute_with_ai_resilience(
                    "health_demo",
                    lambda: f"Success {i}",
                    timeout=1.0
                )
        except Exception:
            pass
    
    # Show comprehensive health metrics
    health = get_ai_service_health("health_demo")
    if isinstance(health, dict):
        print(f"Service: {health.get('service_name', 'unknown')}")
        print(f"Success rate: {health.get('success_rate', 0):.2%}")
        print(f"Total attempts: {health.get('total_attempts', 0)}")
        print(f"Total successes: {health.get('total_successes', 0)}")
        print(f"Total failures: {health.get('total_failures', 0)}")
        print(f"Health status: {health.get('health_status', 'unknown')}")
        print(f"Average response time: {health.get('avg_response_time', 0):.3f}s")


async def demo_performance_impact():
    """Demonstrate performance impact of resilience framework."""
    print("\n=== Demo: Performance Impact ===")
    
    async def fast_operation():
        await asyncio.sleep(0.001)  # Simulate fast operation
        return "result"
    
    # Measure with resilience
    start_time = time.time()
    for _ in range(100):
        await execute_with_ai_resilience(
            "performance_demo",
            fast_operation,
            timeout=1.0
        )
    resilience_time = time.time() - start_time
    
    # Measure without resilience
    start_time = time.time()
    for _ in range(100):
        await fast_operation()
    direct_time = time.time() - start_time
    
    overhead = ((resilience_time - direct_time) / direct_time * 100) if direct_time > 0 else 0
    print(f"Direct execution: {direct_time:.3f}s")
    print(f"With resilience: {resilience_time:.3f}s")
    print(f"Overhead: {overhead:.1f}%")


async def demo_comprehensive_health_report():
    """Generate comprehensive health report for all services."""
    print("\n=== Demo: Comprehensive Health Report ===")
    
    all_health = universal_ai_handler.get_all_health_status()
    
    print("AI Services Health Report:")
    print("=" * 50)
    
    for service_name, health_data in all_health.items():
        if isinstance(health_data, dict):
            print(f"\nService: {service_name}")
            print(f"  Status: {health_data.get('health_status', 'unknown')}")
            print(f"  Success Rate: {health_data.get('success_rate', 0):.2%}")
            print(f"  Total Requests: {health_data.get('total_attempts', 0)}")
            
            if "circuit_breaker" in health_data:
                cb = health_data["circuit_breaker"]
                print(f"  Circuit Breaker: {cb.get('state', 'unknown')}")
                print(f"  Failure Count: {cb.get('failure_count', 0)}")
            
            if "error_distribution" in health_data:
                errors = health_data["error_distribution"]
                if errors:
                    print(f"  Error Types: {list(errors.keys())}")


async def main():
    """Run all demonstrations."""
    print("AI Error Handling and Resilience Framework Demo")
    print("=" * 60)
    
    demos = [
        demo_successful_operations,
        demo_retry_mechanism,
        demo_circuit_breaker,
        demo_fallback_mechanism,
        demo_error_classification,
        demo_health_monitoring,
        demo_performance_impact,
        demo_comprehensive_health_report,
    ]
    
    for demo in demos:
        try:
            await demo()
            await asyncio.sleep(0.5)  # Brief pause between demos
        except Exception as e:
            print(f"Demo failed: {e}")
            logger.exception("Demo error", demo=demo.__name__)
    
    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
