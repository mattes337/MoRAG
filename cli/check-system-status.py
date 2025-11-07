#!/usr/bin/env python3
"""System status check script for MoRAG."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))

import structlog

# Configure logging
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
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def check_project_structure():
    """Check project structure."""
    print("\n" + "=" * 60)
    print("  Project Structure Check")
    print("=" * 60)

    required_dirs = [
        "packages/morag",
        "packages/morag-core",
        "packages/morag-document",
        "packages/morag-services",
        "packages/morag-embedding",
        "packages/morag-graph",
        "tests",
        "cli",
        "docs",
    ]

    all_good = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path}")
            all_good = False

    return all_good


def check_optional_dependencies():
    """Check optional dependencies status."""
    print("\n" + "=" * 60)
    print("  Optional Dependencies Check")
    print("=" * 60)

    try:
        from morag_core.optional_deps import optional_deps

        available = optional_deps.get_available_features()
        missing = optional_deps.get_missing_features()

        if available:
            print(f"\n✅ Available Features ({len(available)}):")
            for feature in available:
                print(f"   • {feature}")

        if missing:
            print(f"\n⚠️  Missing Features ({len(missing)}):")
            for item in missing:
                print(f"   • {item['feature']}")
                print(f"     Install: {item['install_command']}")

        print(f"\nTotal: {len(available)} available, {len(missing)} missing")
        return True

    except Exception as e:
        print(f"❌ Failed to check optional dependencies: {e}")
        return False


def check_graph_processing():
    """Check graph processing availability."""
    print("\n" + "=" * 60)
    print("  Graph Processing Check")
    print("=" * 60)

    try:
        from morag_graph import BaseExtractor, Entity, GraphStorage, Relation

        print("✅ Graph processing components available")

        # Test basic functionality - BaseExtractor is abstract, so skip instantiation
        print("✅ BaseExtractor class available")

        # Test entity creation with UUID format
        import uuid

        entity = Entity(
            id=str(uuid.uuid4()), name="Test Entity", type="test", properties={}
        )
        print("✅ Entity model working")

        relation = Relation(
            id=str(uuid.uuid4()),
            source_id=str(uuid.uuid4()),
            target_id=str(uuid.uuid4()),
            type="test_relation",
            properties={},
        )
        print("✅ Relation model working")

        return True

    except Exception as e:
        print(f"❌ Graph processing not available: {e}")
        return False


def check_core_imports():
    """Check core package imports."""
    print("\n" + "=" * 60)
    print("  Core Package Imports Check")
    print("=" * 60)

    imports_to_test = [
        ("morag_core.config", "Settings"),
        ("morag_core.exceptions", "MoRAGException"),
        ("morag_document.processor", "DocumentProcessor"),
        ("morag_services.embedding", "GeminiEmbeddingService"),
        ("morag_embedding.service", "GeminiEmbeddingService"),
    ]

    all_good = True
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            all_good = False

    return all_good


def check_configuration():
    """Check configuration loading."""
    print("\n" + "=" * 60)
    print("  Configuration Check")
    print("=" * 60)

    try:
        from morag_core.config import Settings

        settings = Settings()

        print(f"✅ Configuration loaded successfully")
        print(f"   • API Host: {settings.api_host}")
        print(f"   • API Port: {settings.api_port}")
        print(f"   • Gemini Model: {settings.gemini_model}")
        print(f"   • Embedding Model: {settings.gemini_embedding_model}")
        print(f"   • Embedding Batch Size: {settings.embedding_batch_size}")
        print(f"   • Performance Monitoring: {settings.enable_performance_monitoring}")

        return True

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def check_performance_optimizations():
    """Check performance optimization features."""
    print("\n" + "=" * 60)
    print("  Performance Optimizations Check")
    print("=" * 60)

    try:
        from morag_core.optimization import PerformanceTracker, ProcessingOptimizer
        from morag_core.performance import PerformanceMonitor

        print("✅ Performance optimization modules available")

        # Test optimization
        optimizer = ProcessingOptimizer()
        config = optimizer.get_optimal_chunk_config(0.5, 10000, "pdf")
        print(f"✅ Chunk optimization working: {config['chunk_size']} chars")

        # Test performance tracking
        tracker = PerformanceTracker()
        start_time = tracker.start_operation("test_operation")
        tracker.end_operation("test_operation", start_time)
        print("✅ Performance tracking working")

        return True

    except Exception as e:
        print(f"❌ Performance optimizations not available: {e}")
        return False


def run_basic_functionality_test():
    """Run a basic functionality test."""
    print("\n" + "=" * 60)
    print("  Basic Functionality Test")
    print("=" * 60)

    try:
        # Test document processor initialization
        from morag_document.processor import DocumentProcessor

        processor = DocumentProcessor()
        print("✅ Document processor initialized")

        # Test embedding service initialization
        from morag_core.config import Settings
        from morag_services.embedding import GeminiEmbeddingService

        settings = Settings()
        embedding_service = GeminiEmbeddingService(api_key=settings.gemini_api_key)
        print("✅ Embedding service initialized")

        print("✅ Basic functionality test passed")
        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Main function to run all checks."""
    print("🔍 MoRAG System Status Check")
    print("=" * 60)

    checks = [
        ("Project Structure", check_project_structure),
        ("Core Imports", check_core_imports),
        ("Configuration", check_configuration),
        ("Optional Dependencies", check_optional_dependencies),
        ("Graph Processing", check_graph_processing),
        ("Performance Optimizations", check_performance_optimizations),
        ("Basic Functionality", run_basic_functionality_test),
    ]

    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n❌ {check_name} check failed with exception: {e}")
            results[check_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All checks passed! MoRAG system is ready.")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
