#!/usr/bin/env python3
"""Comprehensive test for all MoRAG stages."""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add packages to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "packages" / "morag-stages" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-services" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-graph" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-document" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-embedding" / "src"))


async def test_stage_imports():
    """Test if all stage components can be imported."""
    print("\n" + "=" * 60)
    print("  Stage Import Test")
    print("=" * 60)

    try:
        from morag_stages import StageManager, StageStatus, StageType
        from morag_stages.models import StageContext

        print("✅ Stage core imports successful")

        # List available stages
        print(f"\n📋 Available stages ({len(list(StageType))}):")
        for stage in StageType:
            print(f"   • {stage.value}")

        return True
    except Exception as e:
        print(f"❌ Stage import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_stage_manager():
    """Test StageManager initialization."""
    print("\n" + "=" * 60)
    print("  Stage Manager Test")
    print("=" * 60)

    try:
        from morag_stages import StageManager, get_global_registry

        manager = StageManager()
        print("✅ StageManager initialized successfully")

        # Check registered stages via global registry
        registry = get_global_registry()
        registered = registry.get_registered_stages()
        print(f"\n📋 Registered stages ({len(registered)}):")
        for stage_type in registered:
            print(f"   • {stage_type.value}")

        return True
    except Exception as e:
        print(f"❌ StageManager initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_stage_execution():
    """Test executing a simple stage."""
    print("\n" + "=" * 60)
    print("  Stage Execution Test")
    print("=" * 60)

    try:
        from morag_stages import StageManager, StageStatus, StageType
        from morag_stages.models import StageContext

        # Create a test markdown file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.md"
            test_file.write_text(
                """# Test Document

This is a test document for MoRAG stage processing.

## Section 1
This section contains some test content.

## Section 2
This section contains more test content.
"""
            )

            print(f"📄 Created test file: {test_file}")

            # Create context
            output_dir = temp_path / "output"
            output_dir.mkdir(exist_ok=True)

            context = StageContext(
                source_path=test_file, output_dir=output_dir, config={}
            )

            # Test chunker stage (doesn't require external services)
            manager = StageManager()

            print(f"\n🔄 Testing chunker stage...")
            result = await manager.execute_stage(
                StageType.CHUNKER, [test_file], context
            )

            if result.status == StageStatus.COMPLETED:
                print(f"✅ Chunker stage completed successfully")
                print(f"   Output files: {[f.name for f in result.output_files]}")
                print(f"   Execution time: {result.metadata.execution_time:.2f}s")
                return True
            elif result.status == StageStatus.SKIPPED:
                print(f"⏭️  Chunker stage skipped (output exists)")
                return True
            else:
                print(f"❌ Chunker stage failed: {result.error_message}")
                return False

    except Exception as e:
        print(f"❌ Stage execution test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_configuration():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("  Configuration Test")
    print("=" * 60)

    try:
        from morag_core.config import Settings

        settings = Settings()
        print("✅ Configuration loaded successfully")
        print(f"   • Gemini Model: {settings.gemini_model}")
        print(f"   • Embedding Model: {settings.gemini_embedding_model}")
        print(f"   • Chunk Size: {settings.chunk_size}")
        print(f"   • Batch Size: {settings.embedding_batch_size}")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 MoRAG Stage Evaluation - Comprehensive Test Suite")
    print("=" * 60)

    tests = [
        ("Stage Imports", test_stage_imports),
        ("Stage Manager", test_stage_manager),
        ("Configuration", test_configuration),
        ("Stage Execution", test_stage_execution),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n📊 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! MoRAG stages are ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
