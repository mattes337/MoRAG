#!/usr/bin/env python3
"""Comprehensive test runner for all agents."""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

# Add agents to Python path
agents_dir = Path(__file__).parent.parent
sys.path.insert(0, str(agents_dir))
sys.path.insert(0, str(agents_dir.parent))  # Add parent directory for 'agents' module

# Set up test environment
os.environ["GEMINI_API_KEY"] = "test-key-for-testing"


def run_test_file(test_file: str) -> tuple[bool, str, float]:
    """Run a single test file and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_file}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run pytest on the specific file
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                test_file,
                "-v",
                "--tb=short",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            cwd=agents_dir,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {test_file} - PASSED ({duration:.2f}s)")
            return True, result.stdout, duration
        else:
            print(f"❌ {test_file} - FAILED ({duration:.2f}s)")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False, result.stderr, duration

    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {test_file} - ERROR ({duration:.2f}s): {e}")
        return False, str(e), duration


def check_agent_imports():
    """Check if all agent modules can be imported."""
    print("🔍 Checking agent imports...")

    import_tests = [
        ("Base Agent", "agents.base.agent", "BaseAgent"),
        ("Fact Extraction", "agents.extraction.fact_extraction", "FactExtractionAgent"),
        (
            "Entity Extraction",
            "agents.extraction.entity_extraction",
            "EntityExtractionAgent",
        ),
        (
            "Relation Extraction",
            "agents.extraction.relation_extraction",
            "RelationExtractionAgent",
        ),
        (
            "Keyword Extraction",
            "agents.extraction.keyword_extraction",
            "KeywordExtractionAgent",
        ),
        ("Query Analysis", "agents.analysis.query_analysis", "QueryAnalysisAgent"),
        (
            "Content Analysis",
            "agents.analysis.content_analysis",
            "ContentAnalysisAgent",
        ),
        (
            "Sentiment Analysis",
            "agents.analysis.sentiment_analysis",
            "SentimentAnalysisAgent",
        ),
        ("Topic Analysis", "agents.analysis.topic_analysis", "TopicAnalysisAgent"),
        ("Path Selection", "agents.reasoning.path_selection", "PathSelectionAgent"),
        ("Reasoning", "agents.reasoning.reasoning", "ReasoningAgent"),
        ("Decision Making", "agents.reasoning.decision_making", "DecisionMakingAgent"),
        (
            "Context Analysis",
            "agents.reasoning.context_analysis",
            "ContextAnalysisAgent",
        ),
        ("Summarization", "agents.generation.summarization", "SummarizationAgent"),
        (
            "Response Generation",
            "agents.generation.response_generation",
            "ResponseGenerationAgent",
        ),
        ("Explanation", "agents.generation.explanation", "ExplanationAgent"),
        ("Synthesis", "agents.generation.synthesis", "SynthesisAgent"),
        ("Chunking", "agents.processing.chunking", "ChunkingAgent"),
        ("Classification", "agents.processing.classification", "ClassificationAgent"),
        ("Validation", "agents.processing.validation", "ValidationAgent"),
        ("Filtering", "agents.processing.filtering", "FilteringAgent"),
    ]

    failed_imports = []

    for name, module_path, class_name in import_tests:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed_imports.append((name, str(e)))

    if failed_imports:
        print(f"\n⚠️  {len(failed_imports)} import failures detected:")
        for name, error in failed_imports:
            print(f"    - {name}: {error}")
        return False
    else:
        print(f"\n✅ All {len(import_tests)} agent imports successful!")
        return True


def main():
    """Main test execution function."""
    print("🚀 MoRAG Agents Framework - Comprehensive Test Suite")
    print("=" * 60)

    # Check imports first
    if not check_agent_imports():
        print("\n❌ Import checks failed. Cannot proceed with tests.")
        return 1

    # Define test files to run
    test_files = [
        "tests/test_basic_functionality.py",
        "tests/test_extraction_agents.py",
        "tests/test_analysis_agents.py",
        "tests/test_reasoning_agents.py",
        "tests/test_generation_agents.py",
        "tests/test_processing_agents.py",
    ]

    # Track results
    results = []
    total_start_time = time.time()

    # Run each test file
    for test_file in test_files:
        test_path = agents_dir / test_file
        if test_path.exists():
            success, output, duration = run_test_file(str(test_path))
            results.append((test_file, success, duration))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append((test_file, False, 0))

    # Calculate summary
    total_duration = time.time() - total_start_time
    passed_tests = sum(1 for _, success, _ in results if success)
    total_tests = len(results)

    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")

    for test_file, success, duration in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} {test_file:40} ({duration:.2f}s)")

    print(f"\n📈 OVERALL RESULTS:")
    print(f"   Tests Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"   Total Time: {total_duration:.2f}s")

    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! Agents framework is working correctly.")
        print(f"💡 The agentic pattern implementation is ready for production use.")
        return 0
    else:
        print(
            f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review the output above."
        )
        return 1


async def test_agent_factory():
    """Test the agent factory functionality."""
    print("\n🏭 Testing Agent Factory...")

    try:
        from agents.base.config import AgentConfig
        from agents.factory.utils import create_agent, get_agent

        # Test creating agents
        config = AgentConfig(name="test_fact_extraction")
        fact_agent = create_agent("fact_extraction", config)
        print("  ✅ Agent creation successful")

        # Test agent registry
        registered_agent = get_agent("fact_extraction")
        print("  ✅ Agent registry working")

        return True

    except Exception as e:
        print(f"  ❌ Agent factory test failed: {e}")
        return False


def test_configuration_system():
    """Test the configuration system."""
    print("\n⚙️  Testing Configuration System...")

    try:
        from agents.base.config import AgentConfig, ModelConfig, PromptConfig

        # Test basic config
        config = AgentConfig(name="test_agent")
        assert config.name == "test_agent"
        print("  ✅ Basic configuration working")

        # Test model config
        model_config = ModelConfig(temperature=0.7, max_tokens=1000)
        assert model_config.temperature == 0.7
        print("  ✅ Model configuration working")

        # Test prompt config
        prompt_config = PromptConfig(
            system_prompt="Test system prompt",
            user_prompt_template="Test user prompt: {text}",
        )
        assert "Test system prompt" in prompt_config.system_prompt
        print("  ✅ Prompt configuration working")

        return True

    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


if __name__ == "__main__":
    # Run configuration tests
    config_success = test_configuration_system()

    # Run factory tests
    factory_success = asyncio.run(test_agent_factory())

    # Run main test suite
    main_success = main()

    # Final summary
    print(f"\n{'='*60}")
    print("🎯 FINAL VALIDATION")
    print(f"{'='*60}")
    print(f"Configuration System: {'✅ PASS' if config_success else '❌ FAIL'}")
    print(f"Agent Factory: {'✅ PASS' if factory_success else '❌ FAIL'}")
    print(f"Agent Tests: {'✅ PASS' if main_success == 0 else '❌ FAIL'}")

    overall_success = config_success and factory_success and (main_success == 0)

    if overall_success:
        print(f"\n🏆 AGENTS FRAMEWORK VALIDATION COMPLETE!")
        print(f"✅ All systems operational and ready for production")
    else:
        print(f"\n⚠️  Some validation checks failed")
        print(f"❌ Please review and fix issues before production use")

    exit(0 if overall_success else 1)
