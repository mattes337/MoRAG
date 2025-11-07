#!/usr/bin/env python3
"""Simple test to verify agents can be imported."""

import os
import sys
from pathlib import Path

# Add current directory and parent to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

# Set test environment
os.environ["GEMINI_API_KEY"] = "test-key"


def test_basic_imports():
    """Test basic agent imports."""
    print("🧪 Testing Basic Agent Imports")
    print("=" * 40)

    try:
        # Test base imports
        from base.agent import BaseAgent
        from base.config import AgentConfig

        print("✅ Base agent imports successful")

        # Test extraction imports
        from extraction.fact_extraction import FactExtractionAgent

        print("✅ Fact extraction agent import successful")

        # Test factory imports
        from factory.utils import create_agent

        print("✅ Factory imports successful")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_agent_creation():
    """Test creating an agent."""
    print("\n🏭 Testing Agent Creation")
    print("=" * 40)

    try:
        from base.config import AgentConfig
        from extraction.fact_extraction import FactExtractionAgent

        # Create configuration
        config = AgentConfig(name="test_fact_extraction")
        print("✅ Configuration created")

        # Create agent
        agent = FactExtractionAgent(config)
        print("✅ Agent created successfully")

        # Test configuration
        agent.update_config(agent_config={"max_facts": 10, "domain": "test"})
        print("✅ Agent configuration updated")

        return True

    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🚀 Simple Agents Import Test")
    print("=" * 50)

    # Test imports
    import_success = test_basic_imports()

    # Test agent creation
    creation_success = test_agent_creation()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Imports: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"  Creation: {'✅ PASS' if creation_success else '❌ FAIL'}")

    if import_success and creation_success:
        print("\n🎉 All tests passed! Agents framework is working.")
        return 0
    else:
        print("\n⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())
