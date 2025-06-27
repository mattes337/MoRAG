#!/usr/bin/env python3
"""
Demo script showing the PydanticAI foundation for MoRAG.

This script demonstrates:
1. Creating a simple AI agent using the MoRAG PydanticAI foundation
2. Using structured outputs with Pydantic models
3. Error handling and retry logic
4. Agent factory patterns

To run this demo:
1. Set your GEMINI_API_KEY environment variable
2. Run: python examples/pydantic_ai_foundation_demo.py
"""

import os
import sys
import asyncio
from typing import Type, List
from pydantic import BaseModel, Field

# Add the packages directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'morag-core', 'src'))

from morag_core.ai import (
    MoRAGBaseAgent,
    AgentConfig,
    GeminiProvider,
    ProviderConfig,
    create_agent_with_config,
    ConfidenceLevel,
)


class SimpleAnalysisResult(BaseModel):
    """Result of simple text analysis."""
    
    main_topic: str = Field(description="The main topic of the text")
    sentiment: str = Field(description="Overall sentiment (positive, negative, neutral)")
    key_points: List[str] = Field(description="Key points from the text")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the analysis")
    word_count: int = Field(description="Approximate word count")


class SimpleAnalysisAgent(MoRAGBaseAgent[SimpleAnalysisResult]):
    """Simple text analysis agent for demonstration."""
    
    def get_result_type(self) -> Type[SimpleAnalysisResult]:
        return SimpleAnalysisResult
    
    def get_system_prompt(self) -> str:
        return """You are a text analysis agent. Analyze the given text and provide:
1. The main topic in a few words
2. The overall sentiment (positive, negative, or neutral)
3. 3-5 key points from the text
4. Your confidence in the analysis (0.0 to 1.0)
5. An approximate word count

Be concise and accurate in your analysis."""


async def demo_basic_agent():
    """Demonstrate basic agent functionality."""
    print("🤖 Demo 1: Basic Agent Functionality")
    print("=" * 50)
    
    # Create agent with default configuration
    agent = SimpleAnalysisAgent()
    
    # Sample text to analyze
    sample_text = """
    Artificial Intelligence is revolutionizing the way we work and live. 
    From healthcare to transportation, AI is making significant improvements 
    in efficiency and accuracy. However, there are also concerns about job 
    displacement and ethical considerations that need to be addressed. 
    The future of AI looks promising, but we must proceed thoughtfully.
    """
    
    try:
        print(f"📝 Analyzing text: {sample_text[:100]}...")
        
        # Note: This will only work if you have a valid GEMINI_API_KEY
        # For demo purposes, we'll show the structure without actually calling the API
        print("⚠️  Note: This would call the Gemini API if GEMINI_API_KEY is set")
        
        # Show what the agent would do
        print(f"🎯 Agent class: {agent.__class__.__name__}")
        print(f"📋 Result type: {agent.get_result_type().__name__}")
        print(f"💭 System prompt: {agent.get_system_prompt()[:100]}...")
        print(f"🔧 Model: {agent.config.model}")
        print(f"⏱️  Timeout: {agent.config.timeout}s")
        print(f"🔄 Max retries: {agent.config.max_retries}")
        
        # If API key is available, try to run the agent
        if os.getenv("GEMINI_API_KEY"):
            print("\n🚀 Running agent with Gemini API...")
            result = await agent.run(f"Analyze this text: {sample_text}")
            
            print("\n✅ Analysis Result:")
            print(f"📌 Main topic: {result.main_topic}")
            print(f"😊 Sentiment: {result.sentiment}")
            print(f"🔑 Key points:")
            for i, point in enumerate(result.key_points, 1):
                print(f"   {i}. {point}")
            print(f"🎯 Confidence: {result.confidence:.2f}")
            print(f"📊 Word count: {result.word_count}")
        else:
            print("\n⚠️  Set GEMINI_API_KEY environment variable to run the actual analysis")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50 + "\n")


def demo_agent_factory():
    """Demonstrate agent factory patterns."""
    print("🏭 Demo 2: Agent Factory Patterns")
    print("=" * 50)
    
    # Create agent with custom configuration using factory
    agent = create_agent_with_config(
        SimpleAnalysisAgent,
        model="google-gla:gemini-1.5-pro",
        timeout=60,
        temperature=0.2,
        max_retries=5
    )
    
    print(f"🎯 Created agent: {agent.__class__.__name__}")
    print(f"🔧 Model: {agent.config.model}")
    print(f"⏱️  Timeout: {agent.config.timeout}s")
    print(f"🌡️  Temperature: {agent.config.temperature}")
    print(f"🔄 Max retries: {agent.config.max_retries}")
    
    # Show provider information
    provider_info = agent.provider.get_provider_info()
    print(f"🔌 Provider: {provider_info['name']}")
    print(f"✅ Available: {provider_info['available']}")
    print(f"🔑 API key configured: {provider_info['api_key_configured']}")
    
    print("\n" + "=" * 50 + "\n")


def demo_configuration():
    """Demonstrate configuration options."""
    print("⚙️  Demo 3: Configuration Options")
    print("=" * 50)
    
    # Create custom provider configuration
    provider_config = ProviderConfig(
        api_key=os.getenv("GEMINI_API_KEY", "not-set"),
        timeout=45,
        max_retries=3
    )
    
    # Create custom agent configuration
    agent_config = AgentConfig(
        model="google-gla:gemini-1.5-flash",
        timeout=30,
        max_retries=3,
        temperature=0.1,
        max_tokens=1000,
        provider_config=provider_config
    )
    
    print("📋 Agent Configuration:")
    print(f"   Model: {agent_config.model}")
    print(f"   Timeout: {agent_config.timeout}s")
    print(f"   Max retries: {agent_config.max_retries}")
    print(f"   Temperature: {agent_config.temperature}")
    print(f"   Max tokens: {agent_config.max_tokens}")
    
    print("\n📋 Provider Configuration:")
    print(f"   Timeout: {provider_config.timeout}s")
    print(f"   Max retries: {provider_config.max_retries}")
    print(f"   API key set: {'Yes' if provider_config.api_key != 'not-set' else 'No'}")
    
    # Create agent with custom configuration
    provider = GeminiProvider(provider_config)
    agent = SimpleAnalysisAgent(config=agent_config, provider=provider)
    
    print(f"\n🤖 Agent created with custom configuration")
    print(f"   Result type: {agent.get_result_type().__name__}")
    
    print("\n" + "=" * 50 + "\n")


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("🛡️  Demo 4: Error Handling")
    print("=" * 50)
    
    # Create agent with aggressive timeout for demonstration
    agent = create_agent_with_config(
        SimpleAnalysisAgent,
        timeout=1,  # Very short timeout
        max_retries=2
    )
    
    print(f"🤖 Created agent with short timeout: {agent.config.timeout}s")
    print(f"🔄 Max retries: {agent.config.max_retries}")
    
    # Test validation
    print("\n🧪 Testing result validation...")
    try:
        # Test with invalid data
        invalid_result = {
            "main_topic": "test",
            "sentiment": "positive", 
            "key_points": ["point1", "point2"],
            "confidence": 1.5,  # Invalid: > 1.0
            "word_count": 10
        }
        agent._validate_result(invalid_result)
        print("❌ Validation should have failed!")
    except Exception as e:
        print(f"✅ Validation correctly failed: {type(e).__name__}")
    
    # Test with valid data
    try:
        valid_result = {
            "main_topic": "test",
            "sentiment": "positive",
            "key_points": ["point1", "point2"], 
            "confidence": 0.8,
            "word_count": 10
        }
        validated = agent._validate_result(valid_result)
        print(f"✅ Validation successful: {type(validated).__name__}")
    except Exception as e:
        print(f"❌ Validation failed unexpectedly: {e}")
    
    print("\n" + "=" * 50 + "\n")


async def main():
    """Run all demos."""
    print("🚀 MoRAG PydanticAI Foundation Demo")
    print("=" * 60)
    print()
    
    # Check if API key is available
    api_key_available = bool(os.getenv("GEMINI_API_KEY"))
    if api_key_available:
        print("✅ GEMINI_API_KEY found - full functionality available")
    else:
        print("⚠️  GEMINI_API_KEY not found - running in demo mode")
    print()
    
    # Run demos
    await demo_basic_agent()
    demo_agent_factory()
    demo_configuration()
    demo_error_handling()
    
    print("🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Set GEMINI_API_KEY to test actual API calls")
    print("2. Explore the agent classes in packages/morag-core/src/morag_core/ai/")
    print("3. Create your own agents by extending MoRAGBaseAgent")


if __name__ == "__main__":
    asyncio.run(main())
