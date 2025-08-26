"""Example demonstrating the centralized LLM response parsing capabilities."""

import asyncio
from agents.base import BaseAgent, LLMResponseParser, AgentConfig
from agents.extraction.models import FactExtractionResult


class ExampleAgent(BaseAgent[FactExtractionResult]):
    """Example agent demonstrating response parsing."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="example_agent",
            description="Example agent for demonstrating response parsing"
        )
    
    async def process_llm_response(self, raw_response: str) -> dict:
        """Example of using the centralized parser methods."""
        
        # Method 1: Use the base agent's built-in parser methods
        try:
            # Parse JSON with fallback
            parsed_data = self.parse_json_response(
                response=raw_response,
                fallback_value={"entities": [], "facts": []}
            )
            print(f"✅ Parsed using base agent method: {parsed_data}")
            return parsed_data
            
        except Exception as e:
            print(f"❌ Base agent parsing failed: {e}")
        
        # Method 2: Use the static parser directly
        try:
            parsed_data = LLMResponseParser.parse_json_response(
                response=raw_response,
                fallback_value={"entities": [], "facts": []},
                context="example_parsing"
            )
            print(f"✅ Parsed using static method: {parsed_data}")
            return parsed_data
            
        except Exception as e:
            print(f"❌ Static parsing failed: {e}")
            return {}


async def demonstrate_parsing():
    """Demonstrate various parsing scenarios."""
    
    agent = ExampleAgent()
    
    # Test cases with different response formats
    test_cases = [
        # Clean JSON
        '{"entities": ["Dr. Smith", "ADHD"], "facts": ["Dr. Smith treats ADHD"]}',
        
        # JSON in markdown code block
        '''```json
        {
            "entities": ["medication", "therapy"],
            "facts": ["medication helps with symptoms"]
        }
        ```''',
        
        # JSON with extra text
        '''Here are the extracted facts:
        {"entities": ["patient", "treatment"], "facts": ["patient needs treatment"]}
        That's all I found.''',
        
        # Malformed response (will use fallback)
        'This is not valid JSON at all',
    ]
    
    print("🧪 Testing LLM Response Parsing\n")
    
    for i, test_response in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Input: {test_response[:50]}...")
        
        result = await agent.process_llm_response(test_response)
        print(f"Result: {result}")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(demonstrate_parsing())
