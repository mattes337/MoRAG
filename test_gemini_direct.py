#!/usr/bin/env python3
"""Test Gemini API directly to see if it's working."""

import asyncio
import os
import sys
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-embedding" / "src"))

from morag_embedding.gemini_client import GeminiClient

async def test_gemini_direct():
    """Test Gemini API directly."""
    
    print("🔍 Testing Gemini API directly")
    print("=" * 40)
    
    try:
        # Initialize Gemini client
        client = GeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_id="gemini-2.0-flash-lite"
        )
        
        print("✅ Gemini client initialized")
        
        # Simple test prompt
        simple_prompt = """
Extract facts from this German medical text:

ADHS ist eine neurologische Entwicklungsstörung. 
Die Hauptsymptome sind Unaufmerksamkeit, Hyperaktivität und Impulsivität.
ADHS betrifft etwa 5% der Kinder weltweit.
Methylphenidat ist ein häufig verwendetes Medikament zur ADHS-Behandlung.

Respond with JSON array:
[
  {
    "subject": "entity",
    "object": "target", 
    "approach": "method",
    "solution": "outcome",
    "confidence": 0.9
  }
]
"""
        
        print("🔍 Sending simple prompt to Gemini...")
        print(f"Prompt: {simple_prompt}")
        
        response = await client.generate(simple_prompt)
        
        print(f"✅ Response received: {len(response) if response else 0} characters")
        print(f"Response: {response}")
        
        if not response or response.strip() == "[]":
            print("❌ Empty or null response from Gemini!")
        else:
            print("✅ Got valid response from Gemini")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gemini_direct())
