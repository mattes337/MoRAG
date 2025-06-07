#!/usr/bin/env python3
"""
Test script to verify the summarization fix works correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from morag_services.embedding import gemini_service
from morag_services.processing import enhanced_summarization_service, SummaryConfig, SummaryStrategy

async def test_summarization_fix():
    """Test that summarization produces proper summaries instead of truncated text."""
    
    # Test text from the user's example
    test_text = """Effective management of the PIN and PUK is crucial for the security and usability of a PIV smartcard. The PIV standard provides mechanisms for users to change their PIN and for the PIN to be reset in case it becomes blocked. 4.1 Changing the PIN The standard method for a cardholder to change their existing PIN is by using the CHANGE REFERENCE DATA command (INS 24). This command allows the user to authenticate with their current PIN and then set a new PIN value. The new PIN must meet the complexity requirements defined by the organization's security policy. Additionally, the PIV standard includes provisions for PIN retry limits and automatic blocking mechanisms to prevent brute force attacks. The PIN can be changed multiple times as needed, but each change must be properly authenticated. Organizations should establish clear policies for PIN management, including requirements for PIN complexity, change frequency, and user training on proper PIN handling procedures."""
    
    print("🧪 Testing Summarization Fix")
    print("=" * 50)
    print(f"Original text length: {len(test_text)} characters")
    print(f"Original text preview: {test_text[:100]}...")
    
    # Test basic summarization
    print("\n--- Testing Basic Summarization ---")
    try:
        basic_result = await gemini_service.generate_summary(
            test_text,
            max_length=50,
            style="concise"
        )
        
        print(f"Basic summary: {basic_result.summary}")
        print(f"Length: {len(basic_result.summary)} characters")
        
        # Check if it's just truncated text
        is_truncated = basic_result.summary.startswith(test_text[:50])
        print(f"Is truncated text: {'❌ YES' if is_truncated else '✅ NO'}")
        
        basic_success = not is_truncated and len(basic_result.summary) > 0
        
    except Exception as e:
        print(f"❌ Basic summarization failed: {e}")
        basic_success = False
    
    # Test enhanced summarization
    print("\n--- Testing Enhanced Summarization ---")
    try:
        config = SummaryConfig(
            strategy=SummaryStrategy.ABSTRACTIVE,
            max_length=50,
            style="concise",
            enable_refinement=False  # Disable refinement for faster testing
        )
        
        enhanced_result = await enhanced_summarization_service.generate_summary(
            test_text,
            config=config
        )
        
        print(f"Enhanced summary: {enhanced_result.summary}")
        print(f"Length: {len(enhanced_result.summary)} characters")
        print(f"Strategy: {enhanced_result.strategy.value}")
        print(f"Quality score: {enhanced_result.quality.overall:.2f}")
        
        # Check if it's just truncated text
        is_truncated = enhanced_result.summary.startswith(test_text[:50])
        print(f"Is truncated text: {'❌ YES' if is_truncated else '✅ NO'}")
        
        enhanced_success = not is_truncated and len(enhanced_result.summary) > 0
        
    except Exception as e:
        print(f"❌ Enhanced summarization failed: {e}")
        enhanced_success = False
    
    # Test direct text generation
    print("\n--- Testing Direct Text Generation ---")
    try:
        direct_prompt = f"""
        Create a concise summary of the following text in approximately 50 words:
        
        {test_text}
        
        Summary:
        """
        
        direct_result = await gemini_service.generate_text_from_prompt(direct_prompt)
        
        print(f"Direct generation result: {direct_result}")
        print(f"Length: {len(direct_result)} characters")
        
        # Check if it's just truncated text
        is_truncated = direct_result.startswith(test_text[:50])
        print(f"Is truncated text: {'❌ YES' if is_truncated else '✅ NO'}")
        
        direct_success = not is_truncated and len(direct_result) > 0
        
    except Exception as e:
        print(f"❌ Direct text generation failed: {e}")
        direct_success = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results:")
    print(f"- Basic Summarization: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"- Enhanced Summarization: {'✅ PASS' if enhanced_success else '❌ FAIL'}")
    print(f"- Direct Text Generation: {'✅ PASS' if direct_success else '❌ FAIL'}")
    
    overall_success = basic_success and enhanced_success and direct_success
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if not overall_success:
        print("\n💡 Next Steps:")
        print("1. Run the debug_summarization.py script for more detailed diagnostics")
        print("2. Check your GEMINI_API_KEY configuration")
        print("3. Verify your Gemini API quota and billing status")
        print("4. Check the application logs for detailed error messages")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(test_summarization_fix())
    sys.exit(0 if success else 1)
