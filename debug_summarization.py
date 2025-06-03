#!/usr/bin/env python3
"""
Debug script to test summarization functionality and identify issues.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from morag.services.embedding import gemini_service
from morag.services.summarization import enhanced_summarization_service, SummaryConfig, SummaryStrategy
from morag.core.config import settings
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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def test_basic_summarization():
    """Test basic Gemini summarization."""
    print("\n=== Testing Basic Gemini Summarization ===")
    
    test_text = """
    Effective management of the PIN and PUK is crucial for the security and usability of a PIV smartcard. 
    The PIV standard provides mechanisms for users to change their PIN and for the PIN to be reset in case 
    it becomes blocked. The standard method for a cardholder to change their existing PIN is by using the 
    CHANGE REFERENCE DATA command (INS 24). This command allows the user to authenticate with their current 
    PIN and then set a new PIN value. The new PIN must meet the complexity requirements defined by the 
    organization's security policy. Additionally, the PIV standard includes provisions for PIN retry limits 
    and automatic blocking mechanisms to prevent brute force attacks.
    """
    
    try:
        print(f"Original text length: {len(test_text)} characters")
        print(f"Original text preview: {test_text[:100]}...")
        
        result = await gemini_service.generate_summary(
            test_text.strip(),
            max_length=100,
            style="concise"
        )
        
        print(f"\nSummary result:")
        print(f"- Length: {len(result.summary)} characters")
        print(f"- Token count: {result.token_count}")
        print(f"- Model: {result.model}")
        print(f"- Summary: {result.summary}")
        
        # Check if it's just truncated text
        if result.summary.startswith(test_text.strip()[:50]):
            print("\n⚠️  WARNING: Summary appears to be truncated original text!")
        else:
            print("\n✅ Summary appears to be properly generated")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Basic summarization failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

async def test_enhanced_summarization():
    """Test enhanced summarization service."""
    print("\n=== Testing Enhanced Summarization ===")
    
    test_text = """
    Effective management of the PIN and PUK is crucial for the security and usability of a PIV smartcard. 
    The PIV standard provides mechanisms for users to change their PIN and for the PIN to be reset in case 
    it becomes blocked. The standard method for a cardholder to change their existing PIN is by using the 
    CHANGE REFERENCE DATA command (INS 24). This command allows the user to authenticate with their current 
    PIN and then set a new PIN value. The new PIN must meet the complexity requirements defined by the 
    organization's security policy.
    """
    
    try:
        print(f"Original text length: {len(test_text)} characters")
        print(f"Original text preview: {test_text[:100]}...")
        
        config = SummaryConfig(
            strategy=SummaryStrategy.ABSTRACTIVE,
            max_length=100,
            style="concise",
            enable_refinement=True
        )
        
        result = await enhanced_summarization_service.generate_summary(
            test_text.strip(),
            config=config
        )
        
        print(f"\nEnhanced summary result:")
        print(f"- Length: {len(result.summary)} characters")
        print(f"- Token count: {result.token_count}")
        print(f"- Model: {result.model}")
        print(f"- Strategy: {result.strategy.value}")
        print(f"- Quality score: {result.quality.overall}")
        print(f"- Processing time: {result.processing_time:.2f}s")
        print(f"- Refinement iterations: {result.refinement_iterations}")
        print(f"- Summary: {result.summary}")
        
        # Check if it's just truncated text
        if result.summary.startswith(test_text.strip()[:50]):
            print("\n⚠️  WARNING: Enhanced summary appears to be truncated original text!")
        else:
            print("\n✅ Enhanced summary appears to be properly generated")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced summarization failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

async def test_gemini_health():
    """Test Gemini service health."""
    print("\n=== Testing Gemini Service Health ===")
    
    try:
        health = await gemini_service.health_check()
        print(f"Health check result: {health}")
        
        if health["status"] == "healthy":
            print("✅ Gemini service is healthy")
            return True
        else:
            print(f"❌ Gemini service is unhealthy: {health.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def check_configuration():
    """Check configuration settings."""
    print("\n=== Checking Configuration ===")
    
    print(f"Gemini API Key: {'Set' if settings.gemini_api_key else 'NOT SET'}")
    print(f"Gemini Model: {settings.gemini_model}")
    print(f"Gemini Embedding Model: {settings.gemini_embedding_model}")
    
    if not settings.gemini_api_key:
        print("❌ GEMINI_API_KEY is not set!")
        return False
    
    if not settings.gemini_api_key.startswith("AI"):
        print("❌ GEMINI_API_KEY format appears invalid!")
        return False
        
    print("✅ Configuration appears valid")
    return True

async def test_direct_text_generation():
    """Test direct text generation method."""
    print("\n=== Testing Direct Text Generation ===")

    test_prompt = """
    Create a concise summary of the following text in approximately 50 words:

    Effective management of the PIN and PUK is crucial for the security and usability of a PIV smartcard.
    The PIV standard provides mechanisms for users to change their PIN and for the PIN to be reset in case
    it becomes blocked. The standard method for a cardholder to change their existing PIN is by using the
    CHANGE REFERENCE DATA command (INS 24).

    Summary:
    """

    try:
        print(f"Prompt length: {len(test_prompt)} characters")
        print(f"Prompt preview: {test_prompt[:150]}...")

        result = await gemini_service.generate_text_from_prompt(test_prompt)

        print(f"\nDirect generation result:")
        print(f"- Length: {len(result)} characters")
        print(f"- Result: {result}")

        # Check if it's a proper response
        if result.strip() and not result.startswith("Effective management"):
            print("\n✅ Direct text generation appears to work correctly")
            return True
        else:
            print("\n⚠️  WARNING: Direct generation may not be working properly")
            return False

    except Exception as e:
        print(f"\n❌ Direct text generation failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

async def main():
    """Main debug function."""
    print("🔍 MoRAG Summarization Debug Tool")
    print("=" * 50)

    # Check configuration first
    config_ok = check_configuration()
    if not config_ok:
        print("\n❌ Configuration issues detected. Please fix before proceeding.")
        return

    # Test Gemini health
    health_ok = await test_gemini_health()
    if not health_ok:
        print("\n❌ Gemini service health check failed. Please check API key and connectivity.")
        return

    # Test direct text generation
    direct_ok = await test_direct_text_generation()

    # Test basic summarization
    basic_ok = await test_basic_summarization()

    # Test enhanced summarization
    enhanced_ok = await test_enhanced_summarization()

    print("\n" + "=" * 50)
    print("🏁 Debug Summary:")
    print(f"- Configuration: {'✅' if config_ok else '❌'}")
    print(f"- Gemini Health: {'✅' if health_ok else '❌'}")
    print(f"- Direct Text Generation: {'✅' if direct_ok else '❌'}")
    print(f"- Basic Summarization: {'✅' if basic_ok else '❌'}")
    print(f"- Enhanced Summarization: {'✅' if enhanced_ok else '❌'}")

    if not (basic_ok and enhanced_ok):
        print("\n💡 Suggestions:")
        print("1. Check that GEMINI_API_KEY is correctly set in your .env file")
        print("2. Verify your Gemini API quota and billing status")
        print("3. Check the logs above for specific error messages")
        print("4. Try running the document processing with enhanced logging enabled")
        print("5. If direct text generation works but summarization doesn't, there may be a prompt formatting issue")

if __name__ == "__main__":
    asyncio.run(main())
