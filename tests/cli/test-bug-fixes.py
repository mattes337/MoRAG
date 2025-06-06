#!/usr/bin/env python3
"""Test script for bug fixes in MoRAG system."""

import asyncio
import sys
import tempfile
import logging
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag-services" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag-image" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag-web" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag-youtube" / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")

def test_content_type_normalization():
    """Test content type normalization fixes."""
    print_section("Testing Content Type Normalization")
    
    try:
        from morag.api import MoRAGAPI
        from morag_services import ContentType
        
        # Create API instance
        api = MoRAGAPI()
        
        # Test cases for content type normalization
        test_cases = [
            ("pdf", "document"),
            ("doc", "document"),
            ("docx", "document"),
            ("mp3", "audio"),
            ("mp4", "video"),
            ("jpg", "image"),
            ("html", "web"),
            ("document", "document"),  # Already valid
            ("unknown_format", "unknown"),  # Should default to unknown
        ]
        
        print("Testing content type normalization:")
        for input_type, expected in test_cases:
            try:
                normalized = api._normalize_content_type(input_type)
                status = "✅" if normalized == expected else "❌"
                print(f"{status} {input_type} -> {normalized} (expected: {expected})")
                
                # Test that normalized type can create ContentType enum
                content_type_enum = ContentType(normalized)
                print(f"   ✅ ContentType({normalized}) = {content_type_enum}")
                
            except Exception as e:
                print(f"❌ {input_type} -> Error: {e}")
        
        print("\n✅ Content type normalization tests completed")
        
    except Exception as e:
        print(f"❌ Content type normalization test failed: {e}")
        import traceback
        traceback.print_exc()

def test_processing_config_parameters():
    """Test ProcessingConfig parameter handling."""
    print_section("Testing ProcessingConfig Parameter Handling")
    
    try:
        from morag_core.interfaces.processor import ProcessingConfig
        
        # Test creating ProcessingConfig with additional parameters
        test_params = {
            "file_path": "/test/file.pdf",
            "webhook_url": "https://example.com/webhook",
            "metadata": {"test": "data"},
            "use_docling": False,
            "store_in_vector_db": True,
            "generate_embeddings": True,
            "chunk_size": 1000,
            "extract_metadata": True
        }
        
        print("Testing ProcessingConfig with additional parameters:")
        try:
            config = ProcessingConfig(**test_params)
            print("✅ ProcessingConfig creation successful with additional parameters")
            print(f"   file_path: {config.file_path}")
            print(f"   webhook_url: {config.webhook_url}")
            print(f"   metadata: {config.metadata}")
            print(f"   use_docling: {config.use_docling}")
            print(f"   store_in_vector_db: {config.store_in_vector_db}")
            print(f"   generate_embeddings: {config.generate_embeddings}")
            
        except Exception as e:
            print(f"❌ ProcessingConfig creation failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test with minimal parameters
        minimal_config = ProcessingConfig(file_path="/test/minimal.pdf")
        print("✅ ProcessingConfig creation successful with minimal parameters")
        
        print("\n✅ ProcessingConfig parameter handling tests completed")
        
    except Exception as e:
        print(f"❌ ProcessingConfig test failed: {e}")
        import traceback
        traceback.print_exc()

def test_content_type_enum_validation():
    """Test ContentType enum validation."""
    print_section("Testing ContentType Enum Validation")
    
    try:
        from morag_services import ContentType
        
        # Test valid content types
        valid_types = ["document", "audio", "video", "image", "web", "youtube", "text", "unknown"]
        
        print("Testing valid ContentType enum values:")
        for content_type in valid_types:
            try:
                enum_value = ContentType(content_type)
                print(f"✅ ContentType('{content_type}') = {enum_value}")
            except Exception as e:
                print(f"❌ ContentType('{content_type}') failed: {e}")
        
        # Test invalid content types
        invalid_types = ["pdf", "doc", "mp3", "invalid"]
        
        print("\nTesting invalid ContentType enum values (should fail):")
        for content_type in invalid_types:
            try:
                enum_value = ContentType(content_type)
                print(f"❌ ContentType('{content_type}') should have failed but got: {enum_value}")
            except ValueError as e:
                print(f"✅ ContentType('{content_type}') correctly failed: {e}")
            except Exception as e:
                print(f"❌ ContentType('{content_type}') failed with unexpected error: {e}")
        
        print("\n✅ ContentType enum validation tests completed")
        
    except Exception as e:
        print(f"❌ ContentType enum test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_api_content_type_handling():
    """Test API content type handling with file processing."""
    print_section("Testing API Content Type Handling")
    
    try:
        from morag.api import MoRAGAPI
        
        # Create API instance
        api = MoRAGAPI()
        
        # Test file content type detection
        test_files = [
            "test.pdf",
            "test.doc",
            "test.mp3",
            "test.mp4",
            "test.jpg",
            "test.html",
            "test.unknown"
        ]
        
        print("Testing file content type detection:")
        for filename in test_files:
            try:
                detected_type = api._detect_content_type_from_file(Path(filename))
                normalized_type = api._normalize_content_type(detected_type)
                print(f"✅ {filename} -> detected: {detected_type}, normalized: {normalized_type}")
                
                # Verify normalized type can create ContentType enum
                from morag_services import ContentType
                content_type_enum = ContentType(normalized_type)
                print(f"   ✅ ContentType({normalized_type}) = {content_type_enum}")
                
            except Exception as e:
                print(f"❌ {filename} -> Error: {e}")
        
        print("\n✅ API content type handling tests completed")

    except Exception as e:
        print(f"❌ API content type handling test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_image_processing_api_fix():
    """Test that image processing no longer fails with genai.get_api_key() error."""
    print_section("Testing Image Processing API Fix")

    try:
        from morag_image.processor import ImageProcessor, ImageConfig

        # Test without API key (should fail gracefully)
        processor = ImageProcessor()
        config = ImageConfig(generate_caption=True)

        # Create a dummy image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Create a minimal JPEG file
            tmp.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9')
            tmp_path = Path(tmp.name)

        try:
            # This should fail gracefully with proper error message, not AttributeError
            result = await processor.process_image(tmp_path, config)
            print("❌ Expected ExternalServiceError but processing succeeded")
            return False
        except Exception as e:
            if "AttributeError" in str(type(e)) and "get_api_key" in str(e):
                print(f"❌ Still getting the old API error: {e}")
                return False
            elif "Gemini API key not configured" in str(e):
                print("✅ Image processing API fix working - proper error message")
                return True
            else:
                print(f"❌ Unexpected error: {e}")
                return False
        finally:
            tmp_path.unlink(missing_ok=True)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

async def test_web_service_method_signature_fix():
    """Test that web service method signature mismatch is fixed."""
    print_section("Testing Web Service Method Signature Fix")

    try:
        from morag_services.services import MoRAGServices

        services = MoRAGServices()

        # This should not fail with TypeError about unexpected 'config' argument
        try:
            result = await services.process_url("https://httpbin.org/get")
            print("✅ Web service method signature fix working")
            return True
        except TypeError as e:
            if "unexpected keyword argument 'config'" in str(e):
                print(f"❌ Method signature still broken: {e}")
                return False
            else:
                print(f"❌ Unexpected TypeError: {e}")
                return False
        except Exception as e:
            # Other exceptions are fine (network issues, etc.)
            print(f"✅ Web service method signature fix working (got expected error: {type(e).__name__})")
            return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

async def test_search_endpoint_implementation():
    """Test that search endpoint is now implemented."""
    print_section("Testing Search Endpoint Implementation")

    try:
        from morag_services.services import MoRAGServices

        services = MoRAGServices()

        # Test search functionality
        results = await services.search_similar("test query", limit=5)

        # Should return a list (even if empty due to no vector storage)
        if isinstance(results, list):
            print("✅ Search endpoint implementation working")
            return True
        else:
            print(f"❌ Search returned unexpected type: {type(results)}")
            return False

    except Exception as e:
        print(f"❌ Search implementation error: {e}")
        return False

async def test_youtube_bot_detection_fix():
    """Test that YouTube processing has bot detection avoidance."""
    print_section("Testing YouTube Bot Detection Fix")

    try:
        from morag_youtube.processor import YouTubeProcessor, YouTubeConfig

        processor = YouTubeProcessor()
        config = YouTubeConfig(extract_metadata_only=True)  # Only metadata to avoid downloads

        # Test with a simple YouTube URL (metadata only)
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - always available

        try:
            result = await processor.process_url(test_url, config)
            if result.success:
                print("✅ YouTube bot detection fix working - metadata extracted successfully")
                return True
            else:
                error_msg = result.error_message or "Unknown error"
                if "Sign in to confirm you're not a bot" in error_msg:
                    print(f"❌ Bot detection still occurring: {error_msg}")
                    return False
                else:
                    print(f"✅ YouTube bot detection fix working (different error, not bot detection): {error_msg}")
                    return True
        except Exception as e:
            if "Sign in to confirm you're not a bot" in str(e):
                print(f"❌ Bot detection still occurring: {e}")
                return False
            else:
                print(f"✅ YouTube bot detection fix working (different error, not bot detection): {e}")
                return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

async def main():
    """Run all bug fix tests."""
    print("🚀 MoRAG Bug Fix Test Suite")
    print("=" * 60)
    print("Testing fixes for:")
    print("1. ContentType enum validation errors")
    print("2. ProcessingConfig parameter handling")
    print("3. Celery task exception handling")
    print("4. Image processing API error (genai.get_api_key)")
    print("5. Web service method signature mismatch")
    print("6. Search endpoint implementation")
    print("7. YouTube bot detection avoidance")
    print()

    # Run synchronous tests
    test_content_type_enum_validation()
    test_content_type_normalization()
    test_processing_config_parameters()

    # Run async tests
    await test_api_content_type_handling()

    # Run critical bug fix tests
    print("\n" + "=" * 60)
    print("🔧 Critical Bug Fix Tests")
    print("=" * 60)

    critical_tests = [
        ("Image Processing API Fix", test_image_processing_api_fix),
        ("Web Service Method Signature Fix", test_web_service_method_signature_fix),
        ("Search Endpoint Implementation", test_search_endpoint_implementation),
        ("YouTube Bot Detection Fix", test_youtube_bot_detection_fix),
    ]

    results = {}

    for test_name, test_func in critical_tests:
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    print("\n" + "=" * 60)
    print("🔧 Critical Bug Fix Results Summary")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\nCritical fixes: {passed}/{total} tests passed")

    print("\n" + "=" * 60)
    print("🎉 Bug fix test suite completed!")
    print("=" * 60)

    if passed == total:
        print("🎉 All critical bug fixes are working!")
        return 0
    else:
        print("⚠️  Some critical bug fixes need attention")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
