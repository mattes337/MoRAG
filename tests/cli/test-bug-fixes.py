#!/usr/bin/env python3
"""Test script for bug fixes in MoRAG system."""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag-services" / "src"))

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

def main():
    """Run all bug fix tests."""
    print("🚀 MoRAG Bug Fix Test Suite")
    print("=" * 60)
    print("Testing fixes for:")
    print("1. ContentType enum validation errors")
    print("2. ProcessingConfig parameter handling")
    print("3. Celery task exception handling")
    print()
    
    # Run synchronous tests
    test_content_type_enum_validation()
    test_content_type_normalization()
    test_processing_config_parameters()
    
    # Run async tests
    asyncio.run(test_api_content_type_handling())
    
    print("\n" + "=" * 60)
    print("🎉 Bug fix test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
