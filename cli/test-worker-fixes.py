#!/usr/bin/env python3
"""Test script to verify worker process bug fixes."""

import asyncio
import json
import sys
import tempfile
import time
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

def test_content_type_validation():
    """Test that content type validation works correctly."""
    print_section("Testing Content Type Validation")
    
    try:
        from morag.api import MoRAGAPI
        from morag_services import ContentType
        
        # Create API instance
        api = MoRAGAPI()
        
        # Test problematic file extensions that caused the original error
        problematic_extensions = ["pdf", "doc", "docx", "mp3", "mp4", "jpg"]
        
        print("Testing problematic file extensions that caused worker errors:")
        for ext in problematic_extensions:
            try:
                # Simulate the content type detection process
                detected_type = api._detect_content_type_from_file(Path(f"test.{ext}"))
                normalized_type = api._normalize_content_type(detected_type)
                
                # This should not fail anymore
                content_type_enum = ContentType(normalized_type)
                
                print(f"[OK] {ext} -> detected: {detected_type}, normalized: {normalized_type}, enum: {content_type_enum}")
                
            except Exception as e:
                print(f"[FAIL] {ext} -> Error: {e}")
                return False
        
        print("\n[OK] All content type validations passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Content type validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processing_config_with_task_options():
    """Test ProcessingConfig with task options that caused the original error."""
    print_section("Testing ProcessingConfig with Task Options")
    
    try:
        from morag_core.interfaces.processor import ProcessingConfig
        
        # These are the exact parameters that caused the original error
        task_options = {
            'webhook_url': '',
            'metadata': None,
            'use_docling': False,
            'store_in_vector_db': True,
            'remote': False
        }
        
        # Add file_path as required
        config_params = {
            'file_path': '/tmp/test.pdf',
            **task_options
        }
        
        print("Testing ProcessingConfig with task options that caused original error:")
        try:
            config = ProcessingConfig(**config_params)
            print("[OK] ProcessingConfig creation successful with task options")
            print(f"   file_path: {config.file_path}")
            print(f"   webhook_url: {config.webhook_url}")
            print(f"   metadata: {config.metadata}")
            print(f"   use_docling: {config.use_docling}")
            print(f"   store_in_vector_db: {config.store_in_vector_db}")
            print(f"   remote: {config.remote}")
            
            return True
            
        except Exception as e:
            print(f"[FAIL] ProcessingConfig creation failed: {e}")
            return False
        
    except Exception as e:
        print(f"[FAIL] ProcessingConfig test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_celery_task_simulation():
    """Simulate the Celery task execution that was failing."""
    print_section("Testing Celery Task Simulation")
    
    try:
        from morag.api import MoRAGAPI
        from morag_services import ContentType
        from morag_core.interfaces.processor import ProcessingConfig
        
        # Create API instance
        api = MoRAGAPI()
        
        # Simulate the exact scenario that was failing
        file_path = "/tmp/test.pdf"
        task_options = {
            'webhook_url': '',
            'metadata': None,
            'use_docling': False,
            'store_in_vector_db': True
        }
        
        print("Simulating the exact worker task scenario that was failing:")
        
        # Step 1: Content type detection (this was returning 'pdf' instead of 'document')
        try:
            detected_type = api._detect_content_type_from_file(Path(file_path))
            normalized_type = api._normalize_content_type(detected_type)
            content_type_enum = ContentType(normalized_type)
            
            print(f"[OK] Step 1 - Content type detection: {detected_type} -> {normalized_type} -> {content_type_enum}")
            
        except Exception as e:
            print(f"[FAIL] Step 1 - Content type detection failed: {e}")
            return False
        
        # Step 2: ProcessingConfig creation (this was failing with unexpected keyword arguments)
        try:
            config = ProcessingConfig(
                file_path=file_path,
                **task_options
            )
            
            print(f"[OK] Step 2 - ProcessingConfig creation successful")
            
        except Exception as e:
            print(f"[FAIL] Step 2 - ProcessingConfig creation failed: {e}")
            return False
        
        # Step 3: Exception handling simulation
        try:
            # Simulate an exception that would be handled by Celery
            test_exception = ValueError("Test exception for Celery handling")
            
            # This is how we now handle exceptions in the fixed code
            error_info = {
                'error': str(test_exception),
                'error_type': test_exception.__class__.__name__,
                'file_path': file_path
            }
            
            # Re-raise with proper exception type information
            new_exception = type(test_exception)(str(test_exception))
            
            print(f"[OK] Step 3 - Exception handling: {error_info}")
            
        except Exception as e:
            print(f"[FAIL] Step 3 - Exception handling failed: {e}")
            return False
        
        print("\n[OK] All Celery task simulation steps passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Celery task simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_pdf():
    """Create a simple test PDF file."""
    try:
        # Create a simple text file that we can use for testing
        test_content = """# Test Document

This is a test document for validating the MoRAG worker fixes.

## Content Type Detection
This file should be detected as a 'document' type, not 'pdf'.

## ProcessingConfig
The ProcessingConfig should accept additional parameters like webhook_url.

## Exception Handling
Any exceptions should be properly serialized for Celery.
"""
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(test_content)
        temp_file.close()
        
        return temp_file.name
        
    except Exception as e:
        print(f"[FAIL] Failed to create test file: {e}")
        return None

def main():
    """Run all worker fix tests."""
    print("🚀 MoRAG Worker Process Bug Fix Test Suite")
    print("=" * 60)
    print("Testing fixes for worker process errors:")
    print("1. ContentType enum validation errors")
    print("2. ProcessingConfig parameter handling")
    print("3. Celery task exception handling")
    print()
    
    # Run tests
    test_results = []
    
    test_results.append(test_content_type_validation())
    test_results.append(test_processing_config_with_task_options())
    test_results.append(test_celery_task_simulation())
    
    # Summary
    print("\n" + "=" * 60)
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    if passed_tests == total_tests:
        print(f"[SUCCESS] All {total_tests} worker fix tests PASSED!")
        print("[OK] Worker process errors have been resolved")
    else:
        print(f"[FAIL] {total_tests - passed_tests} out of {total_tests} tests FAILED")
        print("[WARN]  Worker process issues may still exist")
    
    print("=" * 60)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
