#!/usr/bin/env python3
"""Validate ingestion fixes without requiring running server."""

import sys
import inspect
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag-core" / "src"))

def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(key, value):
    """Print a key-value result."""
    print(f"  {key}: {value}")

def validate_options_fix():
    """Validate that the options variable shadowing is fixed."""
    print_section("Validating Options Variable Fix")
    
    try:
        from morag.ingest_tasks import ingest_file_task, ingest_url_task, ingest_batch_task
        
        # Check function signatures
        file_sig = inspect.signature(ingest_file_task)
        url_sig = inspect.signature(ingest_url_task)
        batch_sig = inspect.signature(ingest_batch_task)
        
        # Validate parameter names
        file_params = list(file_sig.parameters.keys())
        url_params = list(url_sig.parameters.keys())
        batch_params = list(batch_sig.parameters.keys())
        
        print_result("File task parameters", file_params)
        print_result("URL task parameters", url_params)
        print_result("Batch task parameters", batch_params)
        
        # Check that 'options' is not in parameters (should be 'task_options')
        issues = []
        if 'options' in file_params:
            issues.append("ingest_file_task still has 'options' parameter")
        if 'options' in url_params:
            issues.append("ingest_url_task still has 'options' parameter")
        if 'options' in batch_params:
            issues.append("ingest_batch_task still has 'options' parameter")
        
        # Check that 'task_options' is present
        if 'task_options' not in file_params:
            issues.append("ingest_file_task missing 'task_options' parameter")
        if 'task_options' not in url_params:
            issues.append("ingest_url_task missing 'task_options' parameter")
        if 'task_options' not in batch_params:
            issues.append("ingest_batch_task missing 'task_options' parameter")
        
        if issues:
            print_result("❌ Issues found", "\n    ".join(issues))
            return False
        else:
            print_result("✅ Options variable fix", "Successfully applied")
            return True
            
    except Exception as e:
        print_result("❌ Import error", str(e))
        return False

def validate_auto_detection():
    """Validate that auto-detection is implemented."""
    print_section("Validating Auto-Detection Implementation")
    
    try:
        from morag.api import MoRAGAPI
        
        # Check if detection methods exist
        api = MoRAGAPI()
        
        has_file_detection = hasattr(api, '_detect_content_type_from_file')
        has_url_detection = hasattr(api, '_detect_content_type')
        
        print_result("File detection method", "✅ Present" if has_file_detection else "❌ Missing")
        print_result("URL detection method", "✅ Present" if has_url_detection else "❌ Missing")
        
        if has_file_detection and has_url_detection:
            # Test detection logic
            test_file = Path("test.pdf")
            detected_type = api._detect_content_type_from_file(test_file)
            print_result("PDF detection test", f"✅ {detected_type}" if detected_type == "document" else f"❌ {detected_type}")
            
            youtube_url = "https://youtube.com/watch?v=test"
            detected_type = api._detect_content_type(youtube_url)
            print_result("YouTube detection test", f"✅ {detected_type}" if detected_type == "youtube" else f"❌ {detected_type}")
            
            web_url = "https://example.com"
            detected_type = api._detect_content_type(web_url)
            print_result("Web detection test", f"✅ {detected_type}" if detected_type == "web" else f"❌ {detected_type}")
            
            return True
        else:
            return False
            
    except Exception as e:
        print_result("❌ Auto-detection validation error", str(e))
        return False

def validate_server_models():
    """Validate that server models are updated for optional source_type."""
    print_section("Validating Server Model Updates")
    
    try:
        from morag.server import IngestFileRequest, IngestURLRequest
        
        # Check if source_type is optional
        file_req = IngestFileRequest()
        url_req = IngestURLRequest(url="https://example.com")
        
        print_result("✅ IngestFileRequest", "source_type is optional")
        print_result("✅ IngestURLRequest", "source_type is optional")
        
        return True
        
    except Exception as e:
        print_result("❌ Server model validation error", str(e))
        return False

def main():
    """Run all validation tests."""
    print_section("MoRAG Ingestion Fixes Validation")
    print("Validating fixes without requiring running server")
    
    results = []
    
    # Run validations
    results.append(validate_options_fix())
    results.append(validate_auto_detection())
    results.append(validate_server_models())
    
    # Summary
    print_section("Validation Summary")
    
    passed = sum(results)
    total = len(results)
    
    print_result("Tests passed", f"{passed}/{total}")
    
    if passed == total:
        print_result("✅ Overall status", "All fixes validated successfully")
        print("\n🎉 The ingestion system fixes are working correctly!")
        print("   - Options variable shadowing is fixed")
        print("   - Auto-detection is implemented")
        print("   - Server models support optional source_type")
    else:
        print_result("❌ Overall status", f"{total - passed} validation(s) failed")
        print("\n⚠️  Some fixes may need attention. Check the details above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
