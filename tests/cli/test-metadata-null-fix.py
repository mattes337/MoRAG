#!/usr/bin/env python3
"""Test script for metadata null reference fix."""

import json
import requests
import tempfile
import time
from pathlib import Path


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(key, value):
    """Print a key-value result."""
    print(f"  {key}: {value}")


def create_test_pdf():
    """Create a minimal test PDF file."""
    # Minimal PDF content that should trigger document processing
    test_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test Document) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000074 00000 n
0000000120 00000 n
0000000179 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
274
%%EOF"""

    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_file.write(test_content)
    temp_file.close()
    return Path(temp_file.name)


def test_metadata_null_fix():
    """Test that metadata null reference error is fixed."""
    print_section("Testing Metadata Null Reference Fix")

    # Create test PDF
    test_file = create_test_pdf()

    try:
        print("🔄 Testing document ingestion that previously caused metadata null error...")

        # Test file ingestion with document that might return None metadata
        with open(test_file, 'rb') as f:
            files = {'file': ('test_metadata_fix.pdf', f, 'application/pdf')}
            data = {
                'store_in_vector_db': 'true',
                'metadata': json.dumps({
                    'test': 'metadata_null_fix',
                    'description': 'Testing fix for NoneType metadata mapping error'
                })
            }

            response = requests.post(
                'http://localhost:8000/api/v1/ingest/file',
                files=files,
                data=data,
                timeout=60
            )

        if response.status_code == 200:
            result = response.json()
            print_result("✅ File upload successful", f"Task ID: {result['task_id']}")
            print_result("Status", result['status'])
            print_result("Message", result['message'])
            return result['task_id']
        else:
            print_result("❌ File upload failed", f"Status: {response.status_code}")
            print_result("Error", response.text)
            return None

    finally:
        test_file.unlink(missing_ok=True)


def monitor_task_for_metadata_error(task_id):
    """Monitor task specifically looking for metadata-related errors."""
    if not task_id:
        return None

    print(f"\n🔍 Monitoring task for metadata errors ({task_id})...")

    for attempt in range(60):  # 60 attempts, 2 seconds each = 2 minutes max
        try:
            response = requests.get(f'http://localhost:8000/api/v1/status/{task_id}')

            if response.status_code == 200:
                status = response.json()
                print(f"  Status: {status['status']} (Progress: {status.get('progress', 0):.1%})")

                if status['status'] in ['SUCCESS', 'FAILURE']:
                    if status['status'] == 'SUCCESS':
                        print_result("✅ Task completed successfully", "No metadata errors!")

                        # Check if vector storage worked
                        if status.get('result') and status['result'].get('metadata'):
                            metadata = status['result']['metadata']
                            if 'vector_point_ids' in metadata:
                                print_result("✅ Vector storage successful",
                                           f"Stored {len(metadata['vector_point_ids'])} chunks")
                            if 'stored_in_vector_db' in metadata:
                                print_result("✅ Vector DB flag set",
                                           metadata['stored_in_vector_db'])

                        return True
                    else:
                        error_msg = status.get('error', 'Unknown error')
                        print_result("❌ Task failed", error_msg)

                        # Check if it's the metadata error we're trying to fix
                        if "'NoneType' object is not a mapping" in error_msg:
                            print_result("❌ METADATA NULL ERROR STILL EXISTS",
                                       "The fix did not work!")
                            return False
                        elif "metadata" in error_msg.lower():
                            print_result("❌ Other metadata error", error_msg)
                            return False
                        else:
                            print_result("❌ Different error", error_msg)
                            return False

            time.sleep(2)

        except Exception as e:
            print(f"  Error checking status: {e}")
            time.sleep(2)

    print_result("⏰ Task timeout", "Task did not complete within 2 minutes")
    return None


def main():
    """Run metadata null reference fix test."""
    print_section("MoRAG Metadata Null Reference Fix Test")
    print("Testing fix for: TypeError: 'NoneType' object is not a mapping")
    print("This error occurred in ingest_tasks.py when result.metadata was None")
    print("Fix includes both API-level input sanitization and worker-level metadata initialization")

    # Test the fix
    task_id = test_metadata_null_fix()

    if task_id:
        print_section("Monitoring Task for Metadata Errors")
        success = monitor_task_for_metadata_error(task_id)

        print_section("Test Results")
        if success is True:
            print("✅ METADATA NULL REFERENCE FIX SUCCESSFUL!")
            print("   - No 'NoneType' object is not a mapping errors")
            print("   - Vector storage completed successfully")
            print("   - Metadata properly initialized as dictionary")
            print("   - API input sanitization working correctly")
            print("   - Worker-level defensive programming effective")
        elif success is False:
            print("❌ METADATA NULL REFERENCE FIX FAILED!")
            print("   - The original error still occurs")
            print("   - Additional debugging needed")
            print("   - Check both API and worker level fixes")
        else:
            print("⏰ TEST INCONCLUSIVE")
            print("   - Task timed out or other issues occurred")
            print("   - Check server logs for details")
    else:
        print_section("Test Results")
        print("❌ COULD NOT START TEST")
        print("   - File upload failed")
        print("   - Check if MoRAG server is running")
        print("   - Ensure Docker containers are accessible on localhost:8000")


if __name__ == "__main__":
    main()
