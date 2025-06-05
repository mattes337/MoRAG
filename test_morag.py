#!/usr/bin/env python3
"""Test script for MoRAG system using uploaded files."""

import asyncio
import json
import sys
from pathlib import Path
import time

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag" / "src"))

from morag import MoRAGAPI

async def test_file_processing(file_path: str):
    """Test processing a single file."""
    print(f"\n=== Testing MoRAG with {file_path} ===")
    
    api = MoRAGAPI()
    
    try:
        start_time = time.time()
        
        # Process the file
        print(f"Processing file: {file_path}")
        result = await api.process_file(file_path)
        
        processing_time = time.time() - start_time
        
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Success: {result.success}")
        
        if result.success:
            print(f"Content length: {len(result.content) if result.content else 0} characters")
            print(f"Metadata keys: {list(result.metadata.keys()) if result.metadata else []}")
            
            # Show first 500 characters of content
            if result.content:
                preview = result.content[:500]
                print(f"\nContent preview:\n{preview}")
                if len(result.content) > 500:
                    print("... (truncated)")
        else:
            print(f"Error: {result.error_message}")
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        await api.cleanup()

async def test_health_check():
    """Test the health check functionality."""
    print("\n=== Testing Health Check ===")
    
    api = MoRAGAPI()
    
    try:
        health = await api.health_check()
        print("Health check result:")
        print(json.dumps(health, indent=2))
    except Exception as e:
        print(f"Health check failed: {str(e)}")
    finally:
        await api.cleanup()

async def main():
    """Main test function."""
    print("MoRAG System Test")
    print("=" * 50)
    
    # Test health check first
    await test_health_check()
    
    # Test files to process
    test_files = [
        "uploads/test_video.mp4",
        "uploads/recording.m4a", 
        "uploads/cec42d04_Das DMSO-Handbuch_text.pdf"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            await test_file_processing(file_path)
        else:
            print(f"\nSkipping {file_path} - file not found")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    asyncio.run(main())
