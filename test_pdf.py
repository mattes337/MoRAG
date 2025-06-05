#!/usr/bin/env python3
"""Test MoRAG with a PDF file."""

import requests
import json
import time
from pathlib import Path

def test_pdf_processing():
    """Test PDF processing via REST API."""
    
    # Use absolute path
    pdf_file = Path("D:/Test/MoRAG/uploads/cec42d04_Das DMSO-Handbuch_text.pdf")
    
    if not pdf_file.exists():
        print(f"PDF file {pdf_file} not found")
        return
    
    print(f"Testing MoRAG with {pdf_file.name}")
    print("=" * 50)
    
    # Test file upload and processing
    try:
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, 'application/pdf')}
            
            print("Uploading and processing PDF file...")
            start_time = time.time()
            
            response = requests.post(
                'http://localhost:8000/process/file',
                files=files,
                timeout=120  # 2 minutes timeout
            )
            
            processing_time = time.time() - start_time
            
            print(f"Request completed in {processing_time:.2f} seconds")
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("Processing successful!")
                print(f"Success: {result.get('success', False)}")
                print(f"Content length: {len(result.get('content', ''))}")
                print(f"Metadata keys: {list(result.get('metadata', {}).keys())}")
                
                # Show preview of content
                content = result.get('content', '')
                if content:
                    preview = content[:500]
                    print(f"\nContent preview:\n{preview}")
                    if len(content) > 500:
                        print("... (truncated)")
                        
                # Show metadata
                metadata = result.get('metadata', {})
                if metadata:
                    print(f"\nMetadata preview:")
                    for key, value in list(metadata.items())[:5]:
                        print(f"  {key}: {str(value)[:100]}")
                        
            else:
                print(f"Processing failed: {response.text}")
                
    except requests.exceptions.Timeout:
        print("Request timed out - PDF processing may take longer")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_audio_processing():
    """Test audio processing via REST API."""
    
    # Use absolute path
    audio_file = Path("D:/Test/MoRAG/uploads/recording.m4a")
    
    if not audio_file.exists():
        print(f"Audio file {audio_file} not found")
        return
    
    print(f"Testing MoRAG with {audio_file.name}")
    print("=" * 50)
    
    # Test file upload and processing
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (audio_file.name, f, 'audio/m4a')}
            
            print("Uploading and processing audio file...")
            start_time = time.time()
            
            response = requests.post(
                'http://localhost:8000/process/file',
                files=files,
                timeout=180  # 3 minutes timeout
            )
            
            processing_time = time.time() - start_time
            
            print(f"Request completed in {processing_time:.2f} seconds")
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("Processing successful!")
                print(f"Success: {result.get('success', False)}")
                print(f"Content length: {len(result.get('content', ''))}")
                
                # Show preview of content
                content = result.get('content', '')
                if content:
                    preview = content[:500]
                    print(f"\nContent preview:\n{preview}")
                    if len(content) > 500:
                        print("... (truncated)")
                        
            else:
                print(f"Processing failed: {response.text}")
                
    except requests.exceptions.Timeout:
        print("Request timed out - audio processing may take longer")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("MoRAG File Processing Test")
    print("=" * 40)
    
    # Test PDF processing
    test_pdf_processing()
    print("\n" + "=" * 40 + "\n")
    
    # Test audio processing
    test_audio_processing()
