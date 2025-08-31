#!/usr/bin/env python3
"""
Comprehensive test script for debugging markdown conversion issues.
Tests each processing stage for different input types against remote server.
"""

import requests
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, quote_plus
import argparse

# Configuration
REMOTE_SERVER = "http://morag.drydev.de:8000"
LOCAL_SERVER = "http://localhost:8000"
OUTPUT_DIR = Path("./test_outputs")

class MoRAGTester:
    """Comprehensive tester for MoRAG processing stages."""
    
    def __init__(self, server_url: str = REMOTE_SERVER):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 300  # 5 minutes timeout
        
        # Create output directory
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        print(f"🚀 MoRAG Remote Debug Tester")
        print(f"📡 Server: {self.server_url}")
        print(f"📁 Output: {OUTPUT_DIR.absolute()}")
        print("=" * 60)
    
    def test_server_health(self) -> bool:
        """Test if server is accessible."""
        print("\n🏥 Testing Server Health...")
        try:
            response = self.session.get(f"{self.server_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server accessible: {data.get('name', 'Unknown')}")
                print(f"   Version: {data.get('version', 'Unknown')}")
                print(f"   Available stages: {data.get('available_stages', [])}")
                return True
            else:
                print(f"❌ Server returned {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Server connection failed: {e}")
            return False
    
    def execute_stage(self, stage_name: str, input_files: List[str], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a single stage with given inputs."""
        url = f"{self.server_url}/api/v1/stages/{stage_name}/execute"
        
        data = {
            'input_files': json.dumps(input_files),
            'output_dir': str(OUTPUT_DIR),
            'return_content': 'true'
        }
        
        if config:
            data['config'] = json.dumps(config)
        
        print(f"📤 POST {url}")
        print(f"   Input files: {input_files}")
        print(f"   Config: {config}")
        
        try:
            response = self.session.post(url, data=data)
            print(f"📥 Response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result.get('success', False)}")
                return result
            else:
                print(f"❌ Failed: {response.text}")
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code
                }
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def download_output_files(self, session_id: str) -> List[Path]:
        """Download all output files for a session."""
        print(f"\n📥 Downloading output files for session: {session_id}")
        
        # List files
        list_url = f"{self.server_url}/api/v1/files/list"
        try:
            response = self.session.get(list_url, params={"session_id": session_id})
            if response.status_code != 200:
                print(f"❌ Failed to list files: {response.text}")
                return []
            
            files_data = response.json()
            files = files_data.get('files', [])
            print(f"📋 Found {len(files)} files")
            
            downloaded_files = []
            for file_info in files:
                filename = file_info['filename']
                download_url = f"{self.server_url}/api/v1/files/download"
                
                try:
                    response = self.session.get(
                        download_url, 
                        params={"session_id": session_id, "filename": filename}
                    )
                    
                    if response.status_code == 200:
                        local_path = OUTPUT_DIR / f"{session_id}_{filename}"
                        with open(local_path, 'wb') as f:
                            f.write(response.content)
                        print(f"✅ Downloaded: {local_path}")
                        downloaded_files.append(local_path)
                    else:
                        print(f"❌ Failed to download {filename}: {response.text}")
                        
                except Exception as e:
                    print(f"❌ Download error for {filename}: {e}")
            
            return downloaded_files
            
        except Exception as e:
            print(f"❌ Failed to list/download files: {e}")
            return []
    
    def verify_output_files(self, files: List[Path]) -> Dict[str, Any]:
        """Verify downloaded output files."""
        print(f"\n🔍 Verifying {len(files)} output files...")
        
        verification = {
            "total_files": len(files),
            "files": [],
            "issues": []
        }
        
        for file_path in files:
            file_info = {
                "path": str(file_path),
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "extension": file_path.suffix,
                "content_preview": None
            }
            
            if file_path.exists() and file_path.stat().st_size > 0:
                try:
                    # Try to read content preview
                    if file_path.suffix in ['.md', '.txt', '.json']:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            file_info["content_preview"] = content[:500] + "..." if len(content) > 500 else content
                            
                            # Check for common issues
                            if file_path.suffix == '.md' and len(content.strip()) < 50:
                                verification["issues"].append(f"Markdown file {file_path.name} seems too short")
                            
                            if "error" in content.lower() or "failed" in content.lower():
                                verification["issues"].append(f"File {file_path.name} contains error messages")
                                
                except Exception as e:
                    verification["issues"].append(f"Could not read {file_path.name}: {e}")
            else:
                verification["issues"].append(f"File {file_path.name} is empty or missing")
            
            verification["files"].append(file_info)
            print(f"📄 {file_path.name}: {file_info['size']} bytes")
        
        if verification["issues"]:
            print(f"\n⚠️  Found {len(verification['issues'])} issues:")
            for issue in verification["issues"]:
                print(f"   - {issue}")
        else:
            print("✅ All files look good!")
        
        return verification
    
    def test_input_type(self, input_type: str, test_inputs: List[str], config: Dict[str, Any] = None):
        """Test a specific input type through all processing stages."""
        print(f"\n🧪 Testing {input_type.upper()} Processing")
        print("=" * 50)
        
        results = {
            "input_type": input_type,
            "test_inputs": test_inputs,
            "stages": {},
            "overall_success": True
        }
        
        # Stage 1: Markdown Conversion
        print(f"\n📝 Stage 1: Markdown Conversion")
        stage_config = config or {}
        conversion_result = self.execute_stage("markdown-conversion", test_inputs, stage_config)
        results["stages"]["markdown-conversion"] = conversion_result
        
        if not conversion_result.get("success"):
            print(f"❌ Markdown conversion failed, stopping pipeline")
            results["overall_success"] = False
            return results
        
        # Download and verify files
        session_id = conversion_result.get("session_id")
        if session_id:
            downloaded_files = self.download_output_files(session_id)
            verification = self.verify_output_files(downloaded_files)
            results["stages"]["markdown-conversion"]["downloaded_files"] = [str(f) for f in downloaded_files]
            results["stages"]["markdown-conversion"]["verification"] = verification
        
        # Continue with next stages if markdown conversion succeeded
        # For now, we'll focus on the markdown conversion stage
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Debug MoRAG markdown conversion issues")
    parser.add_argument("--server", default=REMOTE_SERVER, help="Server URL")
    parser.add_argument("--local", action="store_true", help="Use local server")
    parser.add_argument("--type", choices=["youtube", "web", "all"], default="all", help="Test specific type")
    
    args = parser.parse_args()
    
    server_url = LOCAL_SERVER if args.local else args.server
    tester = MoRAGTester(server_url)
    
    # Test server health first
    if not tester.test_server_health():
        print("❌ Server health check failed, exiting")
        sys.exit(1)
    
    # Define test cases
    test_cases = {}
    
    if args.type in ["youtube", "all"]:
        test_cases["youtube"] = {
            "inputs": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
            "config": {
                "extract_metadata": True,
                "extract_transcript": True,
                "use_proxy": True
            }
        }
    
    if args.type in ["web", "all"]:
        test_cases["web"] = {
            "inputs": ["https://example.com"],
            "config": {
                "extract_metadata": True,
                "clean_content": True
            }
        }
    
    # Run tests
    all_results = {}
    for input_type, test_data in test_cases.items():
        try:
            result = tester.test_input_type(
                input_type, 
                test_data["inputs"], 
                test_data["config"]
            )
            all_results[input_type] = result
            
            # Save individual result
            result_file = OUTPUT_DIR / f"{input_type}_test_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Saved results to {result_file}")
            
        except Exception as e:
            print(f"❌ Test failed for {input_type}: {e}")
            all_results[input_type] = {"error": str(e)}
    
    # Save combined results
    combined_file = OUTPUT_DIR / "combined_test_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📊 Test Summary")
    print("=" * 30)
    for input_type, result in all_results.items():
        if "error" in result:
            print(f"❌ {input_type}: {result['error']}")
        else:
            success = result.get("overall_success", False)
            print(f"{'✅' if success else '❌'} {input_type}: {'Success' if success else 'Failed'}")
    
    print(f"\n💾 All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
