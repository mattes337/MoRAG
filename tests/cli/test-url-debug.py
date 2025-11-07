#!/usr/bin/env python3
"""
Simple URL debug test to verify URLPath is working on remote server.
"""

import argparse
import json
import sys

import requests


def test_url_handling(server_url: str, test_url: str):
    """Test URL handling with detailed debugging."""
    print(f"🔍 Testing URL Handling Debug")
    print(f"📡 Server: {server_url}")
    print(f"🔗 Test URL: {test_url}")
    print("=" * 60)

    # Test server health
    print("\n🏥 Testing server health...")
    try:
        response = requests.get(f"{server_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server accessible: {data.get('name', 'Unknown')}")
        else:
            print(f"❌ Server returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return False

    # Execute markdown conversion with detailed logging
    print(f"\n📝 Testing URL processing...")
    stage_url = f"{server_url}/api/v1/stages/markdown-conversion/execute"

    data = {
        "input_files": json.dumps([test_url]),
        "config": json.dumps(
            {
                "extract_metadata": True,
                "debug_mode": True,  # Enable debug mode if available
            }
        ),
        "output_dir": "./test_outputs",
        "return_content": "true",
    }

    print(f"📤 POST {stage_url}")
    print(f"   Input URL: {test_url}")

    try:
        response = requests.post(stage_url, data=data, timeout=60)
        print(f"📥 Response: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful: {result.get('success', False)}")

            # Check for URL corruption in the response
            if "error_message" in result:
                error_msg = result["error_message"]
                print(f"❌ Error message: {error_msg}")

                # Check for URL corruption patterns
                if "https:/" in error_msg and "https://" not in error_msg:
                    print(f"🐛 URL CORRUPTION DETECTED: https:// became https:/")
                    print(f"   This indicates URLPath is NOT being used correctly")
                    return False
                elif "http:/" in error_msg and "http://" not in error_msg:
                    print(f"🐛 URL CORRUPTION DETECTED: http:// became http:/")
                    print(f"   This indicates URLPath is NOT being used correctly")
                    return False
                else:
                    print(f"ℹ️  Error present but no URL corruption detected")

            # Check processing details
            if "processing_details" in result:
                details = result["processing_details"]
                print(f"📊 Processing details: {details}")

            # Check if URLPath was used (look for debug info)
            if "debug_info" in result:
                debug_info = result["debug_info"]
                print(f"🔧 Debug info: {debug_info}")

            return result.get("success", False)

        else:
            print(f"❌ Request failed: {response.text}")

            # Check for URL corruption in error response
            error_text = response.text
            if "https:/" in error_text and "https://" not in error_text:
                print(f"🐛 URL CORRUPTION DETECTED in error response")
                return False

            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Debug URL handling")
    parser.add_argument(
        "--server", default="http://morag.drydev.de:8000", help="Server URL"
    )
    parser.add_argument("--url", default="https://example.com", help="Test URL")

    args = parser.parse_args()

    success = test_url_handling(args.server, args.url)

    if success:
        print(f"\n✅ URL handling test completed successfully!")
        print(f"   URLPath appears to be working correctly")
    else:
        print(f"\n❌ URL handling test failed!")
        print(f"   URLPath may not be working correctly on the server")
        sys.exit(1)


if __name__ == "__main__":
    main()
