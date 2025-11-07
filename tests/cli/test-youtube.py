#!/usr/bin/env python3
"""
YouTube processing test script for remote server debugging.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import requests

# Configuration
REMOTE_SERVER = "http://morag.drydev.de:8000"
LOCAL_SERVER = "http://localhost:8000"
OUTPUT_DIR = Path("./test_outputs")


def test_youtube_processing(
    server_url: str, youtube_url: str, config: Dict[str, Any] = None
):
    """Test YouTube URL processing."""
    print(f"🎥 Testing YouTube Processing")
    print(f"📡 Server: {server_url}")
    print(f"🔗 URL: {youtube_url}")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Default config for YouTube
    default_config = {
        "extract_metadata": True,
        "extract_transcript": True,
        "use_proxy": True,
        "download_audio": False,  # Skip audio download for faster testing
        "extract_chapters": True,
    }

    if config:
        default_config.update(config)

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

    # Execute markdown conversion stage
    print(f"\n📝 Executing markdown-conversion stage...")
    stage_url = f"{server_url}/api/v1/stages/markdown-conversion/execute"

    data = {
        "input_files": json.dumps([youtube_url]),
        "config": json.dumps(default_config),
        "output_dir": str(OUTPUT_DIR),
        "return_content": "true",
    }

    print(f"📤 POST {stage_url}")
    print(f"   Input: {youtube_url}")
    print(f"   Config: {json.dumps(default_config, indent=2)}")

    try:
        response = requests.post(stage_url, data=data, timeout=300)
        print(f"📥 Response: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success', False)}")

            # Save result
            result_file = OUTPUT_DIR / "youtube_test_result.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"💾 Result saved to {result_file}")

            # Download output files if session_id is available
            session_id = result.get("session_id")
            if session_id:
                download_files(server_url, session_id)

            # Print summary
            print(f"\n📊 Processing Summary:")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"   Session ID: {session_id}")

            if result.get("output_files"):
                print(f"   Output files: {len(result['output_files'])}")
                for file_info in result["output_files"]:
                    print(f"     - {file_info.get('filename', 'unknown')}")

            if result.get("error_message"):
                print(f"   Error: {result['error_message']}")

            return result.get("success", False)

        else:
            print(f"❌ Failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def download_files(server_url: str, session_id: str):
    """Download output files for a session."""
    print(f"\n📥 Downloading files for session: {session_id}")

    # List files
    list_url = f"{server_url}/api/v1/files/list"
    try:
        response = requests.get(list_url, params={"session_id": session_id}, timeout=30)
        if response.status_code != 200:
            print(f"❌ Failed to list files: {response.text}")
            return

        files_data = response.json()
        files = files_data.get("files", [])
        print(f"📋 Found {len(files)} files")

        for file_info in files:
            filename = file_info["filename"]
            download_url = f"{server_url}/api/v1/files/download"

            try:
                response = requests.get(
                    download_url,
                    params={"session_id": session_id, "filename": filename},
                    timeout=60,
                )

                if response.status_code == 200:
                    local_path = OUTPUT_DIR / f"youtube_{filename}"
                    with open(local_path, "wb") as f:
                        f.write(response.content)
                    print(f"✅ Downloaded: {local_path} ({len(response.content)} bytes)")

                    # Verify content if it's a text file
                    if filename.endswith((".md", ".txt", ".json")):
                        verify_text_file(local_path)

                else:
                    print(f"❌ Failed to download {filename}: {response.text}")

            except Exception as e:
                print(f"❌ Download error for {filename}: {e}")

    except Exception as e:
        print(f"❌ Failed to list files: {e}")


def verify_text_file(file_path: Path):
    """Verify a text file and show preview."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"📄 {file_path.name}:")
        print(f"   Size: {len(content)} characters")

        if len(content.strip()) == 0:
            print(f"   ⚠️  File is empty!")
        elif len(content.strip()) < 100:
            print(f"   ⚠️  File seems very short")

        # Show preview
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"   Preview: {repr(preview)}")

        # Check for common issues
        if "error" in content.lower() or "failed" in content.lower():
            print(f"   ⚠️  Content contains error messages")

    except Exception as e:
        print(f"   ❌ Could not read file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test YouTube processing")
    parser.add_argument(
        "url",
        nargs="?",
        default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="YouTube URL to test",
    )
    parser.add_argument("--server", default=REMOTE_SERVER, help="Server URL")
    parser.add_argument("--local", action="store_true", help="Use local server")
    parser.add_argument("--config", help="JSON config string")

    args = parser.parse_args()

    server_url = LOCAL_SERVER if args.local else args.server
    config = json.loads(args.config) if args.config else None

    success = test_youtube_processing(server_url, args.url, config)

    if success:
        print(f"\n✅ YouTube processing test completed successfully!")
        print(f"📁 Check {OUTPUT_DIR} for output files")
    else:
        print(f"\n❌ YouTube processing test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
