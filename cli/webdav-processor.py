#!/usr/bin/env python3
"""
MoRAG WebDAV File Processor CLI Script

This script connects to a WebDAV server, processes video files in a specified folder,
and uploads the processed markdown results back to the server using MoRAG.

Requirements:
    pip install webdavclient3 argparse
    pip install -e packages/morag-core
    pip install -e packages/morag-video

Usage:
    python webdav-processor.py --url https://webdav.example.com --username user --password pass --folder /path/to/folder --extension mp4
"""

import argparse
import os
import sys
import asyncio
import tempfile
from pathlib import Path
from typing import List, Optional
from webdav3.client import Client

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path, override=True)

try:
    from morag_services import MoRAGServices, ServiceConfig
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-services")
    sys.exit(1)


class MoRAGWebDAVProcessor:
    def __init__(self, webdav_url: str, subdirectory: str, username: str, password: str, file_extension: str):
        """
        Initialize the MoRAG WebDAV processor.
        
        Args:
            webdav_url: Base WebDAV server URL (without subdirectory)
            subdirectory: Subdirectory path on the WebDAV server
            username: WebDAV username
            password: WebDAV password
            file_extension: File extension to filter (e.g., 'mp4')
        """
        self.webdav_url = webdav_url.rstrip('/')
        self.subdirectory = subdirectory.strip('/') if subdirectory else ''
        self.username = username
        self.password = password
        self.file_extension = file_extension.lower().lstrip('.')
        
        # Initialize WebDAV client with base URL (without subdirectory)
        # We'll include the subdirectory in the paths instead
        self.client = Client({
            'webdav_hostname': self.webdav_url,
            'webdav_login': username,
            'webdav_password': password,
            'webdav_timeout': 30
        })
        
        # Test connection
        try:
            self.client.list()
            print(f"✓ Successfully connected to WebDAV server: {self.webdav_url}")
            if self.subdirectory:
                # Test subdirectory access
                self.client.list(self.subdirectory)
                print(f"✓ Successfully accessed subdirectory: {self.subdirectory}")
        except Exception as e:
            print(f"✗ Failed to connect to WebDAV server: {e}")
            sys.exit(1)
        
        # Initialize MoRAG Services
        try:
            self.morag_services = MoRAGServices()
            print(f"✓ Successfully initialized MoRAG Services")
        except Exception as e:
            print(f"✗ Failed to initialize MoRAG Services: {e}")
            sys.exit(1)

    def get_all_files_recursive(self, folder_path: str) -> List[str]:
        """
        Get all files recursively from the given folder on WebDAV server.
        
        Args:
            folder_path: Path to the folder on WebDAV server
            
        Returns:
            List of file paths relative to the folder
        """
        all_files = []
        
        try:
            # Include subdirectory in the folder path
            if self.subdirectory:
                full_folder_path = f"{self.subdirectory.strip('/')}/{folder_path.strip('./')}"
            else:
                full_folder_path = folder_path.strip('./')
            
            full_folder_path = full_folder_path.strip('/')
            
            print(f"🔍 Checking if folder exists: {full_folder_path}")
            if not self.client.check(full_folder_path):
                print(f"✗ Folder {full_folder_path} does not exist on WebDAV server")
                return []
            
            print("✓ Folder exists, starting file discovery...")
            return self._discover_files_recursive(full_folder_path)
        
        except Exception as e:
            print(f"✗ Error getting files from {folder_path}: {e}")
            return []
    
    def _discover_files_recursive(self, folder_path: str) -> List[str]:
        """
        Recursively discover all files in the given folder path.
        
        Args:
            folder_path: Folder path to search
            
        Returns:
            List of file paths relative to the base folder
        """
        all_files = []
        
        try:
            def _collect_files(current_path: str):
                try:
                    # Normalize the path for WebDAV client
                    webdav_path = current_path if current_path not in ['.', './'] else '/'
                    
                    print(f"📂 Listing contents of: {webdav_path}")
                    items = self.client.list(webdav_path)
                    print(f"📋 Found {len(items)} items in {webdav_path}")
                    
                    for item in items:
                        # Skip current directory entry and parent directory
                        if item in [webdav_path, webdav_path + '/', '../', './']:
                            continue
                            
                        # Remove leading slash and construct relative path
                        item_name = item.lstrip('/')
                        
                        # Handle path construction properly
                        if current_path == '.' or current_path == './':
                            item_path = item_name
                        else:
                            item_path = current_path.rstrip('/') + '/' + item_name
                        
                        # Clean up any double slashes or './' artifacts
                        item_path = item_path.replace('//', '/').replace('/./', '/')
                        
                        # Skip if this would cause infinite recursion
                        if item_path == current_path:
                            continue
                        
                        # Debug: Print what we're examining
                        print(f"🔍 Examining item: '{item}' -> path: '{item_path}'")
                        
                        # Check if it's a directory by checking if item ends with /
                        # or if it doesn't have a file extension
                        if item.endswith('/') or '.' not in item_name:
                            # It's likely a directory - try to recurse
                            try:
                                test_path = '/' + item_path if not item_path.startswith('/') else item_path
                                self.client.list(test_path)
                                # If successful, it's a directory - recurse
                                print(f"📁 Recursing into directory: {item_path}")
                                _collect_files(item_path)
                            except Exception:
                                # If listing fails, treat as file
                                print(f"📄 Adding as file (failed directory test): {item_path}")
                                all_files.append(item_path)
                        else:
                            # It has an extension, so it's likely a file
                            print(f"📄 Adding as file (has extension): {item_path}")
                            all_files.append(item_path)
                            
                except Exception as e:
                    print(f"Warning: Could not list contents of {current_path}: {e}")
            
            _collect_files(folder_path)
            
        except Exception as e:
            print(f"✗ Error getting files from {folder_path}: {e}")
        
        print(f"✓ Found {len(all_files)} files in {folder_path}")
        return all_files

    def should_process_file(self, file_path: str, folder_path: str) -> bool:
        """
        Check if the file should be processed.
        
        Args:
            file_path: Relative path to the file
            folder_path: Base folder path on WebDAV server
            
        Returns:
            True if file should be processed, False otherwise
        """
        # Check if file has the correct extension
        if not file_path.lower().endswith(f'.{self.file_extension}'):
            return False
        
        # Check if markdown file already exists
        file_stem = Path(file_path).stem
        markdown_filename = f"{file_stem}.md"
        
        # Construct the path where the markdown file would be
        file_dir = os.path.dirname(file_path)
        if file_dir:
            markdown_path = f"{folder_path.rstrip('/')}/{file_dir}/{markdown_filename}"
        else:
            markdown_path = f"{folder_path.rstrip('/')}/{markdown_filename}"
        
        # Check if markdown file exists on WebDAV server
        try:
            if self.client.check(markdown_path):
                print(f"  ⏭️  Skipping {file_path} - markdown file already exists: {markdown_filename}")
                return False
        except Exception as e:
            print(f"  ⚠️  Warning: Could not check for existing markdown file: {e}")
        
        return True

    async def process_file(self, remote_file_path: str, folder_path: str) -> Optional[str]:
        """
        Process the file using MoRAG by downloading it locally first, then processing.
        
        Args:
            remote_file_path: Relative path to the file on WebDAV server
            folder_path: Base folder path on WebDAV server
            
        Returns:
            Path to the created markdown file (local path) or None if processing failed
        """
        # Construct full remote path using base URL + subdirectory + folder + file
        path_parts = []
        
        # Add subdirectory if configured
        if self.subdirectory:
            path_parts.append(self.subdirectory.strip('/'))
        
        # Add folder path if not current directory
        if folder_path and folder_path not in ['.', './']:
            path_parts.append(folder_path.strip('/'))
        
        # Add the file path
        path_parts.append(remote_file_path.strip('/'))
        
        # The remote_file_path already includes the subdirectory from discovery
        # So we can use it directly
        webdav_remote_path = remote_file_path
        
        # Create local temporary file path
        local_filename = os.path.basename(remote_file_path)
        
        # Use a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            local_temp_path = Path(temp_dir) / local_filename
            
            try:
                # Download file locally using direct HTTP approach
                print(f"  ⬇️  Downloading {webdav_remote_path}...")
                
                # Use direct HTTP request to bypass webdavclient3 path issues
                import requests
                from requests.auth import HTTPBasicAuth
                
                # Construct the direct URL
                file_url = f"{self.webdav_url}/{webdav_remote_path}"
                
                # Make direct HTTP request
                response = requests.get(
                    file_url,
                    auth=HTTPBasicAuth(self.username, self.password),
                    verify=False,
                    stream=True,
                    timeout=30
                )
                
                if response.status_code == 200:
                    with open(local_temp_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print(f"  ✓ Downloaded to {local_temp_path}")
                else:
                    raise Exception(f"HTTP download failed with status code: {response.status_code}")
                
                # Process the file using MoRAG Services
                print(f"  🔄 Processing {remote_file_path} with MoRAG...")
                
                # Process the file using appropriate MoRAG service based on file extension
                processing_result = await self.morag_services.process_content(
                    path_or_url=str(local_temp_path)
                )
                
                if not processing_result.success:
                    print(f"  ✗ MoRAG processing failed for {remote_file_path}: {processing_result.error_message}")
                    return None
                
                print(f"  ✓ MoRAG processing completed for {remote_file_path}")
                
                # Create markdown file from processing result
                markdown_filename = Path(remote_file_path).stem + ".md"
                markdown_file_path = Path(temp_dir) / markdown_filename
                
                # Write the extracted text content to markdown file
                with open(markdown_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {Path(remote_file_path).name}\n\n")
                    if processing_result.metadata:
                        f.write("## Metadata\n\n")
                        for key, value in processing_result.metadata.items():
                            f.write(f"- **{key}**: {value}\n")
                        f.write("\n")
                    f.write("## Content\n\n")
                    f.write(processing_result.text_content)
                
                if not markdown_file_path.exists():
                    print(f"  ✗ Failed to create markdown file for {remote_file_path}")
                    return None
                
                print(f"  ✓ Generated markdown file: {markdown_file_path.name}")
                return str(markdown_file_path)
                
            except Exception as e:
                print(f"  ✗ Error processing {remote_file_path}: {e}")
                return None

    def upload_processed_files(self, original_file_path: str, processed_files: List[str], folder_path: str):
        """
        Upload processed files back to WebDAV server alongside the original file.
        
        Args:
            original_file_path: Relative path of the original file on WebDAV server
            processed_files: List of local paths to processed files
            folder_path: Base folder path on WebDAV server
        """
        if not processed_files:
            return
        
        # Ensure folder path ends with /
        if not folder_path.endswith('/'):
            folder_path += '/'
            
        # Get the directory of the original file
        original_dir = os.path.dirname(original_file_path)
        
        for local_file in processed_files:
            try:
                # Get just the filename for upload
                filename = os.path.basename(local_file)
                
                # Construct remote path including subdirectory
                path_parts = []
                if self.subdirectory:
                    path_parts.append(self.subdirectory.strip('/'))
                if folder_path and folder_path not in ['.', './']:
                    path_parts.append(folder_path.strip('/'))
                if original_dir:
                    path_parts.append(original_dir.strip('/'))
                path_parts.append(filename)
                
                remote_path = '/'.join(path_parts)
                
                # Upload the file
                self.client.upload_sync(remote_path=remote_path, local_path=local_file)
                print(f"  ✓ Uploaded {filename} to {remote_path}")
                
            except Exception as e:
                print(f"  ✗ Failed to upload {local_file}: {e}")

    async def process_folder(self, folder_path: str):
        """
        Main processing function that handles the entire workflow.
        
        Args:
            folder_path: Path to the folder on WebDAV server to process
        """
        print(f"\n🚀 Starting MoRAG processing of folder: {folder_path}")
        print(f"📁 Processing files with extension: .{self.file_extension}")
        
        # Check if the file extension is supported by detecting content type
        test_filename = f"test.{self.file_extension}"
        content_type = self.morag_services.detect_content_type(test_filename)
        
        if content_type == "unknown":
            print(f"✗ File extension '.{self.file_extension}' is not supported by MoRAG Services")
            print("Supported extensions:")
            print("  Video: mp4, avi, mov, mkv, wmv, flv, webm, m4v")
            print("  Audio: mp3, wav, m4a, flac, aac, ogg")
            print("  Image: jpg, jpeg, png, gif, bmp, webp, tiff, svg")
            print("  Document: pdf, docx, xlsx, pptx, txt, md, html, csv, json, xml")
            return
        
        # Get all files recursively
        all_files = self.get_all_files_recursive(folder_path)
        
        if not all_files:
            print("No files found to process.")
            return
        
        # Ensure folder path ends with /
        if not folder_path.endswith('/'):
            folder_path += '/'
        
        processed_count = 0
        skipped_count = 0
        
        # Process each file
        for file_path in all_files:
            print(f"\n📁 Checking file: {file_path}")
            
            # Check if file should be processed
            if not self.should_process_file(file_path, folder_path):
                if not file_path.lower().endswith(f'.{self.file_extension}'):
                    print(f"  ⏭️  Skipping {file_path} - wrong extension (looking for .{self.file_extension})")
                skipped_count += 1
                continue
            
            try:
                # Process the file using MoRAG
                processed_file = await self.process_file(file_path, folder_path)
                
                if processed_file:
                    # Upload processed file back to WebDAV
                    self.upload_processed_files(file_path, [processed_file], folder_path)
                    processed_count += 1
                else:
                    print(f"  ✗ Failed to process {file_path}")
                
            except Exception as e:
                print(f"  ✗ Error processing {file_path}: {e}")
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Processed: {processed_count} files")
        print(f"📊 Skipped: {skipped_count} files")
        print(f"📊 Total files found: {len(all_files)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process video files in a WebDAV folder using MoRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python webdav-processor.py --url https://webdav.example.com --subdirectory /path --username user --password pass --folder /documents --extension mp4
  python webdav-processor.py -u https://cloud.example.com/webdav -s /shared -U myuser -P mypass -f /projects -e avi
        """
    )
    
    parser.add_argument(
        '--url', '-u',
        required=True,
        help='Base WebDAV server URL (without subdirectory) (e.g., https://webdav.example.com)'
    )
    
    parser.add_argument(
        '--subdirectory', '-s',
        default='',
        help='Subdirectory path on the WebDAV server (e.g., /path/to/subdir)'
    )
    
    parser.add_argument(
        '--username', '-U',
        required=True,
        help='WebDAV username'
    )
    
    parser.add_argument(
        '--password', '-P',
        required=True,
        help='WebDAV password'
    )
    
    parser.add_argument(
        '--folder', '-f',
        required=True,
        help='Folder path on WebDAV server to process (e.g., /documents)'
    )
    
    parser.add_argument(
        '--extension', '-e',
        required=True,
        help='File extension to process (e.g., mp4, avi, mov)'
    )
    
    args = parser.parse_args()
    
    # Validate file extension early (before WebDAV connection)
    try:
        temp_services = MoRAGServices()
        test_filename = f"test.{args.extension.lower().lstrip('.')}"
        content_type = temp_services.detect_content_type(test_filename)
        
        if content_type == "unknown":
            print(f"✗ File extension '.{args.extension}' is not supported by MoRAG Services")
            print("Supported extensions:")
            print("  Video: mp4, avi, mov, mkv, wmv, flv, webm, m4v")
            print("  Audio: mp3, wav, m4a, flac, aac, ogg")
            print("  Image: jpg, jpeg, png, gif, bmp, webp, tiff, svg")
            print("  Document: pdf, docx, xlsx, pptx, txt, md, html, csv, json, xml")
            sys.exit(1)
        else:
            print(f"✓ File extension '.{args.extension}' is supported (detected as {content_type})")
    except Exception as e:
        print(f"✗ Failed to validate file extension: {e}")
        sys.exit(1)
    
    # Initialize processor and run
    processor = MoRAGWebDAVProcessor(args.url, args.subdirectory, args.username, args.password, args.extension)
    
    # Run the async processing
    asyncio.run(processor.process_folder(args.folder))


if __name__ == '__main__':
    main()