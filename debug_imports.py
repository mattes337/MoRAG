#!/usr/bin/env python3
"""Debug script to identify import issues."""

import sys
import traceback

def test_import(module_name):
    """Test importing a module and report any issues."""
    try:
        print(f"Testing import: {module_name}")
        __import__(module_name)
        print(f"✓ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to import {module_name}: {e}")
        traceback.print_exc()
        return False

def main():
    """Test imports step by step."""
    modules_to_test = [
        "morag_core",
        "morag_core.models",
        "morag_core.interfaces.service",
        "morag_services",
        "morag_document",
        "morag_document.service",
        "morag_audio",
        "morag_video", 
        "morag_image",
        "morag_web",
        "morag_youtube",
        "morag_embedding",
        "morag",
        "morag.server"
    ]
    
    for module in modules_to_test:
        success = test_import(module)
        if not success:
            print(f"Stopping at failed import: {module}")
            break
        print()

if __name__ == "__main__":
    main()
