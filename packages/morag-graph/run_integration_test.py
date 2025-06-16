#!/usr/bin/env python3
"""Script to run integration tests with environment variables loaded."""

import os
import subprocess
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if GEMINI_API_KEY is loaded
if not os.getenv('GEMINI_API_KEY'):
    print("Error: GEMINI_API_KEY not found in environment")
    sys.exit(1)

print(f"GEMINI_API_KEY loaded: {os.getenv('GEMINI_API_KEY')[:10]}...")

# Run the test
result = subprocess.run([
    sys.executable, '-m', 'pytest', 
    'tests/test_integration.py::test_entity_extraction_integration',
    '-v', '-s', '--tb=long'
])

sys.exit(result.returncode)