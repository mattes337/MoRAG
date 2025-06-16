import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key found: {bool(api_key)}")
if api_key:
    print(f"API Key starts with: {api_key[:10]}...")
else:
    print("No API key found")

# Test basic import
try:
    from morag_graph.extraction import RelationExtractor
    print("RelationExtractor import successful")
except Exception as e:
    print(f"Import error: {e}")