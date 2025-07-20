#!/usr/bin/env python3
"""
Test the Gemini API key directly to diagnose issues.
"""

import os
from dotenv import load_dotenv

def test_gemini_api_key():
    """Test the Gemini API key directly."""
    
    # Load environment variables
    load_dotenv()
    
    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ No GEMINI_API_KEY found in environment")
        return False
    
    print(f"✅ Found Gemini API key: {gemini_api_key[:10]}...")
    print(f"📋 Full key length: {len(gemini_api_key)} characters")
    
    try:
        # Test with google-genai library (same as Graphiti uses)
        import google.genai as genai
        
        print("🔄 Testing Gemini API with google-genai library...")
        
        # Configure the client
        client = genai.Client(api_key=gemini_api_key)
        
        # Test a simple generation
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents='Say "Hello from Gemini!" in exactly those words.'
        )
        
        print("✅ API call successful!")
        print(f"📝 Response: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        
        # Try to get more details about the error
        if hasattr(e, 'response'):
            print(f"📋 Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'Unknown'}")
            print(f"📋 Response text: {e.response.text if hasattr(e.response, 'text') else 'Unknown'}")
        
        return False

def test_alternative_approach():
    """Test with requests library to get more detailed error info."""
    
    import requests
    
    # Load environment variables
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    print("\n🔄 Testing with direct HTTP request...")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": "Say hello"
            }]
        }]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"📋 HTTP Status: {response.status_code}")
        print(f"📋 Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            print("✅ Direct HTTP request successful!")
            return True
        else:
            print("❌ Direct HTTP request failed")
            return False
            
    except Exception as e:
        print(f"❌ HTTP request error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Gemini API Key")
    print("=" * 50)
    
    success1 = test_gemini_api_key()
    success2 = test_alternative_approach()
    
    if success1 or success2:
        print("\n✅ API key is working!")
    else:
        print("\n❌ API key has issues. Please check:")
        print("  1. API key is correct and not expired")
        print("  2. Generative Language API is enabled in Google Cloud Console")
        print("  3. Billing is set up for the project")
        print("  4. API key has proper permissions")
