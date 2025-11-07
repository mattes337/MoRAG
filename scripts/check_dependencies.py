#!/usr/bin/env python3
"""Check and install missing dependencies for MoRAG transcription features."""

import subprocess
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_dependency(module_name, package_name=None, description=""):
    """Check if a dependency is available."""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✅ {description or module_name} - Available")
        return True
    except ImportError:
        print(f"❌ {description or module_name} - Missing (install with: pip install {package_name})")
        return False

def check_pyannote_audio():
    """Check pyannote.audio specifically."""
    try:
        import pyannote.audio
        print("✅ pyannote.audio - Available")
        return True
    except ImportError:
        print("❌ pyannote.audio - Missing")
        print("   Install with: pip install pyannote.audio")
        print("   Note: Requires authentication token from Hugging Face")
        return False

def check_playwright():
    """Check Playwright specifically."""
    try:
        import playwright
        print("✅ Playwright - Available")

        # Check if browsers are installed
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                # Try to get chromium
                browser = p.chromium.launch(headless=True)
                browser.close()
                print("✅ Playwright browsers - Available")
                return True
        except Exception as e:
            print(f"⚠️  Playwright browsers - Not installed (run: playwright install)")
            return False
    except ImportError:
        print("❌ Playwright - Missing (install with: pip install playwright)")
        return False

def main():
    """Check all dependencies."""
    print("🔍 Checking MoRAG Transcription Dependencies")
    print("=" * 50)

    dependencies = [
        # Core transcription
        ("whisper", "openai-whisper", "Whisper (Speech-to-Text)"),
        ("faster_whisper", "faster-whisper", "Faster Whisper"),

        # Audio processing
        ("librosa", "librosa", "Librosa (Audio Processing)"),
        ("soundfile", "soundfile", "SoundFile (Audio I/O)"),
        ("pydub", "pydub", "PyDub (Audio Manipulation)"),

        # Speaker diarization
        ("pyannote.audio", None, "pyannote.audio (Speaker Diarization)"),

        # Topic segmentation
        ("sentence_transformers", "sentence-transformers", "Sentence Transformers"),
        ("sklearn", "scikit-learn", "Scikit-learn"),
        ("nltk", "nltk", "NLTK"),
        ("spacy", "spacy", "spaCy"),

        # Web scraping
        ("trafilatura", "trafilatura", "Trafilatura (Content Extraction)"),
        ("readability", "readability", "Readability (Content Cleaning)"),
        ("newspaper3k", "newspaper3k", "Newspaper3k (Article Extraction)"),

        # OCR and image processing
        ("pytesseract", "pytesseract", "PyTesseract (OCR)"),
        ("easyocr", "easyocr", "EasyOCR"),
        ("cv2", "opencv-python", "OpenCV"),
        ("PIL", "Pillow", "Pillow (Image Processing)"),
    ]

    missing_deps = []

    for module, package, description in dependencies:
        if module == "pyannote.audio":
            if not check_pyannote_audio():
                missing_deps.append(package or module)
        else:
            if not check_dependency(module, package, description):
                missing_deps.append(package or module)

    # Check Playwright separately
    if not check_playwright():
        missing_deps.append("playwright")

    print("\n" + "=" * 50)

    if missing_deps:
        print(f"❌ {len(missing_deps)} dependencies are missing")
        print("\n📦 To install missing dependencies:")
        print(f"pip install {' '.join(missing_deps)}")

        if "pyannote.audio" in missing_deps:
            print("\n🔑 For pyannote.audio, you'll also need:")
            print("1. Create account at https://huggingface.co/")
            print("2. Accept terms for pyannote models")
            print("3. Set HF_TOKEN environment variable")

        if "playwright" in missing_deps:
            print("\n🌐 For Playwright, after installation run:")
            print("playwright install")

        if "spacy" in missing_deps:
            print("\n🧠 For spaCy, after installation run:")
            print("python -m spacy download en_core_web_sm")

        print(f"\n⚠️  Some features will be disabled without these dependencies")
        return False
    else:
        print("✅ All dependencies are available!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
