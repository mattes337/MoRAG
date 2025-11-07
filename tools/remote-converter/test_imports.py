#!/usr/bin/env python3
"""Test imports for remote converter."""

import sys
from pathlib import Path


def test_imports():
    """Test importing MoRAG packages step by step."""

    # Add MoRAG packages to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))
    sys.path.insert(0, str(project_root / "packages" / "morag-audio" / "src"))
    sys.path.insert(0, str(project_root / "packages" / "morag-video" / "src"))
    sys.path.insert(0, str(project_root / "packages" / "morag-document" / "src"))
    sys.path.insert(0, str(project_root / "packages" / "morag-image" / "src"))
    sys.path.insert(0, str(project_root / "packages" / "morag-web" / "src"))
    sys.path.insert(0, str(project_root / "packages" / "morag-youtube" / "src"))

    print("Testing imports...")

    # Test core imports
    try:
        from morag_core.interfaces.processor import ProcessingResult

        print("✅ ProcessingResult imported")
    except Exception as e:
        print(f"❌ ProcessingResult import failed: {e}")
        return False

    # Test audio imports
    try:
        from morag_audio import AudioProcessor
        from morag_audio.processor import AudioProcessingResult

        print("✅ Audio imports successful")
    except Exception as e:
        print(f"❌ Audio imports failed: {e}")
        return False

    # Test video imports
    try:
        from morag_video import VideoProcessor
        from morag_video.processor import VideoProcessingResult

        print("✅ Video imports successful")
    except Exception as e:
        print(f"❌ Video imports failed: {e}")
        return False

    # Test document imports
    try:
        from morag_document import DocumentProcessor

        print("✅ Document imports successful")
    except Exception as e:
        print(f"❌ Document imports failed: {e}")
        return False

    # Test image imports
    try:
        from morag_image import ImageProcessor
        from morag_image.processor import ImageProcessingResult

        print("✅ Image imports successful")
    except Exception as e:
        print(f"❌ Image imports failed: {e}")
        return False

    # Test web imports
    try:
        from morag_web import WebProcessor
        from morag_web.processor import WebScrapingResult

        print("✅ Web imports successful")
    except Exception as e:
        print(f"❌ Web imports failed: {e}")
        return False

    # Test youtube imports
    try:
        from morag_youtube import YouTubeProcessor
        from morag_youtube.processor import YouTubeDownloadResult

        print("✅ YouTube imports successful")
    except Exception as e:
        print(f"❌ YouTube imports failed: {e}")
        return False

    print("✅ All imports successful!")
    return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
