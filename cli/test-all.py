#!/usr/bin/env python3
"""
MoRAG Complete System Test Script

Usage: python test-all.py

This script runs comprehensive tests for all MoRAG components.
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

try:
    from morag_audio import AudioProcessor, AudioConfig
    from morag_document import DocumentProcessor
    from morag_video import VideoProcessor, VideoConfig
    from morag_image import ImageProcessor
    from morag_web import WebProcessor
    from morag_youtube import YouTubeProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed all MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-audio")
    print("  pip install -e packages/morag-document")
    print("  pip install -e packages/morag-video")
    print("  pip install -e packages/morag-image")
    print("  pip install -e packages/morag-web")
    print("  pip install -e packages/morag-youtube")
    sys.exit(1)


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def print_result(key: str, value: str, indent: int = 0):
    """Print a formatted key-value result."""
    spaces = "  " * indent
    print(f"{spaces}📋 {key}: {value}")


def print_test_result(test_name: str, success: bool, duration: float, error: Optional[str] = None):
    """Print test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status} {test_name} ({duration:.2f}s)")
    if error and not success:
        print(f"    Error: {error}")


class SystemTester:
    """Comprehensive system tester for MoRAG."""
    
    def __init__(self):
        self.config = None
        self.results: Dict[str, Dict] = {}
        
    async def initialize(self) -> bool:
        """Initialize the test environment."""
        try:
            # No global config needed, each processor has its own config
            return True
        except Exception as e:
            print(f"❌ Failed to initialize configuration: {e}")
            return False
    
    async def test_component_initialization(self) -> Dict[str, bool]:
        """Test that all components can be initialized."""
        print_section("Component Initialization Tests")

        results = {}

        # Test AudioProcessor
        start_time = time.time()
        try:
            config = AudioConfig(model_size="base", device="auto")
            processor = AudioProcessor(config)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
        duration = time.time() - start_time
        results["AudioProcessor"] = success
        print_test_result("AudioProcessor initialization", success, duration, error)

        # Test DocumentProcessor
        start_time = time.time()
        try:
            processor = DocumentProcessor()
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
        duration = time.time() - start_time
        results["DocumentProcessor"] = success
        print_test_result("DocumentProcessor initialization", success, duration, error)

        # Test VideoProcessor
        start_time = time.time()
        try:
            config = VideoConfig(extract_audio=False, generate_thumbnails=False)
            processor = VideoProcessor(config)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
        duration = time.time() - start_time
        results["VideoProcessor"] = success
        print_test_result("VideoProcessor initialization", success, duration, error)

        # Test ImageProcessor
        start_time = time.time()
        try:
            processor = ImageProcessor()
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
        duration = time.time() - start_time
        results["ImageProcessor"] = success
        print_test_result("ImageProcessor initialization", success, duration, error)

        # Test WebProcessor
        start_time = time.time()
        try:
            processor = WebProcessor()
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
        duration = time.time() - start_time
        results["WebProcessor"] = success
        print_test_result("WebProcessor initialization", success, duration, error)

        # Test YouTubeProcessor
        start_time = time.time()
        try:
            processor = YouTubeProcessor()
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
        duration = time.time() - start_time
        results["YouTubeProcessor"] = success
        print_test_result("YouTubeProcessor initialization", success, duration, error)

        return results
    
    async def test_basic_functionality(self) -> bool:
        """Test basic system functionality."""
        print_section("Basic Functionality Tests")

        start_time = time.time()
        try:
            # Test that we can import all modules
            import morag_audio
            import morag_document
            import morag_video
            import morag_image
            import morag_web
            import morag_youtube

            success = True
            error = None

        except Exception as e:
            success = False
            error = str(e)

        duration = time.time() - start_time
        print_test_result("Module imports", success, duration, error)

        return success
    
    async def test_sample_files(self) -> Dict[str, bool]:
        """Test processing with sample files if available."""
        print_section("Sample File Processing Tests")
        
        # Look for sample files in common locations
        sample_files = {
            "audio": ["uploads/test_audio.wav", "uploads/Sprache.mp3", "uploads/recording.m4a"],
            "document": ["uploads/test_document.pdf", "uploads/super saftiger Schoko Schmand Kuchen fur ein ganzes Blech - 2014-02-08.pdf"],
            "video": ["uploads/test_video.mp4"],
        }
        
        results = {}
        
        for file_type, file_paths in sample_files.items():
            for file_path in file_paths:
                path = Path(file_path)
                if path.exists():
                    start_time = time.time()
                    try:
                        # Test file access and basic properties
                        file_size = path.stat().st_size
                        success = file_size > 0
                        error = None if success else "File is empty"
                    except Exception as e:
                        success = False
                        error = str(e)
                    
                    duration = time.time() - start_time
                    test_name = f"Sample {file_type} file access ({path.name})"
                    print_test_result(test_name, success, duration, error)
                    results[f"{file_type}_{path.name}"] = success
                    break
            else:
                # No sample file found for this type
                print_test_result(f"Sample {file_type} file", False, 0.0, "No sample file found")
                results[f"{file_type}_sample"] = False
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Dict]:
        """Run all system tests."""
        print_header("MoRAG Complete System Test Suite")
        
        if not await self.initialize():
            return {"initialization": {"success": False}}
        
        # Run all test categories
        test_results = {}

        # Basic functionality tests
        basic_valid = await self.test_basic_functionality()
        test_results["basic_functionality"] = {"success": basic_valid}

        # Component initialization tests
        component_results = await self.test_component_initialization()
        test_results["components"] = component_results

        # Sample file tests
        sample_results = await self.test_sample_files()
        test_results["sample_files"] = sample_results
        
        return test_results
    
    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate a comprehensive test report."""
        print_section("Test Summary Report")
        
        total_tests = 0
        passed_tests = 0
        
        for category, category_results in results.items():
            if isinstance(category_results, dict):
                if "success" in category_results:
                    # Single test result
                    total_tests += 1
                    if category_results["success"]:
                        passed_tests += 1
                    status = "✅ PASS" if category_results["success"] else "❌ FAIL"
                    print(f"  {status} {category}")
                else:
                    # Multiple test results
                    for test_name, success in category_results.items():
                        total_tests += 1
                        if success:
                            passed_tests += 1
                        status = "✅ PASS" if success else "❌ FAIL"
                        print(f"  {status} {category}.{test_name}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 Overall Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("\n🎉 System is ready for use!")
        elif success_rate >= 60:
            print("\n⚠️  System has some issues but may be functional")
        else:
            print("\n❌ System has significant issues and may not work properly")
        
        return f"Test completed: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)"


async def main():
    """Main test function."""
    tester = SystemTester()
    
    try:
        results = await tester.run_all_tests()
        
        # Save detailed results
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate and display report
        summary = tester.generate_report(results)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        print(f"📋 Summary: {summary}")
        
        # Determine exit code based on results
        total_success = all(
            result.get("success", True) if isinstance(result, dict) and "success" in result
            else all(result.values()) if isinstance(result, dict)
            else result
            for result in results.values()
        )
        
        sys.exit(0 if total_success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
