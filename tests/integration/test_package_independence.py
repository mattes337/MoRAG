"""Integration tests for package independence.

This module tests that each MoRAG package can be imported and used
independently without requiring other packages.
"""

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


class TestPackageIndependence:
    """Test that packages can work independently."""

    PACKAGES = [
        "morag_core",
        "morag_services",
        "morag_web",
        "morag_youtube",
        "morag_audio",
        "morag_video",
        "morag_document",
        "morag_image",
    ]

    def test_package_imports_independently(self):
        """Test that each package can be imported without others."""
        for package_name in self.PACKAGES:
            # Test import in isolated subprocess
            result = self._test_import_in_subprocess(package_name)
            assert (
                result.returncode == 0
            ), f"Failed to import {package_name}: {result.stderr}"

    def _test_import_in_subprocess(
        self, package_name: str
    ) -> subprocess.CompletedProcess:
        """Test package import in isolated subprocess."""
        test_script = f"""
import sys
try:
    import {package_name}
    print(f"Successfully imported {package_name}")
    sys.exit(0)
except ImportError as e:
    print(f"Failed to import {package_name}: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"Error importing {package_name}: {{e}}")
    sys.exit(1)
"""

        return subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_core_package_functionality(self):
        """Test core package basic functionality."""
        try:
            from morag_core.interfaces.converter import BaseConverter
            from morag_core.interfaces.processor import BaseProcessor
            from morag_core.models import Document

            # Test basic model creation
            doc = Document(content="Test content", metadata={"test": True})
            assert doc.content == "Test content"
            assert doc.metadata["test"] is True

        except ImportError as e:
            pytest.skip(f"morag_core not available: {e}")

    def test_services_package_functionality(self):
        """Test services package basic functionality."""
        try:
            from morag_services import ServiceConfig

            # Test basic configuration
            config = ServiceConfig()
            assert config is not None

        except ImportError as e:
            pytest.skip(f"morag_services not available: {e}")

    def test_web_package_functionality(self):
        """Test web package basic functionality."""
        try:
            from morag_web import WebProcessor, WebScrapingConfig

            # Test basic processor creation
            config = WebScrapingConfig()
            processor = WebProcessor(config)
            assert processor is not None

        except ImportError as e:
            pytest.skip(f"morag_web not available: {e}")

    def test_audio_package_functionality(self):
        """Test audio package basic functionality."""
        try:
            from morag_audio import AudioConfig, AudioProcessor

            # Test basic processor creation
            config = AudioConfig()
            processor = AudioProcessor(config)
            assert processor is not None

        except ImportError as e:
            pytest.skip(f"morag_audio not available: {e}")

    def test_video_package_functionality(self):
        """Test video package basic functionality."""
        try:
            from morag_video import VideoConfig, VideoProcessor

            # Test basic processor creation
            config = VideoConfig()
            processor = VideoProcessor()
            assert processor is not None

        except ImportError as e:
            pytest.skip(f"morag_video not available: {e}")

    def test_document_package_functionality(self):
        """Test document package basic functionality."""
        try:
            from morag_document import DocumentProcessor

            # Test basic processor creation
            processor = DocumentProcessor()
            assert processor is not None

        except ImportError as e:
            pytest.skip(f"morag_document not available: {e}")

    def test_image_package_functionality(self):
        """Test image package basic functionality."""
        try:
            from morag_image import ImageConfig, ImageProcessor

            # Test basic processor creation
            config = ImageConfig()
            processor = ImageProcessor(config)
            assert processor is not None

        except ImportError as e:
            pytest.skip(f"morag_image not available: {e}")

    def test_youtube_package_functionality(self):
        """Test YouTube package basic functionality."""
        try:
            from morag_youtube import YouTubeProcessor

            # Test basic processor creation
            processor = YouTubeProcessor()
            assert processor is not None

        except ImportError as e:
            pytest.skip(f"morag_youtube not available: {e}")

    def test_package_isolation(self):
        """Test that packages don't interfere with each other."""
        # Import packages in different orders to test isolation
        import_orders = [
            ["morag_core", "morag_services", "morag_web"],
            ["morag_web", "morag_core", "morag_services"],
            ["morag_services", "morag_web", "morag_core"],
        ]

        for order in import_orders:
            result = self._test_import_order_in_subprocess(order)
            assert (
                result.returncode == 0
            ), f"Failed import order {order}: {result.stderr}"

    def _test_import_order_in_subprocess(
        self, packages: List[str]
    ) -> subprocess.CompletedProcess:
        """Test importing packages in specific order."""
        import_statements = "\n".join([f"import {pkg}" for pkg in packages])
        test_script = f"""
import sys
try:
{import_statements}
    print("Successfully imported packages in order: {packages}")
    sys.exit(0)
except Exception as e:
    print(f"Failed to import packages in order {packages}: {{e}}")
    sys.exit(1)
"""

        return subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_no_circular_dependencies(self):
        """Test that there are no circular dependencies between packages."""
        # This is a basic test - more sophisticated dependency analysis
        # would require parsing import statements

        for package_name in self.PACKAGES:
            try:
                # Import package and check if it can be imported cleanly
                result = self._test_import_in_subprocess(package_name)
                assert (
                    result.returncode == 0
                ), f"Circular dependency detected in {package_name}"
            except Exception as e:
                pytest.fail(
                    f"Error testing {package_name} for circular dependencies: {e}"
                )

    def test_package_versions_compatible(self):
        """Test that package versions are compatible."""
        try:
            # Test that we can import multiple packages together
            from morag_core import __version__ as core_version

            # If we get here, basic compatibility is working
            assert core_version is not None

        except ImportError as e:
            pytest.skip(f"Packages not available for version compatibility test: {e}")

    @pytest.mark.slow
    def test_package_memory_isolation(self):
        """Test that packages don't leak memory between imports."""
        import gc

        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        for package_name in self.PACKAGES:
            try:
                # Import package
                module = importlib.import_module(package_name)

                # Force garbage collection
                del module
                gc.collect()

                # Check memory hasn't grown excessively
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory

                # Allow for some memory growth but not excessive
                assert (
                    memory_growth < 100 * 1024 * 1024
                ), f"Excessive memory growth after importing {package_name}"

            except ImportError:
                # Skip if package not available
                continue


class TestPackageStructure:
    """Test package structure compliance."""

    def test_package_has_init_file(self):
        """Test that each package has proper __init__.py file."""
        packages_dir = Path("packages")

        if not packages_dir.exists():
            pytest.skip("Packages directory not found")

        for package_dir in packages_dir.iterdir():
            if package_dir.is_dir() and package_dir.name.startswith("morag"):
                init_file = (
                    package_dir
                    / "src"
                    / package_dir.name.replace("-", "_")
                    / "__init__.py"
                )
                assert init_file.exists(), f"Missing __init__.py in {package_dir.name}"

    def test_package_has_setup_file(self):
        """Test that each package has proper setup configuration."""
        packages_dir = Path("packages")

        if not packages_dir.exists():
            pytest.skip("Packages directory not found")

        for package_dir in packages_dir.iterdir():
            if package_dir.is_dir() and package_dir.name.startswith("morag"):
                # Check for pyproject.toml or setup.py
                has_pyproject = (package_dir / "pyproject.toml").exists()
                has_setup = (package_dir / "setup.py").exists()

                assert (
                    has_pyproject or has_setup
                ), f"Missing setup configuration in {package_dir.name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
