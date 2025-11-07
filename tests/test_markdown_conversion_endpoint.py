"""Tests for markdown conversion endpoint."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from morag.server import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("# Test Document\n\nThis is a test document with some content.")
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for mocking."""
    return "# PDF Document\n\nThis is converted PDF content."


class TestMarkdownConversionEndpoint:
    """Test cases for markdown conversion endpoint."""

    def test_convert_text_file_success(self, client, sample_text_file):
        """Test successful conversion of text file."""
        with open(sample_text_file, "rb") as f:
            response = client.post(
                "/api/convert/markdown",
                files={"file": ("test.txt", f, "text/plain")},
                data={"include_metadata": "true"},
            )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "markdown" in data
        assert "Test Document" in data["markdown"]
        assert "metadata" in data
        assert data["metadata"]["original_format"] == "txt"
        assert data["metadata"]["file_size_bytes"] > 0
        assert data["processing_time_ms"] > 0

    def test_convert_with_metadata_disabled(self, client, sample_text_file):
        """Test conversion with metadata disabled."""
        with open(sample_text_file, "rb") as f:
            response = client.post(
                "/api/convert/markdown",
                files={"file": ("test.txt", f, "text/plain")},
                data={"include_metadata": "false"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "metadata" in data  # Metadata is always included in response

    def test_convert_with_language_hint(self, client, sample_text_file):
        """Test conversion with language hint."""
        with open(sample_text_file, "rb") as f:
            response = client.post(
                "/api/convert/markdown",
                files={"file": ("test.txt", f, "text/plain")},
                data={"language": "en"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["metadata"]["language"] == "en"

    @patch("morag_document.services.markitdown_service.MarkitdownService.convert_file")
    def test_convert_pdf_file(self, mock_convert, client, sample_pdf_content):
        """Test PDF file conversion."""
        mock_convert.return_value = sample_pdf_content

        # Create a dummy PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 dummy content")
            pdf_path = Path(f.name)

        try:
            with open(pdf_path, "rb") as f:
                response = client.post(
                    "/api/convert/markdown",
                    files={"file": ("test.pdf", f, "application/pdf")},
                )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["markdown"] == sample_pdf_content
            assert data["metadata"]["original_format"] == "pdf"
            assert "estimated_page_count" in data["metadata"]

        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    @patch("morag_document.services.markitdown_service.MarkitdownService.convert_file")
    def test_convert_audio_file(self, mock_convert, client):
        """Test audio file conversion."""
        mock_convert.return_value = (
            "# Audio Transcription\n\nThis is the transcribed content."
        )

        # Create a dummy audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"dummy audio content")
            audio_path = Path(f.name)

        try:
            with open(audio_path, "rb") as f:
                response = client.post(
                    "/api/convert/markdown",
                    files={"file": ("test.mp3", f, "audio/mpeg")},
                )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "Audio Transcription" in data["markdown"]
            assert data["metadata"]["original_format"] == "mp3"
            assert data["metadata"]["content_type"] == "audio"

        finally:
            if audio_path.exists():
                audio_path.unlink()

    @patch("morag_document.services.markitdown_service.MarkitdownService.convert_file")
    def test_convert_video_file(self, mock_convert, client):
        """Test video file conversion."""
        mock_convert.return_value = (
            "# Video Transcription\n\nThis is the transcribed video content."
        )

        # Create a dummy video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"dummy video content")
            video_path = Path(f.name)

        try:
            with open(video_path, "rb") as f:
                response = client.post(
                    "/api/convert/markdown",
                    files={"file": ("test.mp4", f, "video/mp4")},
                )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "Video Transcription" in data["markdown"]
            assert data["metadata"]["original_format"] == "mp4"
            assert data["metadata"]["content_type"] == "video"

        finally:
            if video_path.exists():
                video_path.unlink()

    def test_convert_audio_file_fallback(self, client):
        """Test audio file conversion with fallback when markitdown fails."""
        # Create a dummy audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"dummy audio content")
            audio_path = Path(f.name)

        try:
            with open(audio_path, "rb") as f:
                response = client.post(
                    "/api/convert/markdown",
                    files={"file": ("test.mp3", f, "audio/mpeg")},
                )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "Audio File: test.mp3" in data["markdown"]
            assert data["metadata"]["original_format"] == "mp3"

        finally:
            if audio_path.exists():
                audio_path.unlink()

    def test_convert_video_file_fallback(self, client):
        """Test video file conversion with fallback when markitdown fails."""
        # Create a dummy video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"dummy video content")
            video_path = Path(f.name)

        try:
            with open(video_path, "rb") as f:
                response = client.post(
                    "/api/convert/markdown",
                    files={"file": ("test.mp4", f, "video/mp4")},
                )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "Video File: test.mp4" in data["markdown"]
            assert data["metadata"]["original_format"] == "mp4"

        finally:
            if video_path.exists():
                video_path.unlink()

    def test_convert_unsupported_format(self, client):
        """Test conversion of unsupported file format."""
        # Create a dummy file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"dummy content")
            unsupported_path = Path(f.name)

        try:
            with open(unsupported_path, "rb") as f:
                response = client.post(
                    "/api/convert/markdown",
                    files={"file": ("test.xyz", f, "application/octet-stream")},
                )

            # Should either succeed with markitdown fallback or return error
            assert response.status_code in [200, 400]

            if response.status_code == 200:
                data = response.json()
                # If successful, should have some content or error message
                assert "success" in data

        finally:
            if unsupported_path.exists():
                unsupported_path.unlink()

    def test_convert_empty_file(self, client):
        """Test conversion of empty file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            # Create empty file
            empty_path = Path(f.name)

        try:
            with open(empty_path, "rb") as f:
                response = client.post(
                    "/api/convert/markdown",
                    files={"file": ("empty.txt", f, "text/plain")},
                )

            assert response.status_code == 200
            data = response.json()

            # Should handle empty files gracefully
            assert "success" in data
            assert "metadata" in data
            assert data["metadata"]["file_size_bytes"] == 0

        finally:
            if empty_path.exists():
                empty_path.unlink()

    def test_convert_large_file_metadata(self, client):
        """Test that large files are handled and metadata is correct."""
        # Create a larger text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            content = "# Large Document\n\n" + "This is a test line.\n" * 1000
            f.write(content)
            large_path = Path(f.name)

        try:
            with open(large_path, "rb") as f:
                response = client.post(
                    "/api/convert/markdown",
                    files={"file": ("large.txt", f, "text/plain")},
                )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["metadata"]["file_size_bytes"] > 1000
            assert "Large Document" in data["markdown"]

        finally:
            if large_path.exists():
                large_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
