"""Tests for the CLI module."""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from morag_image.cli import main, parse_args, process_directory, process_single_image
from morag_image.service import ImageService

# Skip tests if API key is not available
pytestmark = pytest.mark.skipif(
    os.environ.get("GEMINI_API_KEY") is None
    and os.environ.get("GOOGLE_API_KEY") is None,
    reason="GEMINI_API_KEY environment variable not set",
)


# Test argument parsing
def test_parse_args():
    """Test command line argument parsing."""
    # Test with minimal arguments
    with patch.object(sys, "argv", ["morag_image", "test.jpg"]):
        args = parse_args()
        assert args.input == "test.jpg"
        assert args.output is None
        assert args.caption is False
        assert args.ocr is False
        assert args.metadata is False
        assert args.ocr_engine == "tesseract"
        assert args.max_dimension == 1024
        assert args.max_concurrency == 3

    # Test with all arguments
    with patch.object(
        sys,
        "argv",
        [
            "morag_image",
            "test.jpg",
            "--output",
            "results.json",
            "--caption",
            "--ocr",
            "--metadata",
            "--ocr-engine",
            "easyocr",
            "--api-key",
            "test-key",
            "--max-dimension",
            "800",
            "--max-concurrency",
            "5",
        ],
    ):
        args = parse_args()
        assert args.input == "test.jpg"
        assert args.output == "results.json"
        assert args.caption is True
        assert args.ocr is True
        assert args.metadata is True
        assert args.ocr_engine == "easyocr"
        assert args.api_key == "test-key"
        assert args.max_dimension == 800
        assert args.max_concurrency == 5


# Test processing a single image
@pytest.mark.asyncio
async def test_process_single_image():
    """Test processing a single image file."""
    # Create mock service and file path
    mock_service = MagicMock(spec=ImageService)
    mock_service.process_image.return_value = {
        "caption": "Test caption",
        "extracted_text": "Test text",
        "metadata": {"width": 800, "height": 600},
    }

    file_path = Path("/path/to/test.jpg")
    config = {"generate_caption": True}

    # Process image
    result = await process_single_image(mock_service, file_path, config)

    # Check result
    assert result["file_path"] == str(file_path)
    assert result["caption"] == "Test caption"
    assert result["extracted_text"] == "Test text"
    assert result["metadata"]["width"] == 800

    # Check service was called correctly
    mock_service.process_image.assert_called_once_with(file_path, config)


# Test processing a directory
@pytest.mark.asyncio
async def test_process_directory():
    """Test processing all images in a directory."""
    # Create mock service and directory path
    mock_service = MagicMock(spec=ImageService)
    mock_service.process_batch.return_value = [
        {"file_path": "/path/to/image1.jpg", "caption": "Caption 1"},
        {"file_path": "/path/to/image2.jpg", "caption": "Caption 2"},
    ]

    dir_path = Path("/path/to/images")
    config = {"generate_caption": True}

    # Mock Path.glob to return image files
    with patch.object(Path, "glob") as mock_glob:
        # Set up mock to return different files for different extensions
        def side_effect(pattern):
            if "**/*.jpg" in pattern or "**/*.JPG" in pattern:
                return [Path("/path/to/image1.jpg"), Path("/path/to/image2.jpg")]
            return []

        mock_glob.side_effect = side_effect

        # Process directory
        results = await process_directory(mock_service, dir_path, config, 3)

    # Check results
    assert len(results) == 2
    assert results[0]["caption"] == "Caption 1"
    assert results[1]["caption"] == "Caption 2"

    # Check service was called correctly
    mock_service.process_batch.assert_called_once()
    args, kwargs = mock_service.process_batch.call_args
    assert len(args[0]) == 2  # Two image files
    assert args[1] == config
    assert kwargs["max_concurrency"] == 3


# Test main function with file input
@pytest.mark.asyncio
async def test_main_with_file():
    """Test main function with a file input."""
    # Mock command line arguments
    mock_args = MagicMock()
    mock_args.input = "test.jpg"
    mock_args.output = None
    mock_args.caption = True
    mock_args.ocr = False
    mock_args.metadata = False
    mock_args.ocr_engine = "tesseract"
    mock_args.api_key = None
    mock_args.max_dimension = 1024
    mock_args.max_concurrency = 3

    # Mock Path.is_file and Path.is_dir
    mock_path = MagicMock(spec=Path)
    mock_path.is_file.return_value = True
    mock_path.is_dir.return_value = False

    # Mock process_single_image
    mock_result = {"file_path": "test.jpg", "caption": "Test caption"}

    with patch("morag_image.cli.parse_args", return_value=mock_args), patch(
        "morag_image.cli.Path", return_value=mock_path
    ), patch("morag_image.cli.ImageService"), patch(
        "morag_image.cli.process_single_image", return_value=mock_result
    ), patch(
        "builtins.print"
    ) as mock_print, patch(
        "json.dumps", return_value='{"result": "test"}'
    ):
        # Run main function
        await main()

    # Check output
    mock_print.assert_called_once_with('{"result": "test"}')


# Test main function with directory input and output file
@pytest.mark.asyncio
async def test_main_with_directory_and_output():
    """Test main function with a directory input and output file."""
    # Mock command line arguments
    mock_args = MagicMock()
    mock_args.input = "images/"
    mock_args.output = "results.json"
    mock_args.caption = False
    mock_args.ocr = True
    mock_args.metadata = True
    mock_args.ocr_engine = "easyocr"
    mock_args.api_key = "test-key"
    mock_args.max_dimension = 800
    mock_args.max_concurrency = 5

    # Mock Path.is_file and Path.is_dir
    mock_input_path = MagicMock(spec=Path)
    mock_input_path.is_file.return_value = False
    mock_input_path.is_dir.return_value = True

    mock_output_path = MagicMock(spec=Path)

    # Mock process_directory
    mock_results = [
        {"file_path": "images/1.jpg", "extracted_text": "Text 1"},
        {"file_path": "images/2.jpg", "extracted_text": "Text 2"},
    ]

    # Create a side effect for Path constructor
    def path_side_effect(path_str):
        if path_str == "images/":
            return mock_input_path
        elif path_str == "results.json":
            return mock_output_path
        return MagicMock(spec=Path)

    with patch("morag_image.cli.parse_args", return_value=mock_args), patch(
        "morag_image.cli.Path", side_effect=path_side_effect
    ), patch("morag_image.cli.ImageService"), patch(
        "morag_image.cli.process_directory", return_value=mock_results
    ), patch(
        "builtins.open", mock_open()
    ) as mock_file, patch(
        "json.dump"
    ) as mock_json_dump:
        # Run main function
        await main()

    # Check output
    mock_file.assert_called_once_with(mock_output_path, "w", encoding="utf-8")
    mock_json_dump.assert_called_once()
    args, kwargs = mock_json_dump.call_args
    assert args[0] == mock_results
    assert kwargs["indent"] == 2
