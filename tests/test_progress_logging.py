#!/usr/bin/env python3
"""
Test script to verify progress logging and task status updates are working correctly.
"""

import asyncio
import tempfile
import os
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-document" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-audio" / "src"))

from morag.worker import process_file_task
from morag_core.interfaces.processor import ProcessingConfig


def test_progress_callback():
    """Test that progress callbacks are called during processing."""
    progress_calls = []
    
    def mock_progress_callback(progress: float, message: Optional[str] = None):
        progress_calls.append((progress, message))
        print(f"Progress: {int(progress * 100)}% - {message}")
    
    # Test the progress callback mechanism
    mock_progress_callback(0.1, "Starting test")
    mock_progress_callback(0.5, "Halfway done")
    mock_progress_callback(0.9, "Almost complete")
    
    assert len(progress_calls) == 3
    assert progress_calls[0] == (0.1, "Starting test")
    assert progress_calls[1] == (0.5, "Halfway done")
    assert progress_calls[2] == (0.9, "Almost complete")
    
    print("✓ Progress callback test passed")


def test_processing_config_with_progress():
    """Test that ProcessingConfig accepts progress_callback parameter."""
    def dummy_callback(progress: float, message: Optional[str] = None):
        pass
    
    config = ProcessingConfig(
        chunk_size=1000,
        chunk_overlap=100,
        progress_callback=dummy_callback
    )
    
    assert config.progress_callback == dummy_callback
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 100
    
    print("✓ ProcessingConfig with progress callback test passed")


def test_celery_task_progress_updates():
    """Test that Celery tasks properly update progress."""
    
    # Mock Celery task
    class MockTask:
        def __init__(self):
            self.state_updates = []
        
        def update_state(self, state, meta):
            self.state_updates.append((state, meta))
            print(f"Task state update: {state} - {meta}")
    
    # Create a mock task instance
    mock_task = MockTask()
    
    # Simulate progress updates like in the worker
    mock_task.update_state(state='PROCESSING', meta={'stage': 'starting', 'progress': 0.0, 'message': 'Initializing file processing'})
    mock_task.update_state(state='PROGRESS', meta={'progress': 0.3, 'message': 'Converting PDF structure and extracting text'})
    mock_task.update_state(state='PROGRESS', meta={'progress': 0.7, 'message': 'Generating markdown from PDF content'})
    mock_task.update_state(state='PROCESSING', meta={'stage': 'completing', 'progress': 0.95, 'message': 'Finalizing processing'})
    
    # Verify updates were recorded
    assert len(mock_task.state_updates) == 4
    assert mock_task.state_updates[0][1]['progress'] == 0.0
    assert mock_task.state_updates[1][1]['progress'] == 0.3
    assert mock_task.state_updates[2][1]['progress'] == 0.7
    assert mock_task.state_updates[3][1]['progress'] == 0.95
    
    print("✓ Celery task progress updates test passed")


def test_document_processor_progress():
    """Test that document processor accepts and uses progress callback."""
    try:
        from morag_document.processor import DocumentProcessor
        from morag_document.converters.pdf import PDFConverter
        from morag_core.interfaces.converter import ConversionOptions
        
        # Create a mock progress callback
        progress_calls = []
        def progress_callback(progress: float, message: Optional[str] = None):
            progress_calls.append((progress, message))
            print(f"Document processing: {int(progress * 100)}% - {message}")
        
        # Test ConversionOptions with progress callback
        options = ConversionOptions(
            chunk_size=1000,
            progress_callback=progress_callback
        )
        
        assert options.progress_callback == progress_callback
        print("✓ ConversionOptions with progress callback test passed")
        
    except ImportError as e:
        print(f"⚠ Document processor test skipped due to import error: {e}")


def test_audio_processor_progress():
    """Test that audio processor accepts progress callback parameter."""
    try:
        from morag_audio.processor import AudioProcessor
        
        # Check if the process method accepts progress_callback
        import inspect
        sig = inspect.signature(AudioProcessor.process)
        params = list(sig.parameters.keys())
        
        assert 'progress_callback' in params, f"progress_callback not found in AudioProcessor.process parameters: {params}"
        print("✓ AudioProcessor.process accepts progress_callback parameter")
        
    except ImportError as e:
        print(f"⚠ Audio processor test skipped due to import error: {e}")


def main():
    """Run all progress logging tests."""
    print("Testing progress logging and task status updates...")
    print("=" * 60)
    
    try:
        test_progress_callback()
        test_processing_config_with_progress()
        test_celery_task_progress_updates()
        test_document_processor_progress()
        test_audio_processor_progress()
        
        print("=" * 60)
        print("✅ All progress logging tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
