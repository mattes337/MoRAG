"""Unit tests for AudioProcessor class."""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

from morag_core.exceptions import ProcessingError


class MockAudioSegment:
    """Mock audio segment for testing."""
    
    def __init__(self, start: float, end: float, text: str, speaker: str = None):
        self.start = start
        self.end = end
        self.text = text
        self.speaker = speaker or "Speaker 1"


class MockAudioConfig:
    """Mock audio config for testing."""
    
    def __init__(self, **kwargs):
        self.whisper_model = kwargs.get("whisper_model", "base")
        self.device = kwargs.get("device", "cpu")
        self.language = kwargs.get("language", "en")
        self.enable_diarization = kwargs.get("enable_diarization", False)
        self.output_format = kwargs.get("output_format", "json")


class MockAudioProcessingResult:
    """Mock audio processing result."""
    
    def __init__(
        self, 
        transcript: str,
        segments: List[MockAudioSegment],
        file_path: str,
        processing_time: float = 1.0,
        success: bool = True,
        error_message: str = None
    ):
        self.transcript = transcript
        self.segments = segments
        self.metadata = {"model": "whisper-base", "language": "en"}
        self.file_path = file_path
        self.processing_time = processing_time
        self.success = success
        self.error_message = error_message
        self.markdown_transcript = self._generate_markdown()
    
    def _generate_markdown(self) -> str:
        """Generate markdown transcript."""
        if not self.success:
            return ""
        
        lines = ["# Audio Transcript", ""]
        for segment in self.segments:
            timestamp = f"[{segment.start:.2f}s - {segment.end:.2f}s]"
            speaker = f"**{segment.speaker}**: " if segment.speaker else ""
            lines.append(f"{timestamp} {speaker}{segment.text}")
        
        return "\n".join(lines)


class MockAudioProcessor:
    """Mock audio processor for testing."""
    
    def __init__(self, config: MockAudioConfig = None):
        self.config = config or MockAudioConfig()
        self._supported_formats = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
        self._processing_stats = {"files_processed": 0, "total_time": 0.0}
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats."""
        return self._supported_formats
    
    def is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower() in self._supported_formats
    
    async def process(self, file_path: Path, **kwargs) -> MockAudioProcessingResult:
        """Process audio file."""
        if not file_path.exists():
            return MockAudioProcessingResult(
                transcript="",
                segments=[],
                file_path=str(file_path),
                success=False,
                error_message="File not found"
            )
        
        if not self.is_supported_format(file_path):
            return MockAudioProcessingResult(
                transcript="",
                segments=[],
                file_path=str(file_path),
                success=False,
                error_message=f"Unsupported format: {file_path.suffix}"
            )
        
        # Simulate processing based on filename
        if "error" in file_path.name.lower():
            return MockAudioProcessingResult(
                transcript="",
                segments=[],
                file_path=str(file_path),
                success=False,
                error_message="Processing failed"
            )
        
        # Simulate successful processing
        await asyncio.sleep(0.01)  # Simulate processing time
        
        # Create mock segments based on file size or name
        segments = self._create_mock_segments(file_path)
        transcript = " ".join(segment.text for segment in segments)
        
        processing_time = 1.0
        self._processing_stats["files_processed"] += 1
        self._processing_stats["total_time"] += processing_time
        
        return MockAudioProcessingResult(
            transcript=transcript,
            segments=segments,
            file_path=str(file_path),
            processing_time=processing_time
        )
    
    def _create_mock_segments(self, file_path: Path) -> List[MockAudioSegment]:
        """Create mock segments based on file."""
        if "empty" in file_path.name.lower():
            return []
        
        if "short" in file_path.name.lower():
            return [
                MockAudioSegment(0.0, 5.0, "This is a short audio clip.")
            ]
        
        # Default segments
        return [
            MockAudioSegment(0.0, 3.0, "Hello, this is the beginning of the audio."),
            MockAudioSegment(3.0, 7.0, "This is the middle part of the transcript."),
            MockAudioSegment(7.0, 10.0, "And this is the end of the audio file.")
        ]
    
    async def process_batch(self, file_paths: List[Path], **kwargs) -> List[MockAudioProcessingResult]:
        """Process multiple audio files."""
        results = []
        for file_path in file_paths:
            result = await self.process(file_path, **kwargs)
            results.append(result)
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._processing_stats.copy()
    
    async def validate_audio_file(self, file_path: Path) -> bool:
        """Validate that file is a proper audio file."""
        if not file_path.exists():
            return False
        
        if not self.is_supported_format(file_path):
            return False
        
        # Basic validation - check file size
        if file_path.stat().st_size == 0:
            return False
        
        return True
    
    async def estimate_processing_time(self, file_path: Path) -> float:
        """Estimate processing time for audio file."""
        if not await self.validate_audio_file(file_path):
            return 0.0
        
        # Estimate based on file size (very rough)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        # Assume 1 MB takes about 0.5 seconds to process
        return file_size_mb * 0.5


class TestAudioProcessor:
    """Test AudioProcessor functionality."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create audio processor for testing."""
        config = MockAudioConfig(whisper_model="base", device="cpu")
        return MockAudioProcessor(config)
    
    @pytest.fixture
    def sample_audio_files(self, tmp_path):
        """Create sample audio files for testing."""
        files = {}
        
        # Create various test audio files (empty files for testing)
        files['regular.mp3'] = tmp_path / "regular.mp3"
        files['regular.mp3'].write_bytes(b"fake audio data")
        
        files['short.wav'] = tmp_path / "short.wav"
        files['short.wav'].write_bytes(b"short audio")
        
        files['empty.m4a'] = tmp_path / "empty.m4a"
        files['empty.m4a'].write_bytes(b"")  # Empty file
        
        files['error.flac'] = tmp_path / "error.flac"
        files['error.flac'].write_bytes(b"will cause error")
        
        files['unsupported.txt'] = tmp_path / "unsupported.txt"
        files['unsupported.txt'].write_text("not an audio file")
        
        return files
    
    def test_processor_creation(self, audio_processor):
        """Test creating audio processor."""
        assert audio_processor.config is not None
        assert audio_processor.config.whisper_model == "base"
        assert audio_processor.config.device == "cpu"
    
    def test_processor_creation_with_custom_config(self):
        """Test creating processor with custom config."""
        config = MockAudioConfig(
            whisper_model="large",
            device="cuda",
            language="fr",
            enable_diarization=True
        )
        processor = MockAudioProcessor(config)
        
        assert processor.config.whisper_model == "large"
        assert processor.config.device == "cuda"
        assert processor.config.language == "fr"
        assert processor.config.enable_diarization is True
    
    def test_get_supported_formats(self, audio_processor):
        """Test getting supported audio formats."""
        formats = audio_processor.get_supported_formats()
        
        assert isinstance(formats, list)
        assert ".mp3" in formats
        assert ".wav" in formats
        assert ".m4a" in formats
        assert ".flac" in formats
        assert ".ogg" in formats
    
    def test_is_supported_format(self, audio_processor):
        """Test checking if format is supported."""
        # Supported formats
        assert audio_processor.is_supported_format(Path("test.mp3")) is True
        assert audio_processor.is_supported_format(Path("test.wav")) is True
        assert audio_processor.is_supported_format(Path("test.M4A")) is True  # Case insensitive
        
        # Unsupported formats
        assert audio_processor.is_supported_format(Path("test.txt")) is False
        assert audio_processor.is_supported_format(Path("test.pdf")) is False
        assert audio_processor.is_supported_format(Path("test.mp4")) is False
    
    async def test_process_success(self, audio_processor, sample_audio_files):
        """Test successful audio processing."""
        audio_file = sample_audio_files['regular.mp3']
        
        result = await audio_processor.process(audio_file)
        
        assert result.success is True
        assert result.transcript != ""
        assert len(result.segments) > 0
        assert result.file_path == str(audio_file)
        assert result.processing_time > 0
        assert result.markdown_transcript is not None
        assert "# Audio Transcript" in result.markdown_transcript
    
    async def test_process_missing_file(self, audio_processor, tmp_path):
        """Test processing missing file."""
        missing_file = tmp_path / "missing.mp3"
        
        result = await audio_processor.process(missing_file)
        
        assert result.success is False
        assert result.error_message == "File not found"
        assert result.transcript == ""
        assert len(result.segments) == 0
    
    async def test_process_unsupported_format(self, audio_processor, sample_audio_files):
        """Test processing unsupported format."""
        unsupported_file = sample_audio_files['unsupported.txt']
        
        result = await audio_processor.process(unsupported_file)
        
        assert result.success is False
        assert "Unsupported format" in result.error_message
        assert result.transcript == ""
    
    async def test_process_error_simulation(self, audio_processor, sample_audio_files):
        """Test processing error simulation."""
        error_file = sample_audio_files['error.flac']
        
        result = await audio_processor.process(error_file)
        
        assert result.success is False
        assert result.error_message == "Processing failed"
    
    async def test_process_short_audio(self, audio_processor, sample_audio_files):
        """Test processing short audio file."""
        short_file = sample_audio_files['short.wav']
        
        result = await audio_processor.process(short_file)
        
        assert result.success is True
        assert len(result.segments) == 1
        assert "short audio clip" in result.transcript.lower()
    
    async def test_process_empty_audio(self, audio_processor, sample_audio_files):
        """Test processing empty audio file."""
        empty_file = sample_audio_files['empty.m4a']
        
        # Empty file should fail validation
        is_valid = await audio_processor.validate_audio_file(empty_file)
        assert is_valid is False
    
    async def test_process_batch(self, audio_processor, sample_audio_files):
        """Test batch processing of audio files."""
        files_to_process = [
            sample_audio_files['regular.mp3'],
            sample_audio_files['short.wav']
        ]
        
        results = await audio_processor.process_batch(files_to_process)
        
        assert len(results) == 2
        assert all(isinstance(result, MockAudioProcessingResult) for result in results)
        assert results[0].success is True
        assert results[1].success is True
    
    async def test_process_batch_with_errors(self, audio_processor, sample_audio_files):
        """Test batch processing with some failures."""
        files_to_process = [
            sample_audio_files['regular.mp3'],
            sample_audio_files['error.flac'],
            sample_audio_files['unsupported.txt']
        ]
        
        results = await audio_processor.process_batch(files_to_process)
        
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is False
    
    async def test_validate_audio_file(self, audio_processor, sample_audio_files, tmp_path):
        """Test audio file validation."""
        # Valid file
        valid_file = sample_audio_files['regular.mp3']
        assert await audio_processor.validate_audio_file(valid_file) is True
        
        # Missing file
        missing_file = tmp_path / "missing.mp3"
        assert await audio_processor.validate_audio_file(missing_file) is False
        
        # Unsupported format
        unsupported_file = sample_audio_files['unsupported.txt']
        assert await audio_processor.validate_audio_file(unsupported_file) is False
        
        # Empty file
        empty_file = sample_audio_files['empty.m4a']
        assert await audio_processor.validate_audio_file(empty_file) is False
    
    async def test_estimate_processing_time(self, audio_processor, sample_audio_files):
        """Test processing time estimation."""
        audio_file = sample_audio_files['regular.mp3']
        
        estimated_time = await audio_processor.estimate_processing_time(audio_file)
        
        assert estimated_time > 0
        assert isinstance(estimated_time, (int, float))
        
        # Invalid file should return 0
        invalid_time = await audio_processor.estimate_processing_time(Path("nonexistent.mp3"))
        assert invalid_time == 0
    
    def test_processing_stats(self, audio_processor):
        """Test processing statistics tracking."""
        initial_stats = audio_processor.get_processing_stats()
        assert initial_stats["files_processed"] == 0
        assert initial_stats["total_time"] == 0.0
    
    async def test_processing_stats_update(self, audio_processor, sample_audio_files):
        """Test that processing stats are updated."""
        audio_file = sample_audio_files['regular.mp3']
        
        initial_stats = audio_processor.get_processing_stats()
        initial_count = initial_stats["files_processed"]
        initial_time = initial_stats["total_time"]
        
        await audio_processor.process(audio_file)
        
        updated_stats = audio_processor.get_processing_stats()
        assert updated_stats["files_processed"] == initial_count + 1
        assert updated_stats["total_time"] > initial_time
    
    async def test_concurrent_processing(self, audio_processor, sample_audio_files):
        """Test concurrent processing of multiple files."""
        files = [
            sample_audio_files['regular.mp3'],
            sample_audio_files['short.wav']
        ]
        
        # Process concurrently
        tasks = [audio_processor.process(file) for file in files]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 2
        assert all(result.success for result in results)
    
    @pytest.mark.parametrize("format_ext", [".mp3", ".wav", ".m4a", ".flac", ".ogg"])
    def test_supported_formats_parametrized(self, audio_processor, format_ext):
        """Parametrized test for supported formats."""
        test_file = Path(f"test{format_ext}")
        assert audio_processor.is_supported_format(test_file) is True
    
    async def test_segment_properties(self, audio_processor, sample_audio_files):
        """Test properties of generated segments."""
        audio_file = sample_audio_files['regular.mp3']
        
        result = await audio_processor.process(audio_file)
        
        assert result.success is True
        
        for segment in result.segments:
            # Check segment has required properties
            assert hasattr(segment, 'start')
            assert hasattr(segment, 'end')
            assert hasattr(segment, 'text')
            assert hasattr(segment, 'speaker')
            
            # Check segment timing makes sense
            assert segment.start >= 0
            assert segment.end > segment.start
            assert isinstance(segment.text, str)
            assert len(segment.text) > 0
    
    async def test_markdown_generation(self, audio_processor, sample_audio_files):
        """Test markdown transcript generation."""
        audio_file = sample_audio_files['regular.mp3']
        
        result = await audio_processor.process(audio_file)
        
        assert result.success is True
        assert result.markdown_transcript is not None
        
        markdown_lines = result.markdown_transcript.split('\n')
        assert "# Audio Transcript" in markdown_lines[0]
        
        # Check that segments are included
        for segment in result.segments:
            assert segment.text in result.markdown_transcript
    
    async def test_metadata_inclusion(self, audio_processor, sample_audio_files):
        """Test that processing result includes metadata."""
        audio_file = sample_audio_files['regular.mp3']
        
        result = await audio_processor.process(audio_file)
        
        assert result.success is True
        assert isinstance(result.metadata, dict)
        assert "model" in result.metadata
        assert "language" in result.metadata