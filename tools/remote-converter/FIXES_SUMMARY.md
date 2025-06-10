# Remote Converter Fixes Summary

## Issues Fixed

### 1. 'AudioProcessor' object has no attribute 'process_audio'

**Problem**: The remote converter was calling `processor.process_audio(file_path, options)` but AudioProcessor only has a `process()` method.

**Fix**: Updated method calls in `_process_file()` to use the correct method names:
- `AudioProcessor.process(file_path)` ✓
- `VideoProcessor.process_video(file_path)` ✓ (was incorrectly using process())
- `DocumentProcessor.process_file(file_path)` ✓
- `ImageProcessor.process(file_path)` ✓
- `WebProcessor.process_url(url)` ✓
- `YouTubeProcessor.process_url(url)` ✓

### 2. ProcessingResult.__init__() got an unexpected keyword argument 'text_content'

**Problem**: The remote converter was importing `ProcessingResult` from `morag_core.models` and trying to create it with a `text_content` parameter, but that class doesn't have this parameter.

**Fix**:
- Created a unified `ProcessingResult` class specifically for the remote converter with the required `text_content` parameter
- Added a `convert_to_unified_result()` function to convert processor-specific results to the unified format
- Updated all result creation to use the new unified format

### 3. 'AudioProcessingResult' object has no attribute 'language'

**Problem**: VideoProcessor was trying to access `audio_result.language`, `audio_result.speaker_segments`, and `audio_result.topic_segments` but these attributes don't exist on AudioProcessingResult.

**Fix**:
- Updated VideoProcessor to access correct attributes from AudioProcessingResult.metadata
- Added language detection storage in AudioProcessor
- Fixed attribute access: `audio_result.metadata.get("language", "unknown")`, `audio_result.metadata.get("num_speakers", 0)`, `audio_result.metadata.get("num_topics", 0)`

### 4. VideoProcessor ignoring WHISPER_MODEL_SIZE environment variable

**Problem**: Remote worker was initializing with correct `large-v3` model at startup, but when processing video files, the VideoProcessor was creating a new AudioProcessor instance that defaulted to `base` model size, ignoring the `WHISPER_MODEL_SIZE` environment variable.

**Fix**:
- Added `__post_init__` method to `VideoConfig` class to read environment variables (`WHISPER_MODEL_SIZE`, `MORAG_ENABLE_SPEAKER_DIARIZATION`, `MORAG_ENABLE_TOPIC_SEGMENTATION`)
- Updated `VideoProcessor._get_audio_processor()` to let `AudioConfig` handle environment variables instead of explicitly passing video config values
- Added logic to only override AudioConfig defaults when VideoConfig has non-default values

## Files Modified

### `tools/remote-converter/remote_converter.py`

1. **Updated imports** to import directly from processor modules instead of `__init__.py` files to avoid heavy initialization
2. **Added unified ProcessingResult class** with `text_content` parameter
3. **Added convert_to_unified_result() function** to handle conversion from processor-specific results
4. **Fixed method calls** in `_process_file()` to use correct processor method names
5. **Updated result handling** to use the conversion function

### `packages/morag-video/src/morag_video/processor.py`

1. **Fixed AudioProcessingResult attribute access** to use correct metadata fields
2. **Updated logging** to access language, speakers, and topics from metadata instead of non-existent attributes

### `packages/morag-audio/src/morag_audio/processor.py`

1. **Added language detection storage** to store detected language in metadata for VideoProcessor compatibility

### `packages/morag-video/src/morag_video/processor.py` (Environment Variable Fix)

1. **Added `__post_init__` method to VideoConfig** to read environment variables for audio processing configuration
2. **Updated `_get_audio_processor()` method** to let AudioConfig handle environment variables instead of forcing video config values
3. **Added conditional override logic** to only override AudioConfig defaults when VideoConfig has explicit non-default values

## Key Changes

```python
# Before (broken)
result = await processor.process_audio(file_path, options)
return ProcessingResult(
    success=False,
    text_content="",  # This parameter didn't exist
    metadata={},
    processing_time=0.0,
    error_message="..."
)

# After (fixed)
result = await processor.process(file_path)  # Correct method name
unified_result = convert_to_unified_result(result, processing_time)
return unified_result  # Uses our unified ProcessingResult class
```

## Result Conversion Logic

The `convert_to_unified_result()` function handles different processor result types:

- **AudioProcessingResult**: Extracts `transcript` as `text_content`
- **VideoProcessingResult**: Extracts transcript from `audio_processing_result`
- **BaseProcessingResult**: Handles document/image results from interface
- **ImageProcessingResult**: Combines caption and extracted text
- **WebScrapingResult**: Extracts markdown content
- **YouTubeDownloadResult**: Formats metadata as text content

## Testing

Created test scripts to verify fixes:
- `test_simple.py`: Tests core logic without heavy imports
- `test_imports.py`: Tests import functionality
- `remote_converter_minimal.py`: Minimal version for testing

## Status

✅ **Fixed**: Method name issues (AudioProcessor.process_audio → process, VideoProcessor.process → process_video)
✅ **Fixed**: ProcessingResult parameter issues (added text_content parameter)
✅ **Fixed**: Result conversion logic (unified conversion function)
✅ **Fixed**: Import structure (avoid heavy initialization)
✅ **Fixed**: AudioProcessingResult attribute access (language, speaker_segments, topic_segments → metadata fields)
✅ **Fixed**: VideoProcessor environment variable handling (WHISPER_MODEL_SIZE now properly respected)

The remote converter should now work correctly with all MoRAG processors without any attribute or method errors.
