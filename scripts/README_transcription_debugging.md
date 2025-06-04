# Audio/Video Transcription Debugging and Testing

This directory contains scripts for debugging and testing the audio/video transcription system fixes.

## 🐛 Debugging Scripts

### `debug_transcription_issues.py`
Comprehensive debugging script to identify timestamp and text repetition issues.

**Usage:**
```bash
python scripts/debug_transcription_issues.py
```

**Features:**
- Tests timestamp calculation with detailed logging
- Analyzes text repetition patterns
- Traces the complete transcription pipeline
- Generates detailed debug logs

**Requirements:**
- Test audio file (test_audio.wav, sample.wav, etc.)
- All transcription dependencies installed

### `check_dependencies.py`
Checks for missing dependencies that might cause transcription warnings.

**Usage:**
```bash
python scripts/check_dependencies.py
```

**Features:**
- Checks all optional transcription dependencies
- Provides installation instructions for missing packages
- Identifies which features will be disabled
- Special handling for pyannote.audio, Playwright, and spaCy

### `generate_test_audio.py`
Generates a synthetic test audio file for testing when no real audio is available.

**Usage:**
```bash
python scripts/generate_test_audio.py
```

**Features:**
- Creates 15-second test audio with 2 simulated speakers
- Includes natural speech patterns and pauses
- Generates 16-bit mono WAV file
- Perfect for testing speaker diarization and topic segmentation

## 🧪 Testing Scripts

### `tests/manual/test_transcription_fixes.py`
Comprehensive validation suite for the transcription fixes.

**Usage:**
```bash
python tests/manual/test_transcription_fixes.py
```

**Tests:**
- **Timestamp Fixes**: Validates that topic timestamps are calculated correctly
- **Repetition Prevention**: Checks for text repetition patterns
- **Video Processing**: Tests video transcription with enhanced audio processing
- **Content Quality**: Analyzes the quality of generated markdown content

## 🔧 Fixed Issues

### 1. Timestamp Bug
**Problem:** Topic timestamps always showed as [0] instead of actual timestamps.

**Root Cause:** The `_calculate_topic_timing` method in `topic_segmentation.py` was failing to match sentences with transcript segments.

**Solution:**
- Enhanced text matching algorithm with multiple strategies
- Better fallback mechanisms using proportional mapping
- Improved logging for debugging timestamp calculation
- More lenient similarity thresholds (25% vs 30%)

### 2. Text Repetition Bug
**Problem:** Transcriptions getting stuck in infinite loops, repeating the last sentence multiple times.

**Root Cause:** Lack of deduplication in dialogue creation and repetitive patterns in final output.

**Solution:**
- Added comprehensive text deduplication in converters
- Implemented `_remove_repetitive_patterns` method
- Safeguards against consecutive repeated lines
- End-of-content repetition detection and removal

## 📋 Usage Instructions

### Step 1: Check Dependencies
```bash
python scripts/check_dependencies.py
```

Install any missing dependencies as suggested by the script.

### Step 2: Generate Test Audio (if needed)
```bash
python scripts/generate_test_audio.py
```

This creates a `test_audio.wav` file for testing.

### Step 3: Run Debug Analysis
```bash
python scripts/debug_transcription_issues.py
```

This will analyze the transcription pipeline and identify any remaining issues.

### Step 4: Validate Fixes
```bash
python tests/manual/test_transcription_fixes.py
```

This runs comprehensive tests to ensure the fixes are working correctly.

## 🔍 Debugging Tips

### For Timestamp Issues:
1. Check the debug logs for "Calculating topic timing" messages
2. Look for "Found timing match" vs "NO MATCH FOUND" entries
3. Verify that transcript segments have valid `start_time` and `end_time` attributes
4. Check if the text similarity matching is working correctly

### For Text Repetition Issues:
1. Look for "Skipping duplicate text entry" log messages
2. Check the final content for consecutive repeated lines
3. Examine the last 10-20 lines of the output for patterns
4. Verify that the `_remove_repetitive_patterns` method is being called

### For Missing Dependencies:
1. Run the dependency checker to see what's missing
2. Install optional dependencies based on your needs:
   - `pyannote.audio` for speaker diarization
   - `sentence-transformers` for topic segmentation
   - `playwright` for dynamic web content
   - `trafilatura` for advanced content extraction

## 📊 Expected Output Format

After the fixes, transcriptions should have:

```markdown
# Discussion Topic 1 [45]

SPEAKER_00: This is the first part of the conversation.
SPEAKER_01: And this is the response from the second speaker.

# Discussion Topic 2 [123]

SPEAKER_00: Now we're talking about a different topic.
SPEAKER_01: Yes, this is clearly a new subject.
```

**Key Features:**
- Topic headers with actual start timestamps in seconds: `[45]`, `[123]`
- Proper speaker identification: `SPEAKER_00`, `SPEAKER_01`
- No repetitive text at the end
- Clean, readable conversational format

## 🚨 Troubleshooting

### If timestamps are still showing as [0]:
1. Check if transcript segments have timing information
2. Verify that the text matching algorithm is finding matches
3. Enable debug logging to trace the timestamp calculation
4. Ensure the audio file has clear speech that Whisper can transcribe accurately

### If text repetition persists:
1. Check the debug logs for deduplication messages
2. Verify that the `_remove_repetitive_patterns` method is working
3. Look for issues in the dialogue creation process
4. Check if the topic segmentation is creating valid sentence lists

### If dependencies are missing:
1. Install the specific packages mentioned in the dependency checker
2. For pyannote.audio, you'll need a Hugging Face token
3. For Playwright, run `playwright install` after installation
4. For spaCy, run `python -m spacy download en_core_web_sm`

## 📝 Notes

- The fixes maintain backward compatibility with existing code
- Enhanced logging can be disabled by setting log level to WARNING or higher
- The text deduplication is conservative to avoid removing legitimate repeated content
- Timestamp calculation uses multiple fallback strategies for robustness
