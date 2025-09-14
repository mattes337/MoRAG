# MoRAG Model Configuration Guide

This guide explains how to configure and override AI models used by MoRAG components.

## Audio Processing Models

### Whisper Model Configuration

MoRAG uses OpenAI's Whisper models for speech-to-text transcription. You can override the default model using environment variables.

#### Environment Variables

```bash
# Primary variable (recommended)
WHISPER_MODEL_SIZE=large-v3

# Alternative variable (both are supported)
MORAG_WHISPER_MODEL_SIZE=large-v3
```

#### Available Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | ~39 MB | Fastest | Lowest | Quick testing, low-resource environments |
| `base` | ~74 MB | Fast | Good | Balanced performance |
| `small` | ~244 MB | Medium | Better | Good quality transcription |
| `medium` | ~769 MB | Slower | High | **Default** - Good balance of quality and speed |
| `large-v2` | ~1550 MB | Slowest | Highest | Maximum accuracy |
| `large-v3` | ~1550 MB | Slowest | Highest | **Latest** - Best accuracy, recommended for GPU workers |

#### Recommendations

- **CPU Workers**: Use `base` or `small` for faster processing
- **GPU Workers**: Use `large-v3` for best accuracy
- **Development/Testing**: Use `tiny` for quick iterations
- **Production**: Use `medium` (default) or `large-v3` depending on accuracy requirements

### SpaCy Model Configuration

MoRAG uses spaCy models for natural language processing in topic segmentation.

#### Environment Variable

```bash
# Override spaCy model
MORAG_SPACY_MODEL=de_core_news_sm
```

#### Available Models

| Model | Language | Size | Features |
|-------|----------|------|----------|
| `en_core_web_sm` | English | Small | **Default** - Basic NLP |
| `en_core_web_md` | English | Medium | Better word vectors |
| `en_core_web_lg` | English | Large | Best word vectors |
| `de_core_news_sm` | German | Small | German NLP |
| `de_core_news_md` | German | Medium | Better German NLP |
| `fr_core_news_sm` | French | Small | French NLP |
| `es_core_news_sm` | Spanish | Small | Spanish NLP |

#### Installation

SpaCy models need to be installed separately:

```bash
# English models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md

# German models
python -m spacy download de_core_news_sm
python -m spacy download de_core_news_md

# Other languages
python -m spacy download fr_core_news_sm
python -m spacy download es_core_news_sm
```

## Audio Processing Features

### Speaker Diarization

Enable or disable speaker diarization (who spoke when):

```bash
MORAG_ENABLE_SPEAKER_DIARIZATION=true  # or false
```

### Topic Segmentation

Enable or disable automatic topic detection:

```bash
MORAG_ENABLE_TOPIC_SEGMENTATION=true  # or false
```

### Language Override

Force a specific language for transcription:

```bash
MORAG_AUDIO_LANGUAGE=de  # German
MORAG_AUDIO_LANGUAGE=en  # English
MORAG_AUDIO_LANGUAGE=fr  # French
```

If not set, language will be auto-detected.

### Device Selection

Force a specific device for processing:

```bash
MORAG_AUDIO_DEVICE=auto  # Auto-detect (default)
MORAG_AUDIO_DEVICE=cpu   # Force CPU
MORAG_AUDIO_DEVICE=cuda  # Force GPU
```

## Configuration Examples

### High-Accuracy GPU Setup

```bash
# .env file for GPU worker with maximum accuracy
WHISPER_MODEL_SIZE=large-v3
MORAG_AUDIO_DEVICE=cuda
MORAG_ENABLE_SPEAKER_DIARIZATION=true
MORAG_ENABLE_TOPIC_SEGMENTATION=true
MORAG_SPACY_MODEL=en_core_web_md
```

### Fast CPU Setup

```bash
# .env file for CPU worker optimized for speed
WHISPER_MODEL_SIZE=base
MORAG_AUDIO_DEVICE=cpu
MORAG_ENABLE_SPEAKER_DIARIZATION=false
MORAG_ENABLE_TOPIC_SEGMENTATION=false
```

### German Language Setup

```bash
# .env file for German audio processing
WHISPER_MODEL_SIZE=medium
MORAG_AUDIO_LANGUAGE=de
MORAG_SPACY_MODEL=de_core_news_sm
MORAG_ENABLE_SPEAKER_DIARIZATION=true
MORAG_ENABLE_TOPIC_SEGMENTATION=true
```

### Development/Testing Setup

```bash
# .env file for quick development testing
WHISPER_MODEL_SIZE=tiny
MORAG_AUDIO_DEVICE=cpu
MORAG_ENABLE_SPEAKER_DIARIZATION=false
MORAG_ENABLE_TOPIC_SEGMENTATION=false
```

## Remote Converter Configuration

When using the remote converter, all audio model configurations are supported through environment variables:

```bash
# In tools/remote-converter/.env
WHISPER_MODEL_SIZE=large-v3
MORAG_SPACY_MODEL=de_core_news_sm
MORAG_ENABLE_SPEAKER_DIARIZATION=true
MORAG_ENABLE_TOPIC_SEGMENTATION=true
```

The remote converter will automatically pass these environment variables to the audio processors.

## Troubleshooting

### Model Download Issues

If Whisper models fail to download:

```bash
# Clear cache and retry
rm -rf ~/.cache/whisper
# Models will be downloaded on next use
```

### SpaCy Model Not Found

If you get spaCy model errors:

```bash
# Install the specific model
python -m spacy download en_core_web_sm

# Or install all common models
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### GPU Memory Issues

If you get CUDA out of memory errors:

```bash
# Use a smaller model
WHISPER_MODEL_SIZE=medium  # instead of large-v3

# Or force CPU processing
MORAG_AUDIO_DEVICE=cpu
```

### Performance Optimization

For optimal performance:

1. **Match model size to hardware**: Use larger models on GPU workers, smaller on CPU
2. **Disable unused features**: Turn off diarization/segmentation if not needed
3. **Use appropriate language models**: Use language-specific spaCy models when possible
4. **Monitor resource usage**: Check CPU/GPU/memory usage and adjust accordingly

## Testing Configuration

Test your configuration with the provided test scripts:

```bash
# Test environment variable loading
cd tools/remote-converter
python test_env_config.py

# Test with .env file
python test_env_file.py
```

These scripts will verify that your environment variables are being read correctly and applied to the audio processors.
