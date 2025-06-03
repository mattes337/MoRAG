import os
import tempfile
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import spacy
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import whisper
import uvicorn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import config

# Speaker diarization imports
try:
    from pyannote.audio import Pipeline
    SPEAKER_DIARIZATION_AVAILABLE = True
except ImportError:
    SPEAKER_DIARIZATION_AVAILABLE = False
    Pipeline = None

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE) if config.LOG_FILE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Print configuration if debug mode
if config.DEBUG:
    config.print_config()

app = FastAPI(
    title="Audio/Video Text Extractor",
    description="Transcription service for audio and video files with NLP analysis",
    version="1.0.0",
    debug=config.DEBUG
)

# Add CORS middleware if origins are configured
if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Load models with configuration
logger.info(f"Loading spaCy model: {config.SPACY_MODEL}")
try:
    nlp = spacy.load(config.SPACY_MODEL)
    logger.info("spaCy model loaded successfully")
except OSError as e:
    logger.error(f"Failed to load spaCy model {config.SPACY_MODEL}: {e}")
    raise

logger.info(f"Loading Whisper model: {config.WHISPER_MODEL}")
try:
    whisper_model = whisper.load_model(config.WHISPER_MODEL)
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model {config.WHISPER_MODEL}: {e}")
    raise

# Load speaker diarization model
speaker_pipeline = None
if config.ENABLE_SPEAKER_DIARIZATION and SPEAKER_DIARIZATION_AVAILABLE:
    logger.info(f"Loading speaker diarization model: {config.SPEAKER_DIARIZATION_MODEL}")
    try:
        # Try with authentication token if provided
        if config.HUGGINGFACE_TOKEN:
            logger.info("Using Hugging Face authentication token")
            speaker_pipeline = Pipeline.from_pretrained(
                config.SPEAKER_DIARIZATION_MODEL,
                use_auth_token=config.HUGGINGFACE_TOKEN
            )
        else:
            # Try without token first
            speaker_pipeline = Pipeline.from_pretrained(config.SPEAKER_DIARIZATION_MODEL)

        logger.info("Speaker diarization model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load speaker diarization model: {e}")
        error_str = str(e).lower()
        if ("gated" in error_str or "private" in error_str or "authentication" in error_str or
            "segmentation-3.0" in error_str or "could not download" in error_str):
            logger.warning("The pyannote models are gated and require authentication. Please:")
            logger.warning("1. Visit https://hf.co/pyannote/speaker-diarization-3.1 to accept user conditions")
            logger.warning("2. Visit https://hf.co/pyannote/segmentation-3.0 to accept user conditions")
            logger.warning("3. Create a token at https://hf.co/settings/tokens")
            logger.warning("4. Set HUGGINGFACE_TOKEN environment variable in your .env file")
            logger.warning("5. Restart the service")
            logger.warning("6. Or disable speaker diarization with ENABLE_SPEAKER_DIARIZATION=false")
        logger.warning("Speaker diarization will be disabled - using single speaker fallback")
        speaker_pipeline = None
elif config.ENABLE_SPEAKER_DIARIZATION and not SPEAKER_DIARIZATION_AVAILABLE:
    logger.warning("Speaker diarization enabled but pyannote.audio not available")
    logger.warning("Install pyannote.audio to enable speaker diarization")
    logger.warning("Using single speaker fallback")

class TextProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.max_file_size = config.get_max_file_size_bytes()
        logger.info(f"TextProcessor initialized with max file size: {config.MAX_FILE_SIZE}")

    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file"""
        try:
            logger.info(f"Extracting audio from video: {video_path}")
            video = VideoFileClip(video_path)

            # Create audio file in temp directory
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(config.TEMP_DIR, f"{base_name}_audio.wav")

            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            logger.info(f"Audio extracted successfully: {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Video processing error: {str(e)}")

    def transcribe_with_whisper(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe audio using Whisper with language detection"""
        try:
            logger.info(f"Transcribing audio with Whisper model: {config.WHISPER_MODEL}")

            # Load and preprocess audio for language detection
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(whisper_model.device)

            # Detect language if not specified
            language_info = {}
            if language is None:
                logger.info("Detecting language...")
                _, probs = whisper_model.detect_language(mel)
                detected_language = max(probs, key=probs.get)
                language_confidence = probs[detected_language]

                language_info = {
                    "detected_language": detected_language,
                    "language_confidence": float(language_confidence),
                    "language_probabilities": {lang: float(prob) for lang, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]},
                    "language_source": "auto_detected"
                }

                # Use detected language for transcription
                transcribe_language = detected_language
                logger.info(f"Detected language: {detected_language} (confidence: {language_confidence:.3f})")
            else:
                language_info = {
                    "detected_language": language,
                    "language_confidence": 1.0,
                    "language_probabilities": {language: 1.0},
                    "language_source": "manually_specified"
                }
                transcribe_language = language
                logger.info(f"Using manually specified language: {language}")

            # Transcribe with detected or specified language
            result = whisper_model.transcribe(audio_path, language=transcribe_language)
            transcript = result["text"].strip()

            logger.info(f"Transcription completed, length: {len(transcript)} characters")

            return {
                "transcript": transcript,
                "language_info": language_info,
                "whisper_result": {
                    "language": result.get("language", transcribe_language),
                    "segments": result.get("segments", [])
                }
            }

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Transcription error: {str(e)}")

    def validate_file_size(self, file_size: int) -> None:
        """Validate file size against configured maximum"""
        if file_size > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file_size} bytes) exceeds maximum allowed size ({config.MAX_FILE_SIZE})"
            )

    def perform_speaker_diarization(self, audio_path: str) -> Dict[str, Any]:
        """Perform speaker diarization on audio file"""
        if not speaker_pipeline:
            logger.info("Speaker diarization model not available, using simple fallback")
            return self._simple_speaker_fallback(audio_path)

        try:
            logger.info("Performing speaker diarization...")

            # Apply the pipeline to the audio file
            diarization = speaker_pipeline(audio_path)

            # Extract speaker information
            speakers = {}
            speaker_segments = []
            speaker_counter = 1

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speakers:
                    speakers[speaker] = speaker_counter
                    speaker_counter += 1

                speaker_segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": speakers[speaker],
                    "speaker_label": speaker
                })

            logger.info(f"Speaker diarization completed: {len(speakers)} speakers identified")

            return {
                "speakers": speakers,
                "speaker_segments": speaker_segments,
                "num_speakers": len(speakers)
            }

        except Exception as e:
            logger.error(f"Speaker diarization error: {str(e)}")
            # Fallback to simple method
            return self._simple_speaker_fallback(audio_path)

    def _simple_speaker_fallback(self, audio_path: str) -> Dict[str, Any]:
        """Simple speaker diarization fallback using audio analysis"""
        try:
            # Try to use pydub for basic audio analysis
            from pydub import AudioSegment
            from pydub.silence import split_on_silence

            logger.info("Using simple speaker diarization fallback")

            # Load audio
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0  # Convert to seconds

            # Simple heuristic: if audio is longer than 30 seconds, assume 2 speakers
            # This is a very basic fallback - real speaker diarization would be much more sophisticated
            if duration > 30:
                # Split roughly in half and assume speaker change
                mid_point = duration / 2
                return {
                    "speakers": {"SPEAKER_00": 1, "SPEAKER_01": 2},
                    "speaker_segments": [
                        {"start": 0.0, "end": mid_point, "speaker": 1, "speaker_label": "SPEAKER_00"},
                        {"start": mid_point, "end": duration, "speaker": 2, "speaker_label": "SPEAKER_01"}
                    ],
                    "num_speakers": 2
                }
            else:
                # Short audio, assume single speaker
                return {
                    "speakers": {"SPEAKER_00": 1},
                    "speaker_segments": [
                        {"start": 0.0, "end": duration, "speaker": 1, "speaker_label": "SPEAKER_00"}
                    ],
                    "num_speakers": 1
                }

        except Exception as e:
            logger.warning(f"Simple speaker fallback failed: {e}")
            # Ultimate fallback - single speaker
            return {
                "speakers": {"SPEAKER_00": 1},
                "speaker_segments": [
                    {"start": 0.0, "end": 60.0, "speaker": 1, "speaker_label": "SPEAKER_00"}
                ],
                "num_speakers": 1
            }

    def map_sentences_to_speakers(self, sentences: List, whisper_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict[str, Any]]:
        """Map sentences to speakers based on timing information"""
        sentence_speaker_mapping = []

        # Get sentence timings first
        sentence_timings = self._map_sentences_to_timing(sentences, whisper_segments)

        for i, sentence in enumerate(sentences):
            sentence_text = sentence.text.strip() if hasattr(sentence, 'text') else str(sentence).strip()
            sentence_start_time = sentence_timings[i]

            # Find the speaker for this sentence based on timing
            assigned_speaker = 1  # Default speaker

            for speaker_segment in speaker_segments:
                # Check if sentence timing overlaps with speaker segment
                if (sentence_start_time >= speaker_segment["start"] and
                    sentence_start_time <= speaker_segment["end"]):
                    assigned_speaker = speaker_segment["speaker"]
                    break

            sentence_speaker_mapping.append({
                "speaker": assigned_speaker,
                "text": sentence_text
            })

        return sentence_speaker_mapping

    def segment_text_by_topic_with_timing(self, text: str, whisper_segments: List[Dict], speaker_segments: List[Dict] = None) -> List[Dict[str, Any]]:
        """Segment text into topics with timing information and summaries"""
        if not config.ENABLE_TOPIC_SEGMENTATION:
            # Return single topic with all content
            doc = nlp(text)
            sentences = list(doc.sents)

            # Map sentences to speakers if speaker segments available
            if speaker_segments:
                sentence_objects = self.map_sentences_to_speakers(sentences, whisper_segments, speaker_segments)
            else:
                sentence_objects = [{"speaker": 1, "text": sent.text.strip()} for sent in sentences]

            return [{
                "summary": "",  # Will be filled by LLM later
                "seconds": 0.0,
                "sentences": sentence_objects if sentence_objects else []
            }]

        try:
            doc = nlp(text)
            sentences = list(doc.sents)

            # If we have fewer sentences than minimum, return as single topic
            if len(sentences) < config.MIN_TOPIC_SENTENCES:
                # Map sentences to speakers if speaker segments available
                if speaker_segments:
                    sentence_objects = self.map_sentences_to_speakers(sentences, whisper_segments, speaker_segments)
                else:
                    sentence_objects = [{"speaker": 1, "text": sent.text.strip()} for sent in sentences]

                return [{
                    "summary": "",  # Will be filled by LLM later
                    "seconds": 0.0,
                    "sentences": sentence_objects
                }]

            # Map sentences to timing information from Whisper segments
            sentence_timings = self._map_sentences_to_timing(sentences, whisper_segments)

            # Check if the spaCy model has word vectors
            if not nlp.meta.get('vectors', {}).get('keys', 0):
                logger.warning("spaCy model doesn't have word vectors. Using fallback topic segmentation.")
                return self._fallback_topic_segmentation_with_timing(sentences, sentence_timings, speaker_segments, whisper_segments)

            # Calculate sentence embeddings using spaCy
            sentence_vectors = []
            valid_sentences = []
            valid_timings = []

            for i, sent in enumerate(sentences):
                if sent.vector.any():  # Check if sentence has a valid vector
                    sentence_vectors.append(sent.vector)
                    valid_sentences.append(sent)
                    valid_timings.append(sentence_timings[i])

            if len(sentence_vectors) < 2:
                # Map sentences to speakers if speaker segments available
                if speaker_segments:
                    sentence_objects = self.map_sentences_to_speakers(sentences, whisper_segments, speaker_segments)
                else:
                    sentence_objects = [{"speaker": 1, "text": sent.text.strip()} for sent in sentences]

                return [{
                    "summary": "",  # Will be filled by LLM later
                    "seconds": 0.0,
                    "sentences": sentence_objects
                }]

            # Convert to numpy array for sklearn
            sentence_vectors = np.array(sentence_vectors)

            # Calculate cosine similarities between consecutive sentences
            topic_segments = []
            current_segment = {
                "sentences": [valid_sentences[0]],
                "start_time": valid_timings[0]
            }

            for i in range(1, len(valid_sentences)):
                # Calculate similarity between consecutive sentences
                similarity = cosine_similarity(
                    sentence_vectors[i-1].reshape(1, -1),
                    sentence_vectors[i].reshape(1, -1)
                )[0][0]

                if similarity > config.TOPIC_SIMILARITY_THRESHOLD:
                    current_segment["sentences"].append(valid_sentences[i])
                else:
                    # Topic change detected
                    if len(current_segment["sentences"]) >= config.MIN_TOPIC_SENTENCES:
                        topic_segments.append(current_segment)
                    else:
                        # If segment is too short, merge with previous
                        if topic_segments:
                            topic_segments[-1]["sentences"].extend(current_segment["sentences"])
                        else:
                            # Start new segment anyway if it's the first one
                            topic_segments.append(current_segment)

                    current_segment = {
                        "sentences": [valid_sentences[i]],
                        "start_time": valid_timings[i]
                    }

            # Add the last segment
            if current_segment["sentences"]:
                if len(current_segment["sentences"]) >= config.MIN_TOPIC_SENTENCES or not topic_segments:
                    topic_segments.append(current_segment)
                else:
                    # Merge short final segment with previous
                    if topic_segments:
                        topic_segments[-1]["sentences"].extend(current_segment["sentences"])
                    else:
                        topic_segments.append(current_segment)

            # Ensure we have at least one segment
            if not topic_segments:
                topic_segments = [{
                    "sentences": sentences,
                    "start_time": 0.0
                }]

            # Convert to final format with summaries and speaker information
            topics = []
            for segment in topic_segments:
                # Map sentences to speakers for this segment
                if speaker_segments:
                    sentence_objects = self.map_sentences_to_speakers(segment["sentences"], whisper_segments, speaker_segments)
                else:
                    sentence_objects = [{"speaker": 1, "text": sent.text.strip()} for sent in segment["sentences"]]

                topics.append({
                    "summary": "",  # Will be filled by LLM later
                    "seconds": float(segment["start_time"]),
                    "sentences": sentence_objects
                })

            logger.info(f"Topic segmentation completed: {len(topics)} topics identified")
            return topics

        except Exception as e:
            logger.error(f"Topic segmentation error: {str(e)}")
            # Fallback to single topic with speaker information
            doc = nlp(text)
            sentences = list(doc.sents)

            if speaker_segments:
                sentence_objects = self.map_sentences_to_speakers(sentences, whisper_segments, speaker_segments)
            else:
                sentence_objects = [{"speaker": 1, "text": text}] if text.strip() else []

            return [{
                "summary": "",  # Will be filled by LLM later
                "seconds": 0.0,
                "sentences": sentence_objects
            }]

    def _map_sentences_to_timing(self, sentences: List, whisper_segments: List[Dict]) -> List[float]:
        """Map sentences to timing information from Whisper segments"""
        sentence_timings = []

        if not whisper_segments:
            # No timing information available, use default
            return [0.0] * len(sentences)

        # Create a mapping of text to timing
        segment_text_to_time = {}
        for segment in whisper_segments:
            if 'text' in segment and 'start' in segment:
                segment_text_to_time[segment['text'].strip()] = segment['start']

        # Try to match sentences to segments
        for sentence in sentences:
            sentence_text = sentence.text.strip()
            best_match_time = 0.0

            # Look for exact or partial matches
            for segment_text, start_time in segment_text_to_time.items():
                if sentence_text in segment_text or segment_text in sentence_text:
                    best_match_time = start_time
                    break

            sentence_timings.append(best_match_time)

        return sentence_timings

    def _fallback_topic_segmentation_with_timing(self, sentences: List, sentence_timings: List[float], speaker_segments: List[Dict] = None, whisper_segments: List[Dict] = None) -> List[Dict[str, Any]]:
        """Fallback topic segmentation with timing information"""
        chunk_size = max(config.MIN_TOPIC_SENTENCES, len(sentences) // 3)
        topics = []

        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            chunk_timings = sentence_timings[i:i + chunk_size]

            # Map sentences to speakers for this chunk
            if speaker_segments and whisper_segments:
                sentence_objects = self.map_sentences_to_speakers(chunk, whisper_segments, speaker_segments)
            else:
                sentence_objects = [{"speaker": 1, "text": sent.text.strip()} for sent in chunk]

            start_time = chunk_timings[0] if chunk_timings else 0.0

            topics.append({
                "summary": "",  # Will be filled by LLM later
                "seconds": float(start_time),
                "sentences": sentence_objects
            })

        if not topics:
            # Fallback case
            if speaker_segments and whisper_segments:
                sentence_objects = self.map_sentences_to_speakers(sentences, whisper_segments, speaker_segments)
            else:
                sentence_objects = [{"speaker": 1, "text": sent.text.strip()} for sent in sentences]

            return [{
                "summary": "",  # Will be filled by LLM later
                "seconds": 0.0,
                "sentences": sentence_objects
            }]

        return topics

    def _fallback_topic_segmentation(self, sentences: List) -> List[str]:
        """Fallback topic segmentation based on sentence count"""
        # Simple fallback: split into chunks of sentences
        chunk_size = max(config.MIN_TOPIC_SENTENCES, len(sentences) // 3)
        segments = []

        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            segments.append(" ".join([sent.text.strip() for sent in chunk]))

        return segments if segments else [" ".join([sent.text.strip() for sent in sentences])]

    def structure_text_with_spacy(self, text: str, whisper_segments: List[Dict] = None, speaker_segments: List[Dict] = None) -> Dict[str, Any]:
        """Process text with spaCy for structured extraction"""
        doc = nlp(text)

        # Extract structured information
        sentences = list(doc.sents)

        # Map sentences to speakers if speaker segments available
        if speaker_segments:
            sentence_objects = self.map_sentences_to_speakers(sentences, whisper_segments or [], speaker_segments)
        else:
            sentence_objects = [{"speaker": 1, "text": sent.text.strip()} for sent in sentences]

        structured_data = {
            "raw_text": text,
            "sentences": sentence_objects,
            "entities": [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "description": spacy.explain(ent.label_),
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ],
            "keywords": [
                {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "is_stop": token.is_stop
                }
                for token in doc
                if not token.is_stop and not token.is_punct and len(token.text) > 2
            ],
            "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
            "summary_stats": {
                "total_tokens": len(doc),
                "sentences_count": len(list(doc.sents)),
                "entities_count": len(doc.ents),
                "unique_entities": len(set([ent.label_ for ent in doc.ents]))
            }
        }

        # Add topic segmentation with timing and speaker information
        if whisper_segments is None:
            whisper_segments = []

        topics = self.segment_text_by_topic_with_timing(text, whisper_segments, speaker_segments)
        structured_data["topics"] = topics
        structured_data["summary_stats"]["topics_count"] = len(topics)

        # Add speaker information to summary stats
        if speaker_segments:
            unique_speakers = set(seg["speaker"] for seg in speaker_segments)
            structured_data["summary_stats"]["speakers_count"] = len(unique_speakers)
        else:
            structured_data["summary_stats"]["speakers_count"] = 1

        return structured_data

processor = TextProcessor()

@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    language: str = Query(None, description="Language code for transcription (e.g., 'en', 'es', 'fr'). If not specified, language will be auto-detected.")
):
    """Process audio file and extract structured text with language detection"""
    logger.info(f"Processing audio file: {file.filename}")

    # Validate file format
    supported_formats = config.get_supported_audio_formats()
    if not any(file.filename.lower().endswith(fmt) for fmt in supported_formats):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Supported formats: {', '.join(supported_formats)}"
        )

    # Read file content and validate size
    content = await file.read()
    processor.validate_file_size(len(content))

    # Create temporary file in configured temp directory
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1],
        dir=config.TEMP_DIR
    )
    try:
        tmp_file.write(content)
        tmp_file.flush()
        tmp_file.close()  # Close file before processing

        # Transcribe audio with language detection
        transcription_result = processor.transcribe_with_whisper(tmp_file.name, language=language)
        transcript = transcription_result["transcript"]
        language_info = transcription_result["language_info"]

        # Perform speaker diarization
        speaker_result = processor.perform_speaker_diarization(tmp_file.name)

        # Structure with spaCy including speaker information
        structured_data = processor.structure_text_with_spacy(
            transcript,
            transcription_result["whisper_result"]["segments"],
            speaker_result["speaker_segments"]
        )
        structured_data["source_type"] = "audio"
        structured_data["filename"] = file.filename
        structured_data["file_size"] = len(content)
        structured_data["language_detection"] = language_info
        structured_data["whisper_segments"] = transcription_result["whisper_result"]["segments"]
        structured_data["speaker_diarization"] = speaker_result
        structured_data["models_used"] = {
            "whisper": config.WHISPER_MODEL,
            "spacy": config.SPACY_MODEL,
            "speaker_diarization": config.SPEAKER_DIARIZATION_MODEL if speaker_pipeline else "disabled"
        }

        logger.info(f"Audio processing completed for: {file.filename}")
        return structured_data

    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file.name)
        except (OSError, PermissionError):
            logger.warning(f"Failed to cleanup temporary file: {tmp_file.name}")

@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    language: str = Query(None, description="Language code for transcription (e.g., 'en', 'es', 'fr'). If not specified, language will be auto-detected.")
):
    """Process video file and extract structured text from audio with language detection"""
    logger.info(f"Processing video file: {file.filename}")

    # Validate file format
    supported_formats = config.get_supported_video_formats()
    if not any(file.filename.lower().endswith(fmt) for fmt in supported_formats):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Supported formats: {', '.join(supported_formats)}"
        )

    # Read file content and validate size
    content = await file.read()
    processor.validate_file_size(len(content))

    # Create temporary file in configured temp directory
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1],
        dir=config.TEMP_DIR
    )
    audio_path = None
    try:
        tmp_file.write(content)
        tmp_file.flush()
        tmp_file.close()  # Close file before processing

        # Extract audio from video
        audio_path = processor.extract_audio_from_video(tmp_file.name)

        # Transcribe audio with language detection
        transcription_result = processor.transcribe_with_whisper(audio_path, language=language)
        transcript = transcription_result["transcript"]
        language_info = transcription_result["language_info"]

        # Perform speaker diarization on extracted audio
        speaker_result = processor.perform_speaker_diarization(audio_path)

        # Structure with spaCy including speaker information
        structured_data = processor.structure_text_with_spacy(
            transcript,
            transcription_result["whisper_result"]["segments"],
            speaker_result["speaker_segments"]
        )
        structured_data["source_type"] = "video"
        structured_data["filename"] = file.filename
        structured_data["file_size"] = len(content)
        structured_data["language_detection"] = language_info
        structured_data["whisper_segments"] = transcription_result["whisper_result"]["segments"]
        structured_data["speaker_diarization"] = speaker_result
        structured_data["models_used"] = {
            "whisper": config.WHISPER_MODEL,
            "spacy": config.SPACY_MODEL,
            "speaker_diarization": config.SPEAKER_DIARIZATION_MODEL if speaker_pipeline else "disabled"
        }

        logger.info(f"Video processing completed for: {file.filename}")
        return structured_data

    finally:
        # Clean up temporary files
        try:
            os.unlink(tmp_file.name)
        except (OSError, PermissionError):
            logger.warning(f"Failed to cleanup temporary video file: {tmp_file.name}")
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except (OSError, PermissionError):
                logger.warning(f"Failed to cleanup temporary audio file: {audio_path}")

@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    language: str = Query(None, description="Language code for transcription (e.g., 'en', 'es', 'fr'). If not specified, language will be auto-detected.")
):
    """Process audio or video file and extract structured text with automatic format detection"""
    logger.info(f"Processing file: {file.filename}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Get supported formats
    audio_formats = config.get_supported_audio_formats()
    video_formats = config.get_supported_video_formats()

    # Check if file is audio or video
    filename_lower = file.filename.lower()
    is_audio = any(filename_lower.endswith(fmt) for fmt in audio_formats)
    is_video = any(filename_lower.endswith(fmt) for fmt in video_formats)

    if is_audio:
        logger.info(f"Detected audio file: {file.filename}")
        # Reset file position since we need to read it again
        await file.seek(0)
        return await process_audio(file, language)
    elif is_video:
        logger.info(f"Detected video file: {file.filename}")
        # Reset file position since we need to read it again
        await file.seek(0)
        return await process_video(file, language)
    else:
        # Unsupported format
        all_formats = audio_formats + video_formats
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(sorted(all_formats))}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint with configuration information"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "configuration": {
            "whisper_model": config.WHISPER_MODEL,
            "spacy_model": config.SPACY_MODEL,
            "max_file_size": config.MAX_FILE_SIZE,
            "supported_audio_formats": config.get_supported_audio_formats(),
            "supported_video_formats": config.get_supported_video_formats(),
            "test_mode": config.TEST_MODE,
            "topic_segmentation": {
                "enabled": config.ENABLE_TOPIC_SEGMENTATION,
                "similarity_threshold": config.TOPIC_SIMILARITY_THRESHOLD,
                "min_topic_sentences": config.MIN_TOPIC_SENTENCES
            }
        }
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "Audio/Video Transcription Service API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/config",
            "/process",
            "/process-audio",
            "/process-video"
        ],
        "documentation": "/docs"
    }

@app.get("/config")
async def get_config():
    """Get current service configuration"""
    return {
        "whisper_model": config.WHISPER_MODEL,
        "spacy_model": config.SPACY_MODEL,
        "max_file_size": config.MAX_FILE_SIZE,
        "temp_dir": config.TEMP_DIR,
        "debug": config.DEBUG,
        "log_level": config.LOG_LEVEL,
        "supported_formats": {
            "audio": config.get_supported_audio_formats(),
            "video": config.get_supported_video_formats()
        },
        "topic_segmentation": {
            "enabled": config.ENABLE_TOPIC_SEGMENTATION,
            "similarity_threshold": config.TOPIC_SIMILARITY_THRESHOLD,
            "min_topic_sentences": config.MIN_TOPIC_SENTENCES
        },
        "speaker_diarization": {
            "enabled": config.ENABLE_SPEAKER_DIARIZATION,
            "model": config.SPEAKER_DIARIZATION_MODEL,
            "min_speakers": config.MIN_SPEAKERS,
            "max_speakers": config.MAX_SPEAKERS,
            "available": speaker_pipeline is not None
        }
    }

if __name__ == "__main__":
    logger.info(f"Starting transcription service on {config.HOST}:{config.PORT}")
    if config.RELOAD:
        # For development with reload
        uvicorn.run(
            "app:app",  # Use import string for reload
            host=config.HOST,
            port=config.PORT,
            reload=True,
            log_level=config.LOG_LEVEL.lower()
        )
    else:
        # For production - use import string to enable workers
        if config.MAX_WORKERS > 1:
            uvicorn.run(
                "app:app",  # Use import string for workers
                host=config.HOST,
                port=config.PORT,
                log_level=config.LOG_LEVEL.lower(),
                workers=config.MAX_WORKERS
            )
        else:
            # Single worker mode
            uvicorn.run(
                app,
                host=config.HOST,
                port=config.PORT,
                log_level=config.LOG_LEVEL.lower()
            )
