"""Celery tasks for audio processing."""

from typing import Dict, Any, List, Optional
import structlog
import asyncio
from pathlib import Path

from morag.core.celery_app import celery_app
from morag.tasks.base import ProcessingTask
from morag.processors.audio import audio_processor, AudioConfig
from morag.services.whisper_service import whisper_service
from morag.services.embedding import gemini_service
from morag.services.chunking import chunking_service
from morag.services.storage import qdrant_service
from morag.services.summarization import enhanced_summarization_service, SummaryConfig, SummaryStrategy

logger = structlog.get_logger()

async def _process_audio_file_impl(
    self,
    file_path: str,
    task_id: str,
    config: Optional[Dict[str, Any]] = None,
    use_enhanced_summary: bool = True
) -> Dict[str, Any]:
    """Async implementation of audio file processing."""

    logger.info("Starting audio processing task",
               task_id=task_id,
               file_path=file_path)

    try:
        # Update task status
        self.update_status("PROCESSING", {"stage": "audio_transcription"})

        # Parse config
        audio_config = AudioConfig(**config) if config else AudioConfig()

        # Process audio file
        audio_result = await audio_processor.process_audio_file(
            file_path=file_path,
            config=audio_config
        )

        logger.info("Audio transcription completed",
                   task_id=task_id,
                   text_length=len(audio_result.text),
                   language=audio_result.language,
                   confidence=audio_result.confidence)

        # Update task status
        self.update_status("PROCESSING", {
            "stage": "text_chunking",
            "transcription_complete": True,
            "language": audio_result.language,
            "confidence": audio_result.confidence
        })

        # Chunk the transcribed text
        chunks = await chunking_service.chunk_with_metadata(
            text=audio_result.text,
            strategy="semantic"
        )

        logger.info("Text chunking completed",
                   task_id=task_id,
                   chunk_count=len(chunks))

        # Update task status
        self.update_status("PROCESSING", {
            "stage": "embedding_generation",
            "chunk_count": len(chunks)
        })

        # Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            logger.debug("Processing audio chunk",
                        task_id=task_id,
                        chunk_index=i,
                        chunk_length=len(chunk.text))

            try:
                # Generate summary if requested
                summary = None
                chunk_metadata = {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source_type": "audio",
                    "audio_confidence": audio_result.confidence,
                    "audio_language": audio_result.language
                }

                if use_enhanced_summary:
                    # Use enhanced summarization
                    enhanced_result = await enhanced_summarization_service.generate_summary(
                        chunk.text,
                        config=SummaryConfig(
                            strategy=SummaryStrategy.ABSTRACTIVE,
                            max_length=100,
                            style="concise",
                            enable_refinement=True,
                            document_type=None  # Audio content is general
                        )
                    )
                    summary = enhanced_result.summary

                    # Add enhanced summary metadata
                    chunk_metadata.update({
                        "enhanced_summary": True,
                        "summary_strategy": enhanced_result.strategy.value,
                        "summary_quality": enhanced_result.quality.overall,
                        "summary_processing_time": enhanced_result.processing_time,
                        "summary_refinement_iterations": enhanced_result.refinement_iterations
                    })
                else:
                    # Use basic summarization
                    summary_result = await gemini_service.generate_summary(
                        chunk.text,
                        max_length=100,
                        style="concise"
                    )
                    summary = summary_result.summary
                    chunk_metadata["basic_summary"] = True

                # Generate embedding for combined text and summary
                combined_text = f"{chunk.text}\n\nSummary: {summary}"
                embedding_result = await gemini_service.generate_embedding(
                    combined_text,
                    task_type="retrieval_document"
                )

                # Store in vector database
                point_id = await qdrant_service.store_chunk(
                    chunk_id=f"{task_id}_chunk_{i}",
                    text=chunk.text,
                    summary=summary,
                    embedding=embedding_result.embedding,
                    metadata={
                        "source_type": "audio",
                        "file_path": file_path,
                        "language": audio_result.language,
                        "duration": audio_result.duration,
                        "confidence": audio_result.confidence,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "sentence_count": chunk.sentence_count,
                        "word_count": chunk.word_count,
                        "chunk_type": chunk.chunk_type,
                        **chunk_metadata,
                        "embedding_model": embedding_result.model,
                        "embedding_token_count": embedding_result.token_count,
                        **audio_result.metadata
                    }
                )

                processed_chunks.append({
                    "chunk_id": f"{task_id}_chunk_{i}",
                    "point_id": point_id,
                    "text_length": len(chunk.text),
                    "summary_length": len(summary) if summary else 0,
                    "metadata": chunk_metadata
                })

                # Update progress
                progress = (i + 1) / len(chunks) * 100
                self.update_status("PROCESSING", {
                    "stage": "embedding_generation",
                    "progress": progress,
                    "chunks_processed": i + 1,
                    "total_chunks": len(chunks)
                })

            except Exception as e:
                logger.error("Failed to process audio chunk",
                           task_id=task_id,
                           chunk_index=i,
                           error=str(e))
                # Continue with other chunks
                continue

        # Final result
        result = {
            "task_id": task_id,
            "status": "completed",
            "audio_result": {
                "text": audio_result.text,
                "language": audio_result.language,
                "confidence": audio_result.confidence,
                "duration": audio_result.duration,
                "segments_count": len(audio_result.segments),
                "processing_time": audio_result.processing_time,
                "model_used": audio_result.model_used
            },
            "chunks_processed": len(processed_chunks),
            "total_chunks": len(chunks),
            "processed_chunks": processed_chunks,
            "metadata": {
                "file_path": file_path,
                "use_enhanced_summary": use_enhanced_summary,
                **audio_result.metadata
            }
        }

        self.update_status("SUCCESS", result)

        logger.info("Audio processing task completed",
                   task_id=task_id,
                   chunks_processed=len(processed_chunks),
                   total_text_length=len(audio_result.text))

        return result

    except Exception as e:
        error_msg = f"Audio processing failed: {str(e)}"
        logger.error("Audio processing task failed",
                    task_id=task_id,
                    error=str(e))

        self.update_status("FAILURE", {"error": error_msg})
        raise


@celery_app.task(bind=True, base=ProcessingTask)
def process_audio_file(
    self,
    file_path: str,
    task_id: str,
    config: Optional[Dict[str, Any]] = None,
    use_enhanced_summary: bool = True
) -> Dict[str, Any]:
    """Process audio file with speech-to-text and embedding generation."""
    return asyncio.run(_process_audio_file_impl(self, file_path, task_id, config, use_enhanced_summary))

async def _detect_audio_language_impl(
    self,
    file_path: str,
    task_id: str,
    model_size: str = "base"
) -> Dict[str, Any]:
    """Detect language of audio file."""

    logger.info("Starting audio language detection",
               task_id=task_id,
               file_path=file_path,
               model_size=model_size)

    try:
        self.update_status("PROCESSING", {"stage": "language_detection"})

        # Detect language
        result = await whisper_service.detect_language(
            audio_path=file_path,
            model_size=model_size
        )

        final_result = {
            "task_id": task_id,
            "status": "completed",
            "file_path": file_path,
            "language": result["language"],
            "language_probability": result["language_probability"],
            "all_language_probs": result["all_language_probs"]
        }

        self.update_status("SUCCESS", final_result)

        logger.info("Audio language detection completed",
                   task_id=task_id,
                   detected_language=result["language"],
                   probability=result["language_probability"])

        return final_result

    except Exception as e:
        error_msg = f"Audio language detection failed: {str(e)}"
        logger.error("Audio language detection task failed",
                    task_id=task_id,
                    error=str(e))

        self.update_status("FAILURE", {"error": error_msg})
        raise


@celery_app.task(bind=True, base=ProcessingTask)
def detect_audio_language(
    self,
    file_path: str,
    task_id: str,
    model_size: str = "base"
) -> Dict[str, Any]:
    """Detect language of audio file."""
    return asyncio.run(_detect_audio_language_impl(self, file_path, task_id, model_size))

async def _transcribe_audio_segments_impl(
    self,
    file_path: str,
    task_id: str,
    segments: List[Dict[str, float]],  # [{"start": 0.0, "end": 30.0}, ...]
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Transcribe specific segments of audio file."""

    logger.info("Starting audio segment transcription",
               task_id=task_id,
               file_path=file_path,
               segment_count=len(segments))

    try:
        self.update_status("PROCESSING", {"stage": "segment_transcription"})

        # Parse config
        audio_config = AudioConfig(**config) if config else AudioConfig()

        # Process each segment
        segment_results = []
        for i, segment in enumerate(segments):
            logger.debug("Processing audio segment",
                        task_id=task_id,
                        segment_index=i,
                        start_time=segment["start"],
                        end_time=segment["end"])

            try:
                # For now, transcribe the whole file and extract segment text
                # TODO: Implement actual segment extraction and transcription
                audio_result = await audio_processor.process_audio_file(
                    file_path=file_path,
                    config=audio_config
                )

                # Find segments within the time range
                segment_text_parts = []
                for audio_seg in audio_result.segments:
                    if (audio_seg.start_time >= segment["start"] and
                        audio_seg.end_time <= segment["end"]):
                        segment_text_parts.append(audio_seg.text)

                segment_text = " ".join(segment_text_parts)

                segment_results.append({
                    "segment_index": i,
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment_text,
                    "language": audio_result.language
                })

                # Update progress
                progress = (i + 1) / len(segments) * 100
                self.update_status("PROCESSING", {
                    "stage": "segment_transcription",
                    "progress": progress,
                    "segments_processed": i + 1,
                    "total_segments": len(segments)
                })

            except Exception as e:
                logger.error("Failed to process audio segment",
                           task_id=task_id,
                           segment_index=i,
                           error=str(e))
                segment_results.append({
                    "segment_index": i,
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": "",
                    "error": str(e)
                })

        result = {
            "task_id": task_id,
            "status": "completed",
            "file_path": file_path,
            "segments_processed": len(segment_results),
            "segment_results": segment_results
        }

        self.update_status("SUCCESS", result)

        logger.info("Audio segment transcription completed",
                   task_id=task_id,
                   segments_processed=len(segment_results))

        return result

    except Exception as e:
        error_msg = f"Audio segment transcription failed: {str(e)}"
        logger.error("Audio segment transcription task failed",
                    task_id=task_id,
                    error=str(e))

        self.update_status("FAILURE", {"error": error_msg})
        raise


@celery_app.task(bind=True, base=ProcessingTask)
def transcribe_audio_segments(
    self,
    file_path: str,
    task_id: str,
    segments: List[Dict[str, float]],  # [{"start": 0.0, "end": 30.0}, ...]
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Transcribe specific segments of audio file."""
    return asyncio.run(_transcribe_audio_segments_impl(self, file_path, task_id, segments, config))
