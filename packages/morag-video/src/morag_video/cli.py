"""Command-line interface for morag-video."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import structlog
from morag_video.processor import VideoConfig
from morag_video.service import VideoService

logger = structlog.get_logger(__name__)


async def process_video(
    file_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    output_format: str = "markdown",
    extract_audio: bool = True,
    generate_thumbnails: bool = True,
    extract_keyframes: bool = True,
    enable_enhanced_audio: bool = False,
    enable_speaker_diarization: bool = False,
    enable_topic_segmentation: bool = False,
    enable_ocr: bool = False,
    whisper_model: str = "base",
    language: Optional[str] = None,
    device: str = "auto",
):
    """Process a video file and save or print the results.

    Args:
        file_path: Path to the video file
        output_dir: Directory to save output files (if None, prints to console)
        output_format: Format for output (markdown, json, txt)
        extract_audio: Whether to extract audio from the video
        generate_thumbnails: Whether to generate thumbnails
        extract_keyframes: Whether to extract keyframes
        enable_enhanced_audio: Whether to enable enhanced audio processing
        enable_speaker_diarization: Whether to enable speaker diarization
        enable_topic_segmentation: Whether to enable topic segmentation
        enable_ocr: Whether to enable OCR on keyframes
        whisper_model: Whisper model size to use for transcription
        language: Language code for transcription (auto-detect if None)
        device: Device to use for processing (auto, cpu, cuda)
    """
    # Configure video processing
    config = VideoConfig(
        extract_audio=extract_audio,
        generate_thumbnails=generate_thumbnails,
        extract_keyframes=extract_keyframes,
        enable_enhanced_audio=enable_enhanced_audio,
        enable_speaker_diarization=enable_speaker_diarization,
        enable_topic_segmentation=enable_topic_segmentation,
        enable_ocr=enable_ocr,
        whisper_model=whisper_model,
        language=language,
        device=device,
    )

    # Initialize video service
    video_service = VideoService(config=config, output_dir=output_dir)

    try:
        # Process the video file
        result = await video_service.process_file(
            file_path=file_path,
            save_output=(output_dir is not None),
            output_format=output_format,
        )

        # If no output directory is specified, print results to console
        if not output_dir:
            if output_format == "markdown" and "markdown" in result:
                print(result["markdown"])
            else:
                import json

                print(json.dumps(result, indent=2))

        return result

    except Exception as e:
        logger.error("Video processing failed", error=str(e))
        raise


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Process video files with morag-video")

    # Input file arguments
    parser.add_argument(
        "input_files", nargs="+", help="Path(s) to video file(s) to process"
    )

    # Output arguments
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save output files (if not specified, prints to console)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "txt"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    # Processing options
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio extraction and transcription",
    )
    parser.add_argument(
        "--no-thumbnails", action="store_true", help="Disable thumbnail generation"
    )
    parser.add_argument(
        "--no-keyframes", action="store_true", help="Disable keyframe extraction"
    )

    # Enhanced audio processing options
    parser.add_argument(
        "--enhanced-audio", action="store_true", help="Enable enhanced audio processing"
    )
    parser.add_argument(
        "--diarization", action="store_true", help="Enable speaker diarization"
    )
    parser.add_argument(
        "--topic-segmentation", action="store_true", help="Enable topic segmentation"
    )

    # OCR options
    parser.add_argument("--ocr", action="store_true", help="Enable OCR on keyframes")

    # Whisper model options
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size to use for transcription (default: base)",
    )
    parser.add_argument(
        "--language",
        help="Language code for transcription (auto-detect if not specified)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for processing (default: auto)",
    )

    args = parser.parse_args()

    # Validate input files
    input_files = [Path(f) for f in args.input_files]
    invalid_files = [f for f in input_files if not f.exists()]

    if invalid_files:
        print(
            f"Error: The following files do not exist: {', '.join(str(f) for f in invalid_files)}"
        )
        sys.exit(1)

    # Create output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    # Process each input file
    for file_path in input_files:
        try:
            print(f"Processing {file_path}...")

            # Run the async processing function
            result = asyncio.run(
                process_video(
                    file_path=file_path,
                    output_dir=output_dir,
                    output_format=args.format,
                    extract_audio=not args.no_audio,
                    generate_thumbnails=not args.no_thumbnails,
                    extract_keyframes=not args.no_keyframes,
                    enable_enhanced_audio=args.enhanced_audio
                    or args.diarization
                    or args.topic_segmentation,
                    enable_speaker_diarization=args.diarization,
                    enable_topic_segmentation=args.topic_segmentation,
                    enable_ocr=args.ocr,
                    whisper_model=args.whisper_model,
                    language=args.language,
                    device=args.device,
                )
            )

            if output_dir:
                print(f"Results saved to {output_dir}")

            print(f"Successfully processed {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
