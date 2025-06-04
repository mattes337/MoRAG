"""Command-line interface for morag-audio."""

import os
import sys
import asyncio
import argparse
from pathlib import Path
import structlog

from morag_audio import AudioProcessor, AudioConfig, AudioService

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)

logger = structlog.get_logger()


async def process_audio(args):
    """Process audio files based on command-line arguments."""
    # Create configuration
    config = AudioConfig(
        model_size=args.model_size,
        language=args.language,
        enable_diarization=args.diarization,
        enable_topic_segmentation=args.topic_segmentation,
        device=args.device,
        vad_filter=not args.no_vad,
        word_timestamps=True
    )
    
    # Create service with output directory if specified
    output_dir = Path(args.output_dir) if args.output_dir else None
    service = AudioService(config=config, output_dir=output_dir)
    
    # Process each file
    results = []
    for file_path in args.files:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error("File not found", file_path=str(file_path))
            continue
        
        logger.info("Processing audio file", file_path=str(file_path))
        
        try:
            result = await service.process_file(
                file_path, 
                save_output=args.output_dir is not None,
                output_format=args.format
            )
            
            results.append((file_path, result))
            
            if result["success"]:
                logger.info("Processing completed successfully", 
                           file_path=str(file_path),
                           processing_time=result["processing_time"])
                
                if args.output_dir:
                    logger.info("Output files:", files=result["output_files"])
                elif args.format == "markdown":
                    print(f"\n{result.get('content', '')}\n")
                else:
                    print(f"\nTranscript:\n{result.get('content', '')}\n")
            else:
                logger.error("Processing failed", 
                            file_path=str(file_path),
                            error=result.get("error", "Unknown error"))
        
        except Exception as e:
            logger.error("Error processing file", 
                        file_path=str(file_path),
                        error=str(e))
    
    # Print summary
    if len(results) > 1:
        print("\nProcessing Summary:")
        for file_path, result in results:
            status = "✓" if result["success"] else "✗"
            print(f"{status} {file_path.name} - {result.get('processing_time', 0):.2f}s")


def main():
    """Parse arguments and run the CLI."""
    parser = argparse.ArgumentParser(description="MoRAG Audio Processing Tool")
    
    parser.add_argument("files", nargs="+", help="Audio files to process")
    parser.add_argument("--output-dir", "-o", help="Directory to save output files")
    parser.add_argument("--format", "-f", choices=["markdown", "txt"], default="markdown",
                        help="Output format (default: markdown)")
    
    # Model configuration
    parser.add_argument("--model-size", choices=["tiny", "base", "small", "medium", "large-v2"],
                        default="medium", help="Whisper model size (default: medium)")
    parser.add_argument("--language", "-l", help="Language code (auto-detect if not specified)")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                        help="Device to use for processing (default: auto)")
    
    # Feature flags
    parser.add_argument("--diarization", "-d", action="store_true",
                        help="Enable speaker diarization")
    parser.add_argument("--topic-segmentation", "-t", action="store_true",
                        help="Enable topic segmentation")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable voice activity detection")
    
    args = parser.parse_args()
    
    # Run the async function
    asyncio.run(process_audio(args))


if __name__ == "__main__":
    main()