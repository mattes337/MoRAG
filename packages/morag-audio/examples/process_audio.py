"""Example script for processing audio files with morag-audio."""

import argparse
import asyncio
from pathlib import Path

import structlog
from morag_audio import AudioConfig, AudioProcessor, AudioService

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)

logger = structlog.get_logger()


async def process_audio_file(
    file_path: str, output_dir: str = None, enable_diarization: bool = False
):
    """Process an audio file and print the results."""
    try:
        # Create configuration
        config = AudioConfig(
            model_size="medium",  # Options: tiny, base, small, medium, large-v2
            enable_diarization=enable_diarization,
            enable_topic_segmentation=True,
            device="auto",  # Will use GPU if available
            vad_filter=True,  # Voice activity detection
            word_timestamps=True,
        )

        # Create service with output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            service = AudioService(config=config, output_dir=output_path)
            logger.info(
                "Processing audio file",
                file_path=file_path,
                output_dir=str(output_path),
            )

            # Process file and save results
            result = await service.process_file(
                file_path, save_output=True, output_format="markdown"
            )

            if result["success"]:
                logger.info("Processing completed successfully")
                logger.info("Output files:", files=result["output_files"])
            else:
                logger.error(
                    "Processing failed", error=result.get("error", "Unknown error")
                )
        else:
            # Use processor directly without saving files
            processor = AudioProcessor(config=config)
            logger.info("Processing audio file", file_path=file_path)

            result = await processor.process(file_path)

            if result.success:
                logger.info("Processing completed successfully")
                print(f"\nTranscript:\n{result.transcript}\n")

                if enable_diarization:
                    print("\nSpeaker segments:")
                    for segment in result.segments:
                        speaker = segment.speaker or "Unknown"
                        print(
                            f"[{segment.start:.2f} - {segment.end:.2f}] {speaker}: {segment.text}"
                        )
            else:
                logger.error("Processing failed", error=result.error_message)

    except Exception as e:
        logger.error("Error processing audio file", error=str(e))


def main():
    """Parse arguments and run the example."""
    parser = argparse.ArgumentParser(description="Process audio files with morag-audio")
    parser.add_argument("file_path", help="Path to the audio file to process")
    parser.add_argument("--output-dir", "-o", help="Directory to save output files")
    parser.add_argument(
        "--diarization", "-d", action="store_true", help="Enable speaker diarization"
    )

    args = parser.parse_args()

    # Run the async function
    asyncio.run(
        process_audio_file(
            file_path=args.file_path,
            output_dir=args.output_dir,
            enable_diarization=args.diarization,
        )
    )


if __name__ == "__main__":
    main()
