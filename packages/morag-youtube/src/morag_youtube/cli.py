"""Command-line interface for morag-youtube package."""

import asyncio
import argparse
import sys
from pathlib import Path
import json
import structlog

from .processor import YouTubeProcessor, YouTubeConfig
from .service import YouTubeService

logger = structlog.get_logger(__name__)

def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="YouTube video transcription tool using external service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Global arguments
    parser.add_argument(
        "--service-url",
        type=str,
        help="URL of external YouTube transcription service (overrides YOUTUBE_SERVICE_URL env var)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for external service requests in seconds"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Transcribe command (replaces download)
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe YouTube video")
    transcribe_parser.add_argument("url", help="YouTube video URL")
    transcribe_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for downloaded video files (if --download-video is used)"
    )
    transcribe_parser.add_argument(
        "--download-video",
        action="store_true",
        help="Download video file in addition to transcription"
    )
    transcribe_parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="json",
        help="Output format for transcription results"
    )

    # Health check command
    health_parser = subparsers.add_parser("health", help="Check external service health")

    # Batch command for multiple videos
    batch_parser = subparsers.add_parser("batch", help="Process multiple YouTube videos")
    batch_parser.add_argument("urls", nargs="+", help="YouTube video URLs")
    batch_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for downloaded video files (if --download-video is used)"
    )
    batch_parser.add_argument(
        "--download-video",
        action="store_true",
        help="Download video files in addition to transcription"
    )
    batch_parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="json",
        help="Output format for transcription results"
    )
    batch_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum number of concurrent transcriptions"
    )

    return parser

def _create_config_from_args(args, **kwargs) -> YouTubeConfig:
    """Create YouTubeConfig from command line arguments."""
    config_kwargs = kwargs.copy()

    # Set external service options
    if hasattr(args, 'service_url') and args.service_url:
        config_kwargs['service_url'] = args.service_url
    if hasattr(args, 'timeout') and args.timeout:
        config_kwargs['service_timeout'] = args.timeout

    # Set download options
    if hasattr(args, 'download_video') and args.download_video:
        config_kwargs['download_video'] = True
    if hasattr(args, 'output_dir') and args.output_dir:
        config_kwargs['output_dir'] = args.output_dir

    return YouTubeConfig(**config_kwargs)

async def transcribe_video(args):
    """Transcribe YouTube video using external service."""
    service = YouTubeService(service_url=args.service_url, service_timeout=args.timeout)

    config = _create_config_from_args(args)

    try:
        print(f"Transcribing video: {args.url}")
        print("This may take several minutes...")

        result = await service.process_video(args.url, config)

        if result.success:
            print(f"\nTranscription completed successfully!")

            # Display metadata if available
            if result.metadata:
                print(f"Title: {result.metadata.title}")
                print(f"Duration: {result.metadata.duration} seconds")
                print(f"Uploader: {result.metadata.uploader}")

            # Display transcript info
            if result.transcript:
                print(f"Transcript available in {len(result.transcript_languages)} languages")
                if result.transcript.get('entries'):
                    print(f"Transcript segments: {len(result.transcript['entries'])}")

            # Display video download info if applicable
            if result.video_path:
                print(f"Video downloaded: {result.video_path}")

            # Save results based on format
            if args.output_format == "json":
                output_file = args.output_dir / f"{result.metadata.id if result.metadata else 'transcript'}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "metadata": result.metadata.__dict__ if result.metadata else None,
                        "transcript": result.transcript,
                        "transcript_languages": result.transcript_languages,
                        "video_path": str(result.video_path) if result.video_path else None
                    }, f, indent=2, ensure_ascii=False, default=str)
                print(f"Results saved to: {output_file}")
            else:
                # Text format - save transcript text
                if result.transcript_text:
                    output_file = args.output_dir / f"{result.metadata.id if result.metadata else 'transcript'}.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(result.transcript_text)
                    print(f"Transcript saved to: {output_file}")

            print(f"Processing time: {result.processing_time:.2f} seconds")
        else:
            print(f"Transcription failed: {result.error_message}")
            return 1

    except Exception as e:
        logger.exception("Error transcribing video", error=str(e))
        print(f"Error: {str(e)}")
        return 1

    return 0

async def check_health(args):
    """Check external service health."""
    service = YouTubeService(service_url=args.service_url, service_timeout=args.timeout)

    try:
        health = await service.health_check()

        print("\nService Health Check:")
        print(json.dumps(health, indent=2, ensure_ascii=False))

        if health["status"] == "healthy":
            print("\n✅ Service is healthy and ready")
            return 0
        else:
            print("\n❌ Service is unhealthy")
            return 1

    except Exception as e:
        logger.exception("Error checking service health", error=str(e))
        print(f"Error: {str(e)}")
        return 1

async def process_batch(args):
    """Process multiple YouTube videos."""
    service = YouTubeService(service_url=args.service_url, service_timeout=args.timeout)

    config = _create_config_from_args(args)

    try:
        print(f"Processing {len(args.urls)} videos...")
        print("This may take several minutes per video...")

        results = await service.process_videos(args.urls, config)

        successful = 0
        failed = 0

        for i, result in enumerate(results):
            url = args.urls[i]
            if isinstance(result, Exception):
                print(f"❌ Failed: {url} - {str(result)}")
                failed += 1
            else:
                if result.success:
                    print(f"✅ Success: {url}")
                    if result.metadata:
                        print(f"   Title: {result.metadata.title}")
                    if result.video_path:
                        print(f"   Video: {result.video_path}")
                    successful += 1
                else:
                    print(f"❌ Failed: {url} - {result.error_message}")
                    failed += 1

        print(f"\nBatch processing completed:")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")

        return 0 if failed == 0 else 1

    except Exception as e:
        logger.exception("Error processing batch", error=str(e))
        print(f"Error: {str(e)}")
        return 1
async def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Set up logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Route to appropriate command handler
    if args.command == "transcribe":
        return await transcribe_video(args)
    elif args.command == "health":
        return await check_health(args)
    elif args.command == "batch":
        return await process_batch(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        return 1

def cli_main():
    """Entry point for console script."""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(cli_main())
