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
        description="YouTube video processing tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download YouTube video")
    download_parser.add_argument("url", help="YouTube video URL")
    download_parser.add_argument(
        "-o", "--output-dir", 
        type=Path, 
        default=Path.cwd(),
        help="Output directory for downloaded files"
    )
    download_parser.add_argument(
        "-q", "--quality", 
        default="best",
        help="Video quality (best, worst, or specific format)"
    )
    download_parser.add_argument(
        "--no-audio", 
        action="store_true",
        help="Don't extract audio"
    )
    download_parser.add_argument(
        "--no-subtitles", 
        action="store_true",
        help="Don't download subtitles"
    )
    download_parser.add_argument(
        "--subtitle-langs", 
        default="en",
        help="Subtitle languages (comma-separated)"
    )
    
    # Audio command
    audio_parser = subparsers.add_parser("audio", help="Download only audio from YouTube video")
    audio_parser.add_argument("url", help="YouTube video URL")
    audio_parser.add_argument(
        "-o", "--output-dir", 
        type=Path, 
        default=Path.cwd(),
        help="Output directory for downloaded files"
    )
    
    # Metadata command
    metadata_parser = subparsers.add_parser("metadata", help="Extract metadata from YouTube video")
    metadata_parser.add_argument("url", help="YouTube video URL")
    metadata_parser.add_argument(
        "-o", "--output-file", 
        type=Path, 
        help="Output file for metadata (JSON format)"
    )
    
    # Playlist command
    playlist_parser = subparsers.add_parser("playlist", help="Process YouTube playlist")
    playlist_parser.add_argument("url", help="YouTube playlist URL")
    playlist_parser.add_argument(
        "-o", "--output-dir", 
        type=Path, 
        default=Path.cwd(),
        help="Output directory for downloaded files"
    )
    playlist_parser.add_argument(
        "-q", "--quality", 
        default="best",
        help="Video quality (best, worst, or specific format)"
    )
    playlist_parser.add_argument(
        "--no-audio", 
        action="store_true",
        help="Don't extract audio"
    )
    playlist_parser.add_argument(
        "--no-subtitles", 
        action="store_true",
        help="Don't download subtitles"
    )
    
    # Thumbnail command
    thumbnail_parser = subparsers.add_parser("thumbnail", help="Download thumbnail from YouTube video")
    thumbnail_parser.add_argument("url", help="YouTube video URL")
    thumbnail_parser.add_argument(
        "-o", "--output-dir", 
        type=Path, 
        default=Path.cwd(),
        help="Output directory for downloaded files"
    )
    
    # Subtitles command
    subtitles_parser = subparsers.add_parser("subtitles", help="Download subtitles from YouTube video")
    subtitles_parser.add_argument("url", help="YouTube video URL")
    subtitles_parser.add_argument(
        "-o", "--output-dir", 
        type=Path, 
        default=Path.cwd(),
        help="Output directory for downloaded files"
    )
    subtitles_parser.add_argument(
        "--langs", 
        default="en",
        help="Subtitle languages (comma-separated)"
    )
    
    return parser

async def download_video(args):
    """Download YouTube video."""
    service = YouTubeService()
    
    config = YouTubeConfig(
        quality=args.quality,
        extract_audio=not args.no_audio,
        download_subtitles=not args.no_subtitles,
        subtitle_languages=args.subtitle_langs.split(",")
    )
    
    try:
        result = await service.download_video(
            args.url,
            output_dir=args.output_dir,
            quality=args.quality,
            extract_audio=not args.no_audio,
            download_subtitles=not args.no_subtitles
        )
        
        print(f"\nDownload completed successfully!")
        print(f"Video: {result.video_path}")
        if result.audio_path:
            print(f"Audio: {result.audio_path}")
        if result.subtitle_paths:
            print(f"Subtitles: {', '.join(str(p) for p in result.subtitle_paths)}")
        if result.thumbnail_paths:
            print(f"Thumbnails: {', '.join(str(p) for p in result.thumbnail_paths)}")
        print(f"Total size: {result.file_size / (1024*1024):.2f} MB")
        print(f"Processing time: {result.processing_time:.2f} seconds")
        
    except Exception as e:
        logger.exception("Error downloading video", error=str(e))
        print(f"Error: {str(e)}")
        return 1
    
    return 0

async def download_audio(args):
    """Download audio from YouTube video."""
    service = YouTubeService()
    
    try:
        audio_path = await service.download_audio(args.url, output_dir=args.output_dir)
        print(f"\nAudio downloaded successfully: {audio_path}")
        
    except Exception as e:
        logger.exception("Error downloading audio", error=str(e))
        print(f"Error: {str(e)}")
        return 1
    
    return 0

async def extract_metadata(args):
    """Extract metadata from YouTube video."""
    service = YouTubeService()
    
    try:
        metadata = await service.extract_metadata(args.url)
        
        # Pretty print metadata
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Metadata saved to {args.output_file}")
        else:
            print("\nVideo Metadata:")
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.exception("Error extracting metadata", error=str(e))
        print(f"Error: {str(e)}")
        return 1
    
    return 0

async def process_playlist(args):
    """Process YouTube playlist."""
    service = YouTubeService()
    
    config = YouTubeConfig(
        quality=args.quality,
        extract_audio=not args.no_audio,
        download_subtitles=not args.no_subtitles
    )
    
    try:
        print(f"Processing playlist: {args.url}")
        print("This may take some time depending on the playlist size...")
        
        results = await service.process_playlist(args.url, config)
        
        # Move files to output directory
        output_dir = args.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        
        success_count = 0
        failed_count = 0
        
        for i, result in enumerate(results):
            if result.success:
                success_count += 1
                
                # Move video file
                if result.video_path and result.video_path.exists():
                    new_video_path = output_dir / result.video_path.name
                    result.video_path.rename(new_video_path)
                
                # Move audio file
                if result.audio_path and result.audio_path.exists():
                    new_audio_path = output_dir / result.audio_path.name
                    result.audio_path.rename(new_audio_path)
                
                # Move subtitle files
                for sub_path in result.subtitle_paths:
                    if sub_path.exists():
                        new_sub_path = output_dir / sub_path.name
                        sub_path.rename(new_sub_path)
                
                # Move thumbnail files
                for thumb_path in result.thumbnail_paths:
                    if thumb_path.exists():
                        new_thumb_path = output_dir / thumb_path.name
                        thumb_path.rename(new_thumb_path)
            else:
                failed_count += 1
        
        print(f"\nPlaylist processing completed!")
        print(f"Successfully processed: {success_count} videos")
        print(f"Failed: {failed_count} videos")
        print(f"Files saved to: {output_dir}")
        
    except Exception as e:
        logger.exception("Error processing playlist", error=str(e))
        print(f"Error: {str(e)}")
        return 1
    
    return 0

async def download_thumbnail(args):
    """Download thumbnail from YouTube video."""
    service = YouTubeService()
    
    try:
        thumbnail_path = await service.download_thumbnail(args.url, output_dir=args.output_dir)
        print(f"\nThumbnail downloaded successfully: {thumbnail_path}")
        
    except Exception as e:
        logger.exception("Error downloading thumbnail", error=str(e))
        print(f"Error: {str(e)}")
        return 1
    
    return 0

async def download_subtitles(args):
    """Download subtitles from YouTube video."""
    service = YouTubeService()
    
    try:
        languages = args.langs.split(",")
        subtitle_paths = await service.download_subtitles(
            args.url, 
            languages=languages,
            output_dir=args.output_dir
        )
        
        if subtitle_paths:
            print(f"\nSubtitles downloaded successfully:")
            for path in subtitle_paths:
                print(f"- {path}")
        else:
            print(f"\nNo subtitles found for the requested languages: {languages}")
        
    except Exception as e:
        logger.exception("Error downloading subtitles", error=str(e))
        print(f"Error: {str(e)}")
        return 1
    
    return 0

async def main_async():
    """Async entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    command_handlers = {
        "download": download_video,
        "audio": download_audio,
        "metadata": extract_metadata,
        "playlist": process_playlist,
        "thumbnail": download_thumbnail,
        "subtitles": download_subtitles,
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        return await handler(args)
    else:
        parser.print_help()
        return 0

def main():
    """Entry point for the CLI."""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())