"""Example usage of morag-youtube package for YouTube video processing."""

import asyncio
import os
from pathlib import Path
import json

from morag_youtube.processor import YouTubeProcessor, YouTubeConfig
from morag_youtube.service import YouTubeService

async def basic_example():
    """Basic example of using YouTubeProcessor directly."""
    print("\n=== Basic YouTube Processing Example ===")
    
    # Initialize processor
    processor = YouTubeProcessor()
    
    # Configure processing options
    config = YouTubeConfig(
        quality="best",
        extract_audio=True,
        download_subtitles=True,
        subtitle_languages=["en"],
        download_thumbnails=True
    )
    
    # Process a YouTube video
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example URL
    print(f"Processing video: {video_url}")
    
    result = await processor.process_url(video_url, config)
    
    if result.success:
        print("\nProcessing successful!")
        print(f"Video title: {result.metadata.title}")
        print(f"Video file: {result.video_path}")
        if result.audio_path:
            print(f"Audio file: {result.audio_path}")
        if result.subtitle_paths:
            print(f"Subtitle files: {', '.join(str(p) for p in result.subtitle_paths)}")
        if result.thumbnail_paths:
            print(f"Thumbnail files: {', '.join(str(p) for p in result.thumbnail_paths)}")
        print(f"Total size: {result.file_size / (1024*1024):.2f} MB")
        print(f"Processing time: {result.processing_time:.2f} seconds")
        
        # Print some metadata
        print("\nVideo metadata:")
        print(f"  Duration: {result.metadata.duration:.2f} seconds")
        print(f"  View count: {result.metadata.view_count}")
        print(f"  Upload date: {result.metadata.upload_date}")
        print(f"  Channel: {result.metadata.uploader} ({result.metadata.channel_url})")
        
        # Clean up temporary files
        print("\nCleaning up temporary files...")
        processor.cleanup(result)
    else:
        print(f"Processing failed: {result.error_message}")

async def service_example():
    """Example of using YouTubeService for higher-level operations."""
    print("\n=== YouTube Service Example ===")
    
    # Initialize service
    service = YouTubeService(max_concurrent_downloads=3)
    
    # Create output directory
    output_dir = Path("./youtube_downloads")
    output_dir.mkdir(exist_ok=True)
    
    # Example 1: Download video with custom options
    print("\n1. Downloading video with custom options")
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example URL
    
    try:
        result = await service.download_video(
            video_url,
            output_dir=output_dir,
            quality="best",
            extract_audio=True,
            download_subtitles=True
        )
        
        print(f"Video downloaded: {result.video_path}")
        if result.audio_path:
            print(f"Audio extracted: {result.audio_path}")
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
    
    # Example 2: Extract metadata only
    print("\n2. Extracting video metadata")
    try:
        metadata = await service.extract_metadata(video_url)
        print("Metadata extracted:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False)[:500] + "...")
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
    
    # Example 3: Download only audio
    print("\n3. Downloading only audio")
    try:
        audio_path = await service.download_audio(video_url, output_dir=output_dir)
        print(f"Audio downloaded: {audio_path}")
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")
    
    # Example 4: Download only thumbnail
    print("\n4. Downloading thumbnail")
    try:
        thumbnail_path = await service.download_thumbnail(video_url, output_dir=output_dir)
        print(f"Thumbnail downloaded: {thumbnail_path}")
    except Exception as e:
        print(f"Error downloading thumbnail: {str(e)}")
    
    # Example 5: Process multiple videos concurrently
    print("\n5. Processing multiple videos concurrently")
    video_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",  # Gangnam Style
    ]
    
    try:
        # Configure to only extract metadata to save time
        config = YouTubeConfig(extract_metadata_only=True)
        results = await service.process_videos(video_urls, config)
        
        print(f"Processed {len(results)} videos:")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  Video {i+1}: Error - {str(result)}")
            elif result.success:
                print(f"  Video {i+1}: {result.metadata.title} ({result.metadata.duration:.2f} seconds)")
            else:
                print(f"  Video {i+1}: Failed - {result.error_message}")
    except Exception as e:
        print(f"Error processing videos: {str(e)}")

async def playlist_example():
    """Example of processing a YouTube playlist."""
    print("\n=== YouTube Playlist Example ===")
    
    # Initialize service
    service = YouTubeService()
    
    # Example playlist URL (short playlist for demo purposes)
    playlist_url = "https://www.youtube.com/playlist?list=PLlaN88a7y2_plecYoJxvRFTLHVbIVAOoS"  # Example URL
    
    try:
        # Configure to only extract metadata to save time
        config = YouTubeConfig(extract_metadata_only=True)
        
        print(f"Processing playlist: {playlist_url}")
        print("This may take some time depending on the playlist size...")
        
        results = await service.process_playlist(playlist_url, config)
        
        print(f"\nProcessed {len(results)} videos in playlist:")
        for i, result in enumerate(results):
            if result.success:
                print(f"  {i+1}. {result.metadata.title} ({result.metadata.duration:.2f} seconds)")
            else:
                print(f"  {i+1}. Failed: {result.error_message}")
    except Exception as e:
        print(f"Error processing playlist: {str(e)}")

async def main():
    """Run all examples."""
    print("YouTube Processing Examples")
    print("===========================")
    
    # Run basic example
    await basic_example()
    
    # Run service example
    await service_example()
    
    # Run playlist example
    # Note: This is commented out by default as it may take longer
    # await playlist_example()
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    asyncio.run(main())