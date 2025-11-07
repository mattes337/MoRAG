"""Apify YouTube transcription service client."""

import os
from typing import Any, Dict, List, Optional

import structlog
from apify_client import ApifyClient
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


class ApifyYouTubeServiceError(ProcessingError):
    """Exception raised when Apify YouTube service fails."""


class ApifyYouTubeService:
    """Client for Apify YouTube transcription and metadata extraction service."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        actor_id: Optional[str] = None,
        timeout: int = 600,
    ):
        """Initialize the Apify service client.

        Args:
            api_token: Apify API token (defaults to env var)
            actor_id: Apify actor ID (defaults to env var)
            timeout: Request timeout in seconds (default: 10 minutes)
        """
        self.api_token = api_token or os.getenv("APIFY_API_TOKEN")
        self.actor_id = actor_id or os.getenv(
            "APIFY_YOUTUBE_ACTOR_ID", "SWvgAAm9FpfWHRrUm"
        )
        self.timeout = timeout

        if not self.api_token:
            raise ApifyYouTubeServiceError(
                "Apify API token is required. Set APIFY_API_TOKEN environment variable."
            )

        self.client = ApifyClient(self.api_token)

    async def health_check(self) -> bool:
        """Check if the Apify service is available.

        Returns:
            True if service is available, False otherwise
        """
        try:
            # Try to get actor info to verify connection
            actor_info = self.client.actor(self.actor_id).get()
            return actor_info is not None
        except Exception as e:
            logger.warning("Apify health check failed", error=str(e))
            return False

    async def check_apify_health(self) -> bool:
        """Alias for health_check for consistency with other services.

        Returns:
            True if service is available, False otherwise
        """
        return await self.health_check()

    async def transcribe_videos(
        self,
        video_urls: List[str],
        extract_metadata: bool = True,
        extract_transcript: bool = True,
        use_proxy: bool = True,
    ) -> List[Dict[str, Any]]:
        """Transcribe multiple YouTube videos using Apify service.

        Args:
            video_urls: List of YouTube video URLs
            extract_metadata: Whether to extract video metadata
            extract_transcript: Whether to extract video transcript
            use_proxy: Whether to use Apify proxy

        Returns:
            List of dictionaries containing transcription results and metadata

        Raises:
            ApifyYouTubeServiceError: If the service request fails
        """
        try:
            logger.info(
                "Starting Apify YouTube transcription",
                video_count=len(video_urls),
                extract_metadata=extract_metadata,
                extract_transcript=extract_transcript,
            )

            # Prepare actor input
            run_input = {
                "videoUrls": video_urls,
                "extractMetadata": extract_metadata,
                "extractTranscript": extract_transcript,
            }

            if use_proxy:
                run_input["proxyConfiguration"] = {"useApifyProxy": True}

            # Run the actor and wait for completion
            logger.debug(
                "Sending request to Apify actor",
                actor_id=self.actor_id,
                input_data=run_input,
            )

            run = self.client.actor(self.actor_id).call(run_input=run_input)

            if not run or "defaultDatasetId" not in run:
                raise ApifyYouTubeServiceError(
                    "Apify actor run failed or returned invalid response"
                )

            # Fetch results from the dataset
            results = []
            dataset = self.client.dataset(run["defaultDatasetId"])

            for item in dataset.iterate_items():
                results.append(item)

            logger.info(
                "Apify YouTube transcription completed successfully",
                video_count=len(video_urls),
                results_count=len(results),
            )

            return results

        except Exception as e:
            logger.error("Apify YouTube transcription failed", error=str(e))
            raise ApifyYouTubeServiceError(
                f"Apify service request failed: {str(e)}"
            ) from e

    async def transcribe_video(
        self,
        video_url: str,
        extract_metadata: bool = True,
        extract_transcript: bool = True,
        use_proxy: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe a single YouTube video using Apify service.

        Args:
            video_url: YouTube video URL
            extract_metadata: Whether to extract video metadata
            extract_transcript: Whether to extract video transcript
            use_proxy: Whether to use Apify proxy

        Returns:
            Dictionary containing transcription results and metadata

        Raises:
            ApifyYouTubeServiceError: If the service request fails
        """
        results = await self.transcribe_videos(
            [video_url],
            extract_metadata=extract_metadata,
            extract_transcript=extract_transcript,
            use_proxy=use_proxy,
        )

        if not results:
            raise ApifyYouTubeServiceError(
                f"No results returned for video: {video_url}"
            )

        return results[0]

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL for logging purposes.

        Args:
            url: YouTube video URL

        Returns:
            Video ID string
        """
        import re
        from urllib.parse import parse_qs, urlparse

        # Handle different YouTube URL formats
        if "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        elif "youtube.com/watch" in url:
            parsed = urlparse(url)
            return parse_qs(parsed.query).get("v", ["unknown"])[0]
        elif "youtube.com/embed/" in url:
            return url.split("youtube.com/embed/")[1].split("?")[0]
        else:
            # Try to extract 11-character video ID pattern
            match = re.search(r"[a-zA-Z0-9_-]{11}", url)
            return match.group(0) if match else "unknown"
