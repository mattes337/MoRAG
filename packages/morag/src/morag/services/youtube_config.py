"""YouTube configuration service for remote workers."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import structlog

logger = structlog.get_logger(__name__)

class YouTubeCookieManager:
    """Manage YouTube cookies for authenticated downloads."""

    def __init__(self, cookie_file_path: Optional[str] = None):
        self.cookie_file_path = cookie_file_path or os.getenv('YOUTUBE_COOKIES_FILE')

    def get_yt_dlp_options(self) -> Dict[str, Any]:
        """Get yt-dlp options with cookie configuration."""
        options = {
            # Basic options
            'format': 'best[height<=720]/best',
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US'],

            # Bot detection avoidance
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'referer': 'https://www.youtube.com/',
            'headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },

            # Retry configuration
            'retries': 3,
            'fragment_retries': 3,
            'force_ipv4': True,
        }

        # Add cookies if available
        if self.cookie_file_path and Path(self.cookie_file_path).exists():
            options['cookiefile'] = self.cookie_file_path
            logger.info("Using YouTube cookies", cookie_file=self.cookie_file_path)
        else:
            logger.warning("No YouTube cookies configured - some videos may be inaccessible")

        return options

    def validate_cookie_file(self) -> bool:
        """Validate that cookie file exists and is readable."""
        if not self.cookie_file_path:
            return False

        cookie_path = Path(self.cookie_file_path)
        if not cookie_path.exists():
            logger.error("Cookie file not found", path=self.cookie_file_path)
            return False

        try:
            # Try to read the file
            with open(cookie_path, 'r') as f:
                content = f.read(100)  # Read first 100 chars
                if not content.strip():
                    logger.error("Cookie file is empty", path=self.cookie_file_path)
                    return False

            logger.info("Cookie file validated", path=self.cookie_file_path)
            return True

        except Exception as e:
            logger.error("Failed to read cookie file", path=self.cookie_file_path, error=str(e))
            return False

    def get_cookie_instructions(self) -> str:
        """Get instructions for setting up YouTube cookies."""
        return """
YouTube Cookie Setup Instructions:

1. Install a browser extension to export cookies (e.g., "Get cookies.txt" for Chrome/Firefox)
2. Go to YouTube.com and log in to your account
3. Export cookies to a text file using the extension
4. Save the file and set YOUTUBE_COOKIES_FILE environment variable to the file path
5. Restart the worker to use the new cookies

Example cookie file format:
# Netscape HTTP Cookie File
.youtube.com	TRUE	/	FALSE	1234567890	session_token	abc123...

Note: Keep cookie files secure and don't share them. Cookies may expire and need periodic updates.
"""


class WebProcessingConfig:
    """Configuration for web content processing on remote workers."""

    def __init__(self):
        self.user_agent = os.getenv(
            'WEB_USER_AGENT',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        self.timeout = int(os.getenv('WEB_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('WEB_MAX_RETRIES', '3'))

    def get_requests_config(self) -> Dict[str, Any]:
        """Get requests configuration for web scraping."""
        return {
            'headers': {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            'timeout': self.timeout,
            'allow_redirects': True,
            'verify': True,
        }

    def get_selenium_config(self) -> Dict[str, Any]:
        """Get Selenium configuration for dynamic content."""
        return {
            'user_agent': self.user_agent,
            'timeout': self.timeout,
            'headless': True,
            'disable_images': True,
            'disable_javascript': False,  # Some sites need JS
        }


# Global instances
_youtube_cookie_manager: Optional[YouTubeCookieManager] = None
_web_processing_config: Optional[WebProcessingConfig] = None


def get_youtube_cookie_manager() -> YouTubeCookieManager:
    """Get or create YouTube cookie manager instance."""
    global _youtube_cookie_manager
    if _youtube_cookie_manager is None:
        _youtube_cookie_manager = YouTubeCookieManager()
    return _youtube_cookie_manager


def get_web_processing_config() -> WebProcessingConfig:
    """Get or create web processing config instance."""
    global _web_processing_config
    if _web_processing_config is None:
        _web_processing_config = WebProcessingConfig()
    return _web_processing_config
