"""Graphiti connection service for MoRAG integration."""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import structlog

from .config import GraphitiConfig, create_graphiti_instance

logger = structlog.get_logger(__name__)


class GraphitiConnectionService:
    """Service for managing Graphiti connections and basic operations."""

    def __init__(self, config: Optional[GraphitiConfig] = None):
        """Initialize the Graphiti connection service.
        
        Args:
            config: Optional Graphiti configuration. If None, loads from environment.
        """
        self.config = config
        self._graphiti = None
        self._connected = False

    async def connect(self) -> bool:
        """Establish connection to Graphiti.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._graphiti = create_graphiti_instance(self.config)
            
            # Test the connection by attempting to create a simple episode
            await self.validate_connection()
            
            self._connected = True
            logger.info("Graphiti connection established successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to connect to Graphiti", error=str(e))
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Graphiti."""
        if self._graphiti and hasattr(self._graphiti, 'close'):
            try:
                await self._graphiti.close()
            except Exception as e:
                logger.warning("Error during Graphiti disconnect", error=str(e))
        
        self._graphiti = None
        self._connected = False
        logger.info("Graphiti connection closed")

    async def validate_connection(self) -> bool:
        """Validate the Graphiti connection by performing a basic operation.
        
        Returns:
            True if connection is valid, False otherwise
        """
        if not self._graphiti:
            return False
            
        try:
            # Create a test episode to validate connection
            test_episode_name = f"connection_test_{datetime.now().isoformat()}"
            test_content = "This is a connection test episode."
            
            # Add episode to Graphiti with correct parameters
            await self._graphiti.add_episode(
                name=test_episode_name,
                episode_body=test_content,
                source_description="MoRAG connection test",
                reference_time=datetime.now()
            )
            
            logger.debug("Connection validation successful", episode_name=test_episode_name)
            return True
            
        except Exception as e:
            logger.error("Connection validation failed", error=str(e))
            return False

    async def create_episode(
        self,
        name: str,
        content: str,
        source_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reference_time: Optional[datetime] = None
    ) -> bool:
        """Create a new episode in Graphiti.
        
        Args:
            name: Episode name/identifier
            content: Episode content
            source_description: Optional source description
            metadata: Optional metadata dictionary
            
        Returns:
            True if episode created successfully, False otherwise
        """
        if not self._connected or not self._graphiti:
            logger.error("Not connected to Graphiti")
            return False
            
        try:
            if reference_time is None:
                reference_time = datetime.now()

            await self._graphiti.add_episode(
                name=name,
                episode_body=content,  # Use episode_body instead of content
                source_description=source_description or "MoRAG episode",
                reference_time=reference_time
            )

            logger.info("Episode created successfully", episode_name=name)
            return True

        except Exception as e:
            error_msg = str(e)

            # Check for specific JSON parsing errors that might be recoverable
            if "Unterminated string" in error_msg or "Failed to parse structured response" in error_msg:
                logger.warning(
                    "Episode creation failed due to LLM response parsing error",
                    episode_name=name,
                    error=error_msg,
                    content_preview=content[:200] if content else "None"
                )

                # Try to create episode with simplified content to avoid LLM processing issues
                try:
                    # Create a simplified version of the content that's less likely to cause parsing issues
                    simplified_content = self._simplify_content_for_retry(content)

                    await self._graphiti.add_episode(
                        name=f"{name}_simplified",
                        episode_body=simplified_content,
                        source_description=f"Simplified version - {source_description or 'MoRAG episode'}",
                        reference_time=reference_time
                    )

                    logger.info("Episode created successfully with simplified content", episode_name=f"{name}_simplified")
                    return True

                except Exception as retry_error:
                    logger.error(
                        "Failed to create episode even with simplified content",
                        episode_name=name,
                        original_error=error_msg,
                        retry_error=str(retry_error)
                    )
            else:
                logger.error("Failed to create episode", episode_name=name, error=error_msg)

            return False

    def _simplify_content_for_retry(self, content: str) -> str:
        """Simplify content to reduce the likelihood of LLM parsing errors.

        Args:
            content: Original content that may be causing parsing issues

        Returns:
            Simplified content that's less likely to cause LLM errors
        """
        if not content:
            return content

        # Remove or escape characters that commonly cause JSON parsing issues
        simplified = content

        # Replace problematic characters
        simplified = simplified.replace('"', "'")  # Replace double quotes with single quotes
        simplified = simplified.replace('\n', ' ')  # Replace newlines with spaces
        simplified = simplified.replace('\r', ' ')  # Replace carriage returns
        simplified = simplified.replace('\t', ' ')  # Replace tabs with spaces

        # Remove excessive whitespace
        import re
        simplified = re.sub(r'\s+', ' ', simplified)

        # Truncate if too long (very long content can cause LLM issues)
        if len(simplified) > 2000:
            simplified = simplified[:2000] + "... [content truncated for processing]"

        # Remove any remaining problematic characters
        simplified = re.sub(r'[^\x20-\x7E ]', '', simplified)  # Keep only printable ASCII + space

        return simplified.strip()

    async def search_episodes(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for episodes in Graphiti.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of episode dictionaries
        """
        if not self._connected or not self._graphiti:
            logger.error("Not connected to Graphiti")
            return []
            
        try:
            # Use the correct Graphiti search API
            results = await self._graphiti.search(query=query)

            # Apply limit manually since Graphiti core doesn't support it directly
            if limit and len(results) > limit:
                results = results[:limit]

            logger.info("Episode search completed", query=query, result_count=len(results))
            return results
            
        except Exception as e:
            logger.error("Failed to search episodes", query=query, error=str(e))
            return []

    @property
    def is_connected(self) -> bool:
        """Check if connected to Graphiti."""
        return self._connected

    @property
    def graphiti_instance(self):
        """Get the underlying Graphiti instance."""
        return self._graphiti

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Convenience function for creating connection service
def create_connection_service(config: Optional[GraphitiConfig] = None) -> GraphitiConnectionService:
    """Create a Graphiti connection service.
    
    Args:
        config: Optional Graphiti configuration
        
    Returns:
        GraphitiConnectionService instance
    """
    return GraphitiConnectionService(config)
