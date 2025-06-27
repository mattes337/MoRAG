"""AI model providers for MoRAG agents."""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class ProviderConfig(BaseModel):
    """Configuration for AI providers."""
    
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for the provider API")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    extra_headers: Dict[str, str] = Field(default_factory=dict, description="Extra headers for requests")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Extra parameters for requests")


class GeminiProvider:
    """Gemini provider for PydanticAI agents."""
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize the Gemini provider.
        
        Args:
            config: Provider configuration
        """
        self.config = config or ProviderConfig()
        self.logger = logger.bind(provider="gemini")
        
        # Get API key from config or environment
        self.api_key = self.config.api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            self.logger.warning("No Gemini API key found in config or GEMINI_API_KEY environment variable")
        
        self._configure_provider()
    
    def _configure_provider(self):
        """Configure the Gemini provider."""
        try:
            import google.generativeai as genai
            
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.logger.info("Gemini provider configured successfully")
            else:
                self.logger.warning("Gemini provider not configured - no API key available")
                
        except ImportError as e:
            self.logger.error("Failed to import google.generativeai", error=str(e))
            raise ImportError(
                "google-generativeai is required for Gemini provider. "
                "Install it with: pip install google-generativeai"
            ) from e
        except Exception as e:
            self.logger.error("Failed to configure Gemini provider", error=str(e))
            raise
    
    def get_model_name(self, model: str) -> str:
        """Get the full model name for Gemini.
        
        Args:
            model: The model identifier
            
        Returns:
            The full model name for Gemini
        """
        # Handle different model name formats
        if model.startswith("google-gla:"):
            return model
        elif model.startswith("gemini"):
            return f"google-gla:{model}"
        else:
            # Default to Gemini 1.5 Flash if not specified
            return "google-gla:gemini-1.5-flash"
    
    def is_available(self) -> bool:
        """Check if the Gemini provider is available.
        
        Returns:
            True if the provider is available, False otherwise
        """
        try:
            import google.generativeai as genai
            return self.api_key is not None
        except ImportError:
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider.
        
        Returns:
            Dictionary with provider information
        """
        return {
            "name": "gemini",
            "available": self.is_available(),
            "api_key_configured": self.api_key is not None,
            "config": self.config.model_dump(exclude={"api_key"})  # Exclude sensitive data
        }


class ProviderFactory:
    """Factory for creating AI providers."""
    
    _providers = {
        "gemini": GeminiProvider,
    }
    
    @classmethod
    def create_provider(
        self,
        provider_name: str,
        config: Optional[ProviderConfig] = None
    ) -> GeminiProvider:
        """Create a provider instance.
        
        Args:
            provider_name: Name of the provider
            config: Provider configuration
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If the provider is not supported
        """
        if provider_name not in self._providers:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Supported providers: {list(self._providers.keys())}"
            )
        
        provider_class = self._providers[provider_name]
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """Get available providers and their availability status.
        
        Returns:
            Dictionary mapping provider names to availability status
        """
        availability = {}
        for name, provider_class in cls._providers.items():
            try:
                provider = provider_class()
                availability[name] = provider.is_available()
            except Exception:
                availability[name] = False
        
        return availability
