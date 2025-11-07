"""AI model providers for MoRAG agents."""

import os
from typing import Optional, Dict, Any, Type, TypeVar, Union
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


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


class OutlinesProvider:
    """Outlines provider for structured generation with guaranteed valid outputs."""

    def __init__(self, config: Optional[ProviderConfig] = None, provider: str = "gemini", model: str = "gemini-1.5-flash"):
        """Initialize the Outlines provider.

        Args:
            config: Provider configuration
            provider: The underlying provider to use (gemini, openai)
            model: The model identifier
        """
        self.config = config or ProviderConfig()
        self.provider = provider
        self.model = model
        self.logger = logger.bind(provider="outlines", underlying_provider=provider)

        # Get API key from config or environment
        self.api_key = self.config.api_key or self._get_api_key_for_provider(provider)
        if not self.api_key:
            self.logger.warning(f"No API key found for provider {provider}")

        self._outlines_model = None

    def _get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get API key for the specified provider from environment."""
        if provider == "gemini":
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        return None

    def get_generator(self, output_type: Type[T]):
        """Get an Outlines generator for the specified output type.

        Args:
            output_type: The Pydantic model type for structured output

        Returns:
            Outlines Generator instance

        Raises:
            ImportError: If outlines is not installed
            ValueError: If provider is not supported
        """
        try:
            from outlines import Generator

            if self._outlines_model is None:
                self._outlines_model = self._create_outlines_model()

            return Generator(self._outlines_model, output_type)

        except ImportError as e:
            self.logger.error("Failed to import outlines", error=str(e))
            raise ImportError(
                "outlines is required for structured generation. "
                "Install it with: pip install outlines"
            ) from e

    def _create_outlines_model(self):
        """Create the underlying Outlines model based on provider."""
        try:
            if self.provider == "gemini":
                return self._create_gemini_outlines_model()
            elif self.provider == "openai":
                return self._create_openai_outlines_model()
            else:
                raise ValueError(f"Unsupported provider for Outlines: {self.provider}")

        except Exception as e:
            self.logger.error("Failed to create Outlines model", error=str(e), provider=self.provider)
            raise

    def _create_gemini_outlines_model(self):
        """Create Outlines model for Gemini using OpenAI-compatible endpoint."""
        try:
            import openai
            from outlines import from_openai

            # Use OpenAI-compatible endpoint for Gemini
            client = openai.OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=self.api_key
            )

            # Map Gemini model names to OpenAI-compatible format
            model_name = self._map_gemini_model_name(self.model)

            self.logger.info("Creating Gemini Outlines model", model=model_name)
            return from_openai(client, model_name)

        except ImportError as e:
            self.logger.error("Failed to import required packages for Gemini Outlines", error=str(e))
            raise ImportError(
                "openai package is required for Gemini Outlines integration. "
                "Install it with: pip install openai"
            ) from e

    def _create_openai_outlines_model(self):
        """Create Outlines model for OpenAI."""
        try:
            import openai
            from outlines import from_openai

            client = openai.OpenAI(api_key=self.api_key)

            self.logger.info("Creating OpenAI Outlines model", model=self.model)
            return from_openai(client, self.model)

        except ImportError as e:
            self.logger.error("Failed to import openai for Outlines", error=str(e))
            raise ImportError(
                "openai package is required for OpenAI Outlines integration. "
                "Install it with: pip install openai"
            ) from e

    def _map_gemini_model_name(self, model: str) -> str:
        """Map Gemini model names to OpenAI-compatible format.

        Args:
            model: The Gemini model identifier

        Returns:
            The mapped model name for OpenAI-compatible endpoint
        """
        # Handle different model name formats
        if model.startswith("gemini"):
            return model
        else:
            # Default to Gemini 1.5 Flash if not specified
            return "gemini-1.5-flash"

    def is_available(self) -> bool:
        """Check if the Outlines provider is available.

        Returns:
            True if the provider is available, False otherwise
        """
        try:
            import outlines
            return self.api_key is not None
        except ImportError:
            return False

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider.

        Returns:
            Dictionary with provider information
        """
        return {
            "name": "outlines",
            "underlying_provider": self.provider,
            "model": self.model,
            "available": self.is_available(),
            "api_key_configured": self.api_key is not None,
            "config": self.config.model_dump(exclude={"api_key"})  # Exclude sensitive data
        }


class ProviderFactory:
    """Factory for creating AI providers."""

    _providers = {
        "gemini": GeminiProvider,
        "outlines": OutlinesProvider,
    }

    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        config: Optional[ProviderConfig] = None,
        **kwargs
    ) -> Union[GeminiProvider, OutlinesProvider]:
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

        provider_class = cls._providers[provider_name]
        if provider_name == "outlines":
            # OutlinesProvider needs additional parameters
            return provider_class(config, **kwargs)
        else:
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
