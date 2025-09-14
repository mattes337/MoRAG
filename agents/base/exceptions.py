"""Exception classes for MoRAG agents."""


class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class ConfigurationError(AgentError):
    """Raised when agent configuration is invalid."""
    pass


class PromptGenerationError(AgentError):
    """Raised when prompt generation fails."""
    pass


class ValidationError(AgentError):
    """Raised when input or output validation fails."""
    pass


class RetryExhaustedError(AgentError):
    """Raised when all retry attempts are exhausted."""
    pass


class ModelError(AgentError):
    """Raised when the underlying model fails."""
    pass


class TimeoutError(AgentError):
    """Raised when agent execution times out."""
    pass


class RateLimitError(AgentError):
    """Raised when rate limits are exceeded."""
    pass
