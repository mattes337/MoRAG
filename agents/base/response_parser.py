"""Robust LLM response parsing utilities for agents."""

import json
import re
import structlog
from typing import Any, Dict, Optional, Union, Type, TypeVar
from .exceptions import ValidationError

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class LLMResponseParseError(Exception):
    """Exception raised when LLM response parsing fails."""
    pass


class LLMResponseParser:
    """Robust parser for LLM responses with fallback strategies."""
    
    @staticmethod
    def parse_json_response(
        response: Union[str, dict],
        fallback_value: Optional[Any] = None,
        context: str = "unknown"
    ) -> Any:
        """Parse JSON from LLM response with robust error handling.

        Args:
            response: Raw LLM response text or dict (for testing)
            fallback_value: Value to return if parsing fails
            context: Context for logging purposes

        Returns:
            Parsed JSON data or fallback value

        Raises:
            LLMResponseParseError: If parsing fails and no fallback provided
        """
        # Handle dict input (for testing)
        if isinstance(response, dict):
            return response

        if not response or not response.strip():
            if fallback_value is not None:
                return fallback_value
            raise LLMResponseParseError("Empty response")
        
        # Clean the response
        cleaned_response = response.strip()
        
        # Try direct JSON parsing first
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', cleaned_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in response
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON array in response
        json_match = re.search(r'\[.*\]', cleaned_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If all parsing attempts fail
        logger.warning(
            "Failed to parse JSON from LLM response",
            context=context,
            response_preview=response[:200],
            response_length=len(response)
        )
        
        if fallback_value is not None:
            return fallback_value
        
        raise LLMResponseParseError(f"Could not parse JSON from response in context: {context}")

    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """Extract the first valid JSON object from text.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            First valid JSON object found, or None
        """
        try:
            return LLMResponseParser.parse_json_response(text)
        except LLMResponseParseError:
            return None

    @staticmethod
    def clean_response(response: str) -> str:
        """Clean LLM response for better parsing.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned response text
        """
        # Remove common prefixes/suffixes
        cleaned = response.strip()
        
        # Remove markdown code block markers
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        return cleaned.strip()

    @staticmethod
    def parse_and_validate(
        response: str,
        result_type: Type[T],
        context: str = "unknown",
        strict_json: bool = True
    ) -> T:
        """Parse response and validate against result type.
        
        Args:
            response: Raw LLM response
            result_type: Expected result type (Pydantic model)
            context: Context for error reporting
            strict_json: Whether to require strict JSON parsing
            
        Returns:
            Validated result instance
            
        Raises:
            ValidationError: If parsing or validation fails
        """
        try:
            if strict_json:
                # Use robust JSON parsing
                parsed_data = LLMResponseParser.parse_json_response(
                    response, 
                    context=context
                )
                
                # Validate with Pydantic model
                if hasattr(result_type, 'model_validate'):
                    # Pydantic v2
                    return result_type.model_validate(parsed_data)
                elif hasattr(result_type, 'parse_obj'):
                    # Pydantic v1
                    return result_type.parse_obj(parsed_data)
                else:
                    # Regular class instantiation
                    return result_type(**parsed_data)
            else:
                # For non-JSON outputs, create a simple wrapper
                if hasattr(result_type, 'model_validate'):
                    return result_type.model_validate({"content": response})
                else:
                    return result_type(content=response)
                    
        except Exception as e:
            # Handle response preview for both string and dict
            if isinstance(response, str):
                response_preview = response[:200]
            else:
                response_preview = str(response)[:200]

            logger.error(
                "Response parsing and validation failed",
                context=context,
                error=str(e),
                response_preview=response_preview
            )
            raise ValidationError(f"Result validation failed in {context}: {e}") from e
