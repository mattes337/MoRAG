"""LLM response parsing utilities for MoRAG agents.

This module provides utilities for parsing and cleaning LLM responses,
particularly for extracting JSON from various response formats.
"""

import json
import re
from typing import Any, Dict, Optional


class LLMResponseParseError(Exception):
    """Exception raised when LLM response parsing fails."""

    pass


class LLMResponseParser:
    """Utility class for parsing LLM responses."""

    @staticmethod
    def clean_response(response: str) -> str:
        """Clean LLM response by removing markdown code blocks and extra whitespace.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response string
        """
        if not response:
            return ""

        # Remove markdown code blocks
        response = re.sub(r"```(?:json)?\s*", "", response)
        response = re.sub(r"```\s*$", "", response)

        # Remove extra whitespace
        response = response.strip()

        return response

    @staticmethod
    def extract_json_from_text(text: str) -> Optional[str]:
        """Extract JSON from text that may contain other content.

        Args:
            text: Text that may contain JSON

        Returns:
            Extracted JSON string or None if not found
        """
        if not text:
            return None

        # Try to find JSON in markdown code blocks
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL
        )
        if json_match:
            return json_match.group(1)

        # Try to find JSON object or array
        json_match = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        return None

    @staticmethod
    def parse_json_response(
        response: str,
        fallback_value: Optional[Any] = None,
        context: Optional[str] = None,
    ) -> Any:
        """Parse JSON from LLM response with fallback handling.

        Args:
            response: Raw LLM response
            fallback_value: Value to return if parsing fails
            context: Context string for error messages

        Returns:
            Parsed JSON data or fallback value

        Raises:
            LLMResponseParseError: If parsing fails and no fallback provided
        """
        if not response or not response.strip():
            if fallback_value is not None:
                return fallback_value
            raise LLMResponseParseError(
                f"Empty response{f' in {context}' if context else ''}"
            )

        # Clean the response
        cleaned = LLMResponseParser.clean_response(response)

        # Try direct JSON parsing
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from text
        json_str = LLMResponseParser.extract_json_from_text(cleaned)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # If we have a fallback, use it
        if fallback_value is not None:
            return fallback_value

        # Otherwise raise an error
        error_msg = (
            f"Failed to parse JSON from response{f' in {context}' if context else ''}"
        )
        raise LLMResponseParseError(error_msg)


# Convenience functions for backward compatibility
def parse_json_response(
    response: str, fallback_value: Optional[Any] = None, context: Optional[str] = None
) -> Any:
    """Parse JSON from LLM response."""
    return LLMResponseParser.parse_json_response(response, fallback_value, context)


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON from text."""
    return LLMResponseParser.extract_json_from_text(text)


def clean_json_response(response: str) -> str:
    """Clean LLM response."""
    return LLMResponseParser.clean_response(response)
