"""Utility functions for parsing LLM responses with robust error handling."""

import json
import re
import structlog
from typing import Any, Dict, List, Optional, Union

logger = structlog.get_logger(__name__)


class LLMResponseParseError(Exception):
    """Exception raised when LLM response cannot be parsed."""
    pass


def parse_json_response(
    response_text: str,
    fallback_value: Any = None,
    context: str = "unknown"
) -> Any:
    """
    Parse JSON from LLM response with robust error handling.
    
    Args:
        response_text: Raw response text from LLM
        fallback_value: Value to return if parsing fails
        context: Context description for logging
        
    Returns:
        Parsed JSON object or fallback_value
        
    Raises:
        LLMResponseParseError: If parsing fails and no fallback provided
    """
    if not response_text:
        if fallback_value is not None:
            logger.warning(
                "Empty LLM response, using fallback",
                context=context,
                fallback_type=type(fallback_value).__name__
            )
            return fallback_value
        raise LLMResponseParseError("Empty response from LLM")
    
    # Clean the response text
    cleaned_text = _clean_response_text(response_text)
    
    if not cleaned_text:
        if fallback_value is not None:
            logger.warning(
                "No content after cleaning LLM response, using fallback",
                context=context,
                original_length=len(response_text),
                fallback_type=type(fallback_value).__name__
            )
            return fallback_value
        raise LLMResponseParseError("No content after cleaning response")
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        if fallback_value is not None:
            logger.warning(
                "Failed to parse JSON from LLM response, using fallback",
                context=context,
                error=str(e),
                response_preview=cleaned_text[:100],
                response_length=len(cleaned_text),
                fallback_type=type(fallback_value).__name__
            )
            return fallback_value
        
        logger.error(
            "Failed to parse JSON from LLM response",
            context=context,
            error=str(e),
            response_preview=cleaned_text[:200],
            response_length=len(cleaned_text)
        )
        raise LLMResponseParseError(f"JSON parsing failed: {e}")


def _clean_response_text(response_text: str) -> str:
    """Clean response text by removing markdown and extracting JSON."""
    text = response_text.strip()
    
    # Remove markdown code blocks
    if text.startswith('```json'):
        text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
    elif text.startswith('```'):
        text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
    
    text = text.strip()
    
    # If text doesn't start with JSON, try to extract it
    if text and not text.startswith(('{', '[')):
        # Look for JSON object or array
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Try to find JSON boundaries more aggressively
            start_idx = max(text.find('{'), text.find('['))
            if start_idx != -1:
                # Find matching closing bracket
                if text[start_idx] == '{':
                    end_idx = text.rfind('}') + 1
                else:
                    end_idx = text.rfind(']') + 1
                
                if end_idx > start_idx:
                    text = text[start_idx:end_idx]
    
    return text.strip()


def extract_json_objects(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract all JSON objects from response text.
    
    Args:
        response_text: Raw response text that may contain multiple JSON objects
        
    Returns:
        List of parsed JSON objects
    """
    objects = []
    cleaned_text = _clean_response_text(response_text)
    
    if not cleaned_text:
        return objects
    
    # Try to parse as single object first
    try:
        obj = json.loads(cleaned_text)
        if isinstance(obj, list):
            objects.extend(obj)
        else:
            objects.append(obj)
        return objects
    except json.JSONDecodeError:
        pass
    
    # Try to find multiple JSON objects
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, cleaned_text, re.DOTALL)
    
    for match in matches:
        try:
            obj = json.loads(match)
            objects.append(obj)
        except json.JSONDecodeError:
            continue
    
    return objects


def validate_json_structure(
    data: Any,
    required_fields: List[str],
    context: str = "unknown"
) -> bool:
    """
    Validate that parsed JSON has required structure.
    
    Args:
        data: Parsed JSON data
        required_fields: List of required field names
        context: Context description for logging
        
    Returns:
        True if structure is valid, False otherwise
    """
    if not isinstance(data, dict):
        logger.warning(
            "JSON data is not a dictionary",
            context=context,
            data_type=type(data).__name__
        )
        return False
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        logger.warning(
            "JSON data missing required fields",
            context=context,
            missing_fields=missing_fields,
            available_fields=list(data.keys())
        )
        return False
    
    return True
