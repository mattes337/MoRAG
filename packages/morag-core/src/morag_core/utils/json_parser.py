"""Enhanced JSON parsing utilities with error recovery for LLM responses."""

import json
import re
import structlog
from typing import Any, Dict, List, Optional, Union

logger = structlog.get_logger(__name__)


class JSONParsingError(Exception):
    """Custom exception for JSON parsing errors."""
    pass


class EnhancedJSONParser:
    """Enhanced JSON parser with robust error recovery for malformed LLM responses."""
    
    def __init__(self):
        self.logger = logger.bind(component="json_parser")
    
    def parse_json_response(self, response: str, fallback_value: Any = None) -> Any:
        """Parse JSON response with multi-level error handling.
        
        Args:
            response: Raw response string from LLM
            fallback_value: Value to return if all parsing attempts fail
            
        Returns:
            Parsed JSON data or fallback_value
            
        Raises:
            JSONParsingError: If parsing fails and no fallback provided
        """
        if not response or not response.strip():
            if fallback_value is not None:
                return fallback_value
            raise JSONParsingError("Empty response")
        
        # Log the raw response for debugging
        self.logger.debug("Parsing JSON response", response_length=len(response))
        
        # Step 1: Try direct JSON parsing
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            self.logger.debug("Direct JSON parsing failed", error=str(e))
        
        # Step 2: Extract JSON from markdown or wrapped text
        try:
            cleaned_response = self._extract_json_from_text(response)
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            self.logger.debug("Cleaned JSON parsing failed", error=str(e))
        
        # Step 3: Fix common JSON issues
        try:
            fixed_response = self._fix_common_json_issues(response)
            return json.loads(fixed_response)
        except json.JSONDecodeError as e:
            self.logger.debug("Fixed JSON parsing failed", error=str(e))
        
        # Step 4: Extract partial JSON
        try:
            partial_json = self._extract_partial_json(response)
            if partial_json is not None:
                return partial_json
        except Exception as e:
            self.logger.debug("Partial JSON extraction failed", error=str(e))
        
        # Step 5: Try to extract array elements
        try:
            array_elements = self._extract_array_elements(response)
            if array_elements:
                return array_elements
        except Exception as e:
            self.logger.debug("Array element extraction failed", error=str(e))
        
        # All parsing attempts failed
        self.logger.error("All JSON parsing attempts failed", response=response[:500])
        
        if fallback_value is not None:
            return fallback_value
        
        raise JSONParsingError(f"Failed to parse JSON response: {response[:200]}...")
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from text that may be wrapped in markdown or other formatting."""
        text = text.strip()
        
        # Look for JSON block in markdown
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        
        # Try to find JSON object boundaries
        if not text.startswith(("{", "[")):
            # Look for first { or [
            start_brace = text.find("{")
            start_bracket = text.find("[")
            
            if start_brace == -1 and start_bracket == -1:
                return text
            
            if start_brace == -1:
                start = start_bracket
            elif start_bracket == -1:
                start = start_brace
            else:
                start = min(start_brace, start_bracket)
            
            text = text[start:]
        
        if not text.endswith(("}", "]")):
            # Look for last } or ]
            end_brace = text.rfind("}")
            end_bracket = text.rfind("]")
            
            if end_brace == -1 and end_bracket == -1:
                return text
            
            if end_brace == -1:
                end = end_bracket + 1
            elif end_bracket == -1:
                end = end_brace + 1
            else:
                end = max(end_brace, end_bracket) + 1
            
            text = text[:end]
        
        return text
    
    def _fix_common_json_issues(self, text: str) -> str:
        """Fix common JSON formatting issues."""
        text = self._extract_json_from_text(text)
        
        # Fix trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix missing quotes around keys
        text = re.sub(r'(\w+)(\s*:\s*)', r'"\1"\2', text)
        
        # Fix single quotes to double quotes
        text = re.sub(r"'([^']*)'", r'"\1"', text)
        
        # Fix unterminated strings - add closing quote before next key or end
        text = self._fix_unterminated_strings(text)
        
        # Fix missing commas between objects/arrays
        text = re.sub(r'}\s*{', r'},{', text)
        text = re.sub(r']\s*\[', r'],[', text)
        
        return text
    
    def _fix_unterminated_strings(self, text: str) -> str:
        """Fix unterminated strings in JSON."""
        # This is a complex operation, so we'll use a simple heuristic
        # Look for patterns like: "key": "value without closing quote
        
        # Pattern: "key": "value followed by newline or next key
        pattern = r'"([^"]*)":\s*"([^"]*?)(?=\s*[,}\]\n]|$|"[^"]*":)'
        
        def fix_match(match):
            key = match.group(1)
            value = match.group(2)
            # If value doesn't end with quote, add it
            if not value.endswith('"'):
                return f'"{key}": "{value}"'
            return match.group(0)
        
        return re.sub(pattern, fix_match, text)
    
    def _extract_partial_json(self, text: str) -> Optional[Any]:
        """Extract valid JSON objects from partially malformed text."""
        text = self._extract_json_from_text(text)
        
        # Try to find complete JSON objects within the text
        if text.startswith("{"):
            return self._extract_partial_object(text)
        elif text.startswith("["):
            return self._extract_partial_array(text)
        
        return None
    
    def _extract_partial_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a partial JSON object."""
        result = {}
        brace_count = 0
        in_string = False
        escape_next = False
        current_key = None
        current_value = ""
        in_key = False
        in_value = False
        
        i = 0
        while i < len(text):
            char = text[i]
            
            if escape_next:
                escape_next = False
                current_value += char
                i += 1
                continue
            
            if char == '\\':
                escape_next = True
                current_value += char
                i += 1
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                if not in_string and in_key:
                    current_key = current_value.strip('"')
                    current_value = ""
                    in_key = False
                elif not in_string and in_value:
                    try:
                        # Try to parse the value
                        if current_value.startswith('"'):
                            result[current_key] = current_value.strip('"')
                        else:
                            result[current_key] = json.loads(current_value)
                    except:
                        result[current_key] = current_value.strip('"')
                    current_value = ""
                    in_value = False
                    current_key = None
                elif in_string:
                    current_value += char
                else:
                    if not in_key and not in_value:
                        in_key = True
                    current_value += char
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        break
                elif char == ':' and in_key:
                    in_key = False
                    in_value = True
                    current_key = current_value.strip().strip('"')
                    current_value = ""
                elif char == ',' and in_value:
                    try:
                        if current_value.strip():
                            if current_value.strip().startswith('"'):
                                result[current_key] = current_value.strip().strip('"')
                            else:
                                result[current_key] = json.loads(current_value.strip())
                    except:
                        result[current_key] = current_value.strip().strip('"')
                    current_value = ""
                    in_value = False
                    current_key = None
                elif in_value:
                    current_value += char
                elif in_key:
                    current_value += char
            else:
                current_value += char
            
            i += 1
        
        # Handle any remaining value
        if current_key and current_value.strip():
            try:
                if current_value.strip().startswith('"'):
                    result[current_key] = current_value.strip().strip('"')
                else:
                    result[current_key] = json.loads(current_value.strip())
            except:
                result[current_key] = current_value.strip().strip('"')
        
        return result if result else None
    
    def _extract_partial_array(self, text: str) -> Optional[List[Any]]:
        """Extract a partial JSON array."""
        result = []
        bracket_count = 0
        in_string = False
        escape_next = False
        current_element = ""
        
        i = 0
        while i < len(text):
            char = text[i]
            
            if escape_next:
                escape_next = False
                current_element += char
                i += 1
                continue
            
            if char == '\\':
                escape_next = True
                current_element += char
                i += 1
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                current_element += char
            elif not in_string:
                if char == '[':
                    bracket_count += 1
                    if bracket_count > 1:
                        current_element += char
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        # End of array
                        if current_element.strip():
                            try:
                                result.append(json.loads(current_element.strip()))
                            except:
                                result.append(current_element.strip().strip('"'))
                        break
                    else:
                        current_element += char
                elif char == ',' and bracket_count == 1:
                    # Element separator
                    if current_element.strip():
                        try:
                            result.append(json.loads(current_element.strip()))
                        except:
                            result.append(current_element.strip().strip('"'))
                    current_element = ""
                else:
                    current_element += char
            else:
                current_element += char
            
            i += 1
        
        return result if result else None
    
    def _extract_array_elements(self, text: str) -> Optional[List[Any]]:
        """Extract individual array elements even if the array structure is broken."""
        # Look for patterns that might be array elements
        elements = []
        
        # Try to find JSON objects within the text
        object_pattern = r'\{[^{}]*\}'
        matches = re.findall(object_pattern, text)
        
        for match in matches:
            try:
                obj = json.loads(match)
                elements.append(obj)
            except:
                continue
        
        return elements if elements else None


# Global parser instance
_parser = EnhancedJSONParser()

def parse_json_response(response: str, fallback_value: Any = None) -> Any:
    """Parse JSON response with enhanced error recovery.
    
    Args:
        response: Raw response string from LLM
        fallback_value: Value to return if parsing fails
        
    Returns:
        Parsed JSON data or fallback_value
    """
    return _parser.parse_json_response(response, fallback_value)
