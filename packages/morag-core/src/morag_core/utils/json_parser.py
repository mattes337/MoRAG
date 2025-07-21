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
            self.logger.debug("Direct JSON parsing failed", error=str(e), error_position=getattr(e, 'pos', None))

        # Step 2: Extract JSON from markdown or wrapped text
        try:
            cleaned_response = self._extract_json_from_text(response)
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            self.logger.debug("Cleaned JSON parsing failed", error=str(e), error_position=getattr(e, 'pos', None))

        # Step 3: Fix common JSON issues
        try:
            fixed_response = self._fix_common_json_issues(response)
            return json.loads(fixed_response)
        except json.JSONDecodeError as e:
            self.logger.debug("Fixed JSON parsing failed", error=str(e), error_position=getattr(e, 'pos', None))

        # Step 4: Try aggressive string fixing for unterminated strings
        try:
            aggressively_fixed = self._aggressive_string_fix(response)
            return json.loads(aggressively_fixed)
        except json.JSONDecodeError as e:
            self.logger.debug("Aggressive string fix failed", error=str(e), error_position=getattr(e, 'pos', None))

        # Step 5: Extract partial JSON
        try:
            partial_json = self._extract_partial_json(response)
            if partial_json is not None:
                return partial_json
        except Exception as e:
            self.logger.debug("Partial JSON extraction failed", error=str(e))

        # Step 6: Try to extract array elements
        try:
            array_elements = self._extract_array_elements(response)
            if array_elements:
                return array_elements
        except Exception as e:
            self.logger.debug("Array element extraction failed", error=str(e))

        # Step 7: Last resort - try to extract any valid JSON fragments
        try:
            fragments = self._extract_json_fragments(response)
            if fragments:
                return fragments
        except Exception as e:
            self.logger.debug("JSON fragment extraction failed", error=str(e))

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
        """Fix unterminated strings in JSON with enhanced detection."""
        # Enhanced approach to handle various unterminated string patterns

        # First, handle simple cases where quotes are missing at the end
        # Pattern: "key": "value without closing quote followed by comma, brace, bracket, or end
        pattern1 = r'"([^"]*)":\s*"([^"]*?)(?=\s*[,}\]\n]|$|"[^"]*":)'

        def fix_simple_match(match):
            key = match.group(1)
            value = match.group(2)
            # If value doesn't end with quote, add it
            if not value.endswith('"'):
                return f'"{key}": "{value}"'
            return match.group(0)

        text = re.sub(pattern1, fix_simple_match, text)

        # Handle more complex cases with escaped characters
        # Look for strings that start with quote but don't have proper closing
        lines = text.split('\n')
        fixed_lines = []

        for line in lines:
            # Check if line has an unterminated string
            if self._has_unterminated_string(line):
                line = self._fix_line_unterminated_string(line)
            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _has_unterminated_string(self, line: str) -> bool:
        """Check if a line has an unterminated string."""
        # Count quotes, accounting for escaped quotes
        quote_count = 0
        i = 0
        while i < len(line):
            if line[i] == '"' and (i == 0 or line[i-1] != '\\'):
                quote_count += 1
            elif line[i] == '\\' and i + 1 < len(line):
                i += 1  # Skip escaped character
            i += 1

        # If odd number of quotes, we likely have an unterminated string
        return quote_count % 2 == 1

    def _fix_line_unterminated_string(self, line: str) -> str:
        """Fix unterminated string in a single line."""
        # Find the last unmatched quote and add closing quote before special characters
        quote_positions = []
        i = 0
        while i < len(line):
            if line[i] == '"' and (i == 0 or line[i-1] != '\\'):
                quote_positions.append(i)
            elif line[i] == '\\' and i + 1 < len(line):
                i += 1  # Skip escaped character
            i += 1

        if len(quote_positions) % 2 == 1:
            # Find where to insert the closing quote
            last_quote_pos = quote_positions[-1]
            # Look for next comma, brace, bracket, or end of line
            insert_pos = len(line)
            for pos in range(last_quote_pos + 1, len(line)):
                if line[pos] in ',}]\n':
                    insert_pos = pos
                    break

            # Insert closing quote
            line = line[:insert_pos] + '"' + line[insert_pos:]

        return line

    def _aggressive_string_fix(self, text: str) -> str:
        """Aggressively fix JSON strings by handling various edge cases."""
        text = self._extract_json_from_text(text)

        # First apply standard fixes
        text = self._fix_common_json_issues(text)

        # Handle the specific error case: "Unterminated string starting at: line 1 column 13"
        # This often happens when LLM generates malformed JSON with unclosed quotes
        try:
            # Count quotes to detect unterminated strings
            quote_count = text.count('"')
            if quote_count % 2 != 0:
                # Odd number of quotes - likely unterminated string
                # Find the position around character 12-13 where the error occurs
                if len(text) > 12:
                    # Look for unterminated strings around the error position
                    start_pos = max(0, 10)  # Start a bit before position 12
                    end_pos = min(len(text), 50)  # Look ahead to find a good closing point

                    # Find the last quote before position 12
                    last_quote_before = text.rfind('"', 0, 13)
                    if last_quote_before != -1:
                        # Find the next quote after the error position
                        next_quote_after = text.find('"', 13)
                        if next_quote_after == -1:
                            # No closing quote found, try to add one at a reasonable position
                            # Look for natural break points like comma, brace, or bracket
                            break_chars = [',', '}', ']', '\n', ':', ' ']
                            break_pos = -1
                            search_start = 13

                            for char in break_chars:
                                pos = text.find(char, search_start)
                                if pos != -1 and (break_pos == -1 or pos < break_pos):
                                    break_pos = pos

                            if break_pos != -1:
                                # Insert closing quote before the break character
                                text = text[:break_pos] + '"' + text[break_pos:]
                            else:
                                # No break found, add quote at the end
                                text = text + '"'
        except Exception as e:
            self.logger.debug("Quote fixing failed", error=str(e))

        # Handle specific unterminated string patterns
        # Pattern: "key": "value that ends abruptly at line end
        text = re.sub(r'"([^"]*)":\s*"([^"]*?)$', r'"\1": "\2"', text, flags=re.MULTILINE)

        # Pattern: "key": "value that ends abruptly before next key
        text = re.sub(r'"([^"]*)":\s*"([^"]*?)\s*"([^"]*)":', r'"\1": "\2", "\3":', text)

        # Handle cases where quotes are missing entirely
        # Pattern: key: value (no quotes)
        text = re.sub(r'(\w+):\s*([^",}\]\n]+)', r'"\1": "\2"', text)

        # Fix newlines within strings
        text = re.sub(r'"([^"]*)\n([^"]*)"', r'"\1 \2"', text)

        # Ensure proper comma placement
        text = re.sub(r'"\s*\n\s*"', r'",\n"', text)

        return text

    def _extract_json_fragments(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract any valid JSON fragments from malformed text."""
        result = {}

        # Try to extract key-value pairs using regex
        # Pattern: "key": "value" or "key": value
        kv_pattern = r'"([^"]+)":\s*(?:"([^"]*)"|([^,}\]\n]+))'
        matches = re.findall(kv_pattern, text)

        for match in matches:
            key = match[0]
            value = match[1] if match[1] else match[2]

            # Try to parse value as JSON if possible
            try:
                if value.strip().startswith(('{', '[')):
                    result[key] = json.loads(value.strip())
                elif value.strip().lower() in ('true', 'false'):
                    result[key] = value.strip().lower() == 'true'
                elif value.strip().isdigit():
                    result[key] = int(value.strip())
                elif self._is_float(value.strip()):
                    result[key] = float(value.strip())
                else:
                    result[key] = value.strip()
            except:
                result[key] = value.strip()

        return result if result else None

    def _is_float(self, value: str) -> bool:
        """Check if a string represents a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
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


def parse_llm_response_with_retry(
    response: str,
    fallback_value: Any = None,
    max_attempts: int = 3,
    context: str = "LLM response"
) -> Any:
    """Parse LLM response with multiple attempts and enhanced error recovery.

    This function provides additional retry logic specifically for LLM responses
    that may have transient formatting issues.

    Args:
        response: Raw response string from LLM
        fallback_value: Value to return if all parsing attempts fail
        max_attempts: Maximum number of parsing attempts
        context: Context description for logging

    Returns:
        Parsed JSON data or fallback_value

    Raises:
        JSONParsingError: If parsing fails and no fallback provided
    """
    last_error = None

    for attempt in range(max_attempts):
        try:
            return _parser.parse_json_response(response, fallback_value)
        except JSONParsingError as e:
            last_error = e
            logger.debug(
                f"JSON parsing attempt {attempt + 1} failed",
                context=context,
                error=str(e),
                attempt=attempt + 1,
                max_attempts=max_attempts
            )

            if attempt < max_attempts - 1:
                # Try preprocessing the response differently for next attempt
                response = _preprocess_for_retry(response, attempt)

    # All attempts failed
    logger.error(
        f"All {max_attempts} JSON parsing attempts failed",
        context=context,
        error=str(last_error),
        response_preview=response[:200] if response else "None"
    )

    if fallback_value is not None:
        return fallback_value

    raise last_error


def _preprocess_for_retry(response: str, attempt: int) -> str:
    """Preprocess response differently for each retry attempt."""
    if attempt == 0:
        # First retry: try removing extra whitespace and normalizing quotes
        response = re.sub(r'\s+', ' ', response.strip())
        response = response.replace('"', '"').replace('"', '"')  # Normalize smart quotes
        response = response.replace(''', "'").replace(''', "'")  # Normalize smart apostrophes

        # Handle the specific "Unterminated string starting at: line 1 column 13" error
        # Try to fix unterminated strings by adding missing quotes
        if len(response) > 12:
            # Check if there's an unterminated string around position 12
            quote_count = response[:13].count('"')
            if quote_count % 2 != 0:  # Odd number of quotes means unterminated string
                # Find a good place to close the string
                remaining = response[13:]
                # Look for natural break points
                break_chars = [',', '}', ']', '\n', ':', ' ']
                break_pos = -1
                for char in break_chars:
                    pos = remaining.find(char)
                    if pos != -1 and (break_pos == -1 or pos < break_pos):
                        break_pos = pos

                if break_pos != -1:
                    # Insert closing quote before the break character
                    response = response[:13 + break_pos] + '"' + response[13 + break_pos:]
                else:
                    # No break found, add quote at the end
                    response = response + '"'

    elif attempt == 1:
        # Second retry: try more aggressive cleaning
        response = re.sub(r'[^\x20-\x7E\n\r\t]', '', response)  # Remove non-printable chars
        response = re.sub(r'\n+', '\n', response)  # Normalize newlines

        # Try to balance quotes
        quote_count = response.count('"')
        if quote_count % 2 != 0:
            response = response + '"'  # Add missing closing quote

    return response
