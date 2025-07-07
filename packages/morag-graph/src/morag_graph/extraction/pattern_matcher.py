"""Pattern-based entity extraction for enhanced accuracy."""

import re
import structlog
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..models import Entity

logger = structlog.get_logger(__name__)


class PatternType(str, Enum):
    """Types of patterns for entity matching."""
    REGEX = "regex"
    EXACT = "exact"
    FUZZY = "fuzzy"
    CONTEXTUAL = "contextual"


@dataclass
class EntityPattern:
    """Represents a pattern for entity matching."""
    pattern: str
    entity_type: str
    pattern_type: PatternType
    confidence: float
    context_keywords: Optional[List[str]] = None
    case_sensitive: bool = False
    description: str = ""


class EntityPatternMatcher:
    """Pattern-based entity matcher with curated knowledge bases."""
    
    def __init__(self):
        """Initialize the pattern matcher with curated patterns."""
        self.patterns: List[EntityPattern] = []
        self.logger = logger.bind(component="pattern_matcher")
        self._load_default_patterns()
    
    def _load_default_patterns(self):
        """Load default entity patterns.

        Note: This method loads minimal generic patterns. For domain-specific
        patterns, use add_pattern() or load_patterns_from_file() methods.
        """
        # Only include very generic, domain-agnostic patterns
        generic_patterns = [
            # Generic organization suffixes
            EntityPattern(
                pattern=r"\b[A-Z][a-z]+ (?:Inc|Corp|Corporation|LLC|Ltd|Limited|Company|Co)\b",
                entity_type="ORGANIZATION",
                pattern_type=PatternType.REGEX,
                confidence=0.7,
                description="Company suffixes"
            ),
        ]

        # Date patterns (generic, domain-agnostic)
        date_patterns = [
            EntityPattern(
                pattern=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
                entity_type="DATE",
                pattern_type=PatternType.REGEX,
                confidence=0.95,
                description="Full date format"
            ),
            EntityPattern(
                pattern=r"\b\d{1,2}/\d{1,2}/\d{4}\b",
                entity_type="DATE",
                pattern_type=PatternType.REGEX,
                confidence=0.9,
                description="MM/DD/YYYY format"
            ),
            EntityPattern(
                pattern=r"\b\d{4}-\d{2}-\d{2}\b",
                entity_type="DATE",
                pattern_type=PatternType.REGEX,
                confidence=0.95,
                description="ISO date format"
            ),
        ]

        # Combine all patterns
        self.patterns.extend(generic_patterns)
        self.patterns.extend(date_patterns)
        
        self.logger.info(f"Loaded {len(self.patterns)} default patterns")
    
    def add_pattern(self, pattern: EntityPattern):
        """Add a custom pattern."""
        self.patterns.append(pattern)
        self.logger.debug(f"Added pattern: {pattern.description}")
    
    def extract_entities(self, text: str, min_confidence: float = 0.6) -> List[Entity]:
        """Extract entities using pattern matching.
        
        Args:
            text: Text to extract entities from
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of Entity objects
        """
        entities = []
        
        for pattern in self.patterns:
            if pattern.confidence < min_confidence:
                continue
                
            matches = self._find_pattern_matches(text, pattern)
            for match in matches:
                entity = Entity(
                    name=match["text"],
                    type=pattern.entity_type,
                    confidence=pattern.confidence,
                    attributes={
                        "extraction_method": "pattern_matching",
                        "pattern_type": pattern.pattern_type.value,
                        "pattern_description": pattern.description,
                        "start_pos": match["start"],
                        "end_pos": match["end"],
                        "source_text": match["text"]
                    }
                )
                entities.append(entity)
        
        # Deduplicate entities
        entities = self._deduplicate_entities(entities)
        
        self.logger.info(f"Pattern matching found {len(entities)} entities")
        return entities
    
    def _find_pattern_matches(self, text: str, pattern: EntityPattern) -> List[Dict[str, Any]]:
        """Find matches for a specific pattern."""
        matches = []
        
        if pattern.pattern_type == PatternType.REGEX:
            flags = 0 if pattern.case_sensitive else re.IGNORECASE
            for match in re.finditer(pattern.pattern, text, flags):
                matches.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        elif pattern.pattern_type == PatternType.EXACT:
            # Simple exact string matching
            search_text = text if pattern.case_sensitive else text.lower()
            search_pattern = pattern.pattern if pattern.case_sensitive else pattern.pattern.lower()
            
            start = 0
            while True:
                pos = search_text.find(search_pattern, start)
                if pos == -1:
                    break
                matches.append({
                    "text": text[pos:pos + len(search_pattern)],
                    "start": pos,
                    "end": pos + len(search_pattern)
                })
                start = pos + 1
        
        return matches
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on text and position overlap."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda e: e.attributes.get("start_pos", 0))
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            start_pos = entity.attributes.get("start_pos", 0)
            end_pos = entity.attributes.get("end_pos", 0)
            
            for existing in deduplicated:
                existing_start = existing.attributes.get("start_pos", 0)
                existing_end = existing.attributes.get("end_pos", 0)
                
                # Check for position overlap
                if (start_pos < existing_end and end_pos > existing_start):
                    # Keep the entity with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated
