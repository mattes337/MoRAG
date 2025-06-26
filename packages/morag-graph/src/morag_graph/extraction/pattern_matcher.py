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
        """Load default entity patterns."""
        # Technology patterns
        tech_patterns = [
            EntityPattern(
                pattern=r"\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin)\b",
                entity_type="TECHNOLOGY",
                pattern_type=PatternType.REGEX,
                confidence=0.9,
                description="Programming languages"
            ),
            EntityPattern(
                pattern=r"\b(?:React|Vue|Angular|Django|Flask|Spring|Express|Laravel|Rails)\b",
                entity_type="TECHNOLOGY",
                pattern_type=PatternType.REGEX,
                confidence=0.85,
                description="Web frameworks"
            ),
            EntityPattern(
                pattern=r"\b(?:AWS|Azure|GCP|Google Cloud|Docker|Kubernetes|Jenkins|GitHub|GitLab)\b",
                entity_type="TECHNOLOGY",
                pattern_type=PatternType.REGEX,
                confidence=0.9,
                description="Cloud and DevOps tools"
            ),
            EntityPattern(
                pattern=r"\b(?:PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB)\b",
                entity_type="TECHNOLOGY",
                pattern_type=PatternType.REGEX,
                confidence=0.9,
                description="Databases"
            ),
        ]
        
        # Organization patterns
        org_patterns = [
            EntityPattern(
                pattern=r"\b(?:Microsoft|Google|Apple|Amazon|Meta|Facebook|Netflix|Tesla|OpenAI|Anthropic)\b",
                entity_type="ORGANIZATION",
                pattern_type=PatternType.REGEX,
                confidence=0.95,
                description="Major tech companies"
            ),
            EntityPattern(
                pattern=r"\b[A-Z][a-z]+ (?:Inc|Corp|Corporation|LLC|Ltd|Limited|Company|Co)\b",
                entity_type="ORGANIZATION",
                pattern_type=PatternType.REGEX,
                confidence=0.7,
                description="Company suffixes"
            ),
        ]
        
        # Location patterns
        location_patterns = [
            EntityPattern(
                pattern=r"\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|Fort Worth|Columbus|Charlotte|San Francisco|Indianapolis|Seattle|Denver|Washington|Boston|El Paso|Nashville|Detroit|Oklahoma City|Portland|Las Vegas|Memphis|Louisville|Baltimore|Milwaukee|Albuquerque|Tucson|Fresno|Sacramento|Kansas City|Long Beach|Mesa|Atlanta|Colorado Springs|Virginia Beach|Raleigh|Omaha|Miami|Oakland|Minneapolis|Tulsa|Wichita|New Orleans|Arlington|Cleveland|Tampa|Bakersfield|Aurora|Honolulu|Anaheim|Santa Ana|Corpus Christi|Riverside|Lexington|Stockton|Toledo|St. Paul|Newark|Greensboro|Plano|Henderson|Lincoln|Buffalo|Jersey City|Chula Vista|Fort Wayne|Orlando|St. Petersburg|Chandler|Laredo|Norfolk|Durham|Madison|Lubbock|Irvine|Winston-Salem|Glendale|Garland|Hialeah|Reno|Chesapeake|Gilbert|Baton Rouge|Irving|Scottsdale|North Las Vegas|Fremont|Boise|Richmond|San Bernardino|Birmingham|Spokane|Rochester|Des Moines|Modesto|Fayetteville|Tacoma|Oxnard|Fontana|Columbus|Montgomery|Moreno Valley|Shreveport|Aurora|Yonkers|Akron|Huntington Beach|Little Rock|Augusta|Amarillo|Glendale|Mobile|Grand Rapids|Salt Lake City|Tallahassee|Huntsville|Grand Prairie|Knoxville|Worcester|Newport News|Brownsville|Overland Park|Santa Clarita|Providence|Garden Grove|Chattanooga|Oceanside|Jackson|Fort Lauderdale|Santa Rosa|Rancho Cucamonga|Port St. Lucie|Tempe|Ontario|Vancouver|Cape Coral|Sioux Falls|Springfield|Peoria|Pembroke Pines|Elk Grove|Salem|Lancaster|Corona|Eugene|Palmdale|Salinas|Springfield|Pasadena|Fort Collins|Hayward|Pomona|Cary|Rockford|Alexandria|Escondido|McKinney|Kansas City|Joliet|Sunnyvale|Torrance|Bridgeport|Lakewood|Hollywood|Paterson|Naperville|Syracuse|Mesquite|Dayton|Savannah|Clarksville|Orange|Pasadena|Fullerton|Killeen|Frisco|Hampton|McAllen|Warren|Bellevue|West Valley City|Columbia|Olathe|Sterling Heights|New Haven|Miramar|Waco|Thousand Oaks|Cedar Rapids|Charleston|Sioux City|Round Rock|Fargo|Columbia|Coral Springs|Stamford|Concord|Daly City|Richardson|Gainesville|Sugar Land|Clearwater|Hamilton|Billings|Lowell|West Jordan|Allentown|Norwalk|Broken Arrow|Inglewoo|Pueblo|Reading|Lawrence|Santa Clara|Springfield|Greeley|Palm Bay|Westland|Arvada|Topeka|Dearborn|Odessa|Westminster|Aurora|Louisville|Antioch|Evansville|Abilene|Beaumont|Magalia|Carrollton|Fairfield|Provo|West Palm Beach|Thornton|Manchester|Roseville|Surprise|Murfreesboro|Lewisville|Ventura|Lansing|Richmond|Pearland|Flint|Lowell|Tyler|Sandy Springs|West Covina|Hillsboro|Green Bay|Akron|McAllen|Burbank|Renton|Vista|Davie|Marietta|Boulder|Napa|Plantation|Alameda|Compton|South Bend|Brockton|Roanoke|Spokane Valley|Temecula|Sandy|Vacaville|Livermore|Fall River|Clovis|Woodbridge|San Mateo|Hemet|Lake Forest|Redwood City|Bellflower|Lakeland|Merced|Napa|Redding|Chico|Buena Park|Lake Charles|Danbury|Warwick|Eagan|Minnetonka|Shawnee|Southfield|Lynchburg|Longmont|Orem|Ogden|Troy|Tigard|Racine|Bloomington|Fishers|Carmel|Muncie|Danville|Medford|Broomfield|Champaign|Gulfport|Decatur|Elmhurst|Dekalb)\b",
                entity_type="LOCATION",
                pattern_type=PatternType.REGEX,
                confidence=0.8,
                description="US cities"
            ),
        ]
        
        # Date patterns
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
        
        # Money patterns
        money_patterns = [
            EntityPattern(
                pattern=r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b",
                entity_type="MONEY",
                pattern_type=PatternType.REGEX,
                confidence=0.95,
                description="Dollar amounts"
            ),
            EntityPattern(
                pattern=r"\b\d+(?:\.\d+)?\s*(?:million|billion|trillion)\s*dollars?\b",
                entity_type="MONEY",
                pattern_type=PatternType.REGEX,
                confidence=0.9,
                description="Large amounts in words"
            ),
        ]
        
        # Product patterns
        product_patterns = [
            EntityPattern(
                pattern=r"\b(?:iPhone|iPad|MacBook|iMac|Apple Watch|AirPods|Android|Windows|Office|Excel|Word|PowerPoint|Outlook|Teams|Slack|Zoom|Figma|Photoshop|Illustrator|InDesign|Premiere|After Effects)\b",
                entity_type="PRODUCT",
                pattern_type=PatternType.REGEX,
                confidence=0.9,
                description="Popular software and hardware products"
            ),
        ]
        
        # Combine all patterns
        self.patterns.extend(tech_patterns)
        self.patterns.extend(org_patterns)
        self.patterns.extend(location_patterns)
        self.patterns.extend(date_patterns)
        self.patterns.extend(money_patterns)
        self.patterns.extend(product_patterns)
        
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
