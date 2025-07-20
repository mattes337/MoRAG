"""Semantic relation enhancer for creating richer, more meaningful relation types."""

import re
import logging
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)

class RelationCategory(str, Enum):
    """Enhanced categories for semantic relations."""
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    FUNCTIONAL = "functional"
    SPATIAL = "spatial"
    SEMANTIC = "semantic"
    COLLABORATIVE = "collaborative"
    KNOWLEDGE = "knowledge"
    CREATION = "creation"
    COMMUNICATION = "communication"
    MEDICAL = "medical"
    TECHNICAL = "technical"
    BUSINESS = "business"
    ACADEMIC = "academic"

class SemanticRelationEnhancer:
    """Enhances relation extraction with richer semantic types."""
    
    def __init__(self):
        """Initialize the semantic relation enhancer."""
        self._relation_patterns = self._build_relation_patterns()
        self._domain_indicators = self._build_domain_indicators()
        
    def _build_relation_patterns(self) -> Dict[RelationCategory, List[Dict]]:
        """Build patterns for detecting specific relation types."""
        return {
            RelationCategory.CAUSAL: [
                {
                    "patterns": [r"(.+)\s+(causes?|leads?\s+to|results?\s+in|triggers?|induces?)\s+(.+)"],
                    "relation_types": ["causes", "triggers", "induces", "results_in"]
                },
                {
                    "patterns": [r"(.+)\s+(prevents?|blocks?|inhibits?|stops?)\s+(.+)"],
                    "relation_types": ["prevents", "inhibits", "blocks"]
                },
                {
                    "patterns": [r"(.+)\s+(enables?|allows?|facilitates?|promotes?)\s+(.+)"],
                    "relation_types": ["enables", "facilitates", "promotes"]
                }
            ],
            RelationCategory.TEMPORAL: [
                {
                    "patterns": [r"(.+)\s+(before|prior\s+to|precedes?)\s+(.+)"],
                    "relation_types": ["precedes", "occurs_before"]
                },
                {
                    "patterns": [r"(.+)\s+(after|following|succeeds?)\s+(.+)"],
                    "relation_types": ["follows", "occurs_after"]
                },
                {
                    "patterns": [r"(.+)\s+(during|while|throughout)\s+(.+)"],
                    "relation_types": ["occurs_during", "concurrent_with"]
                }
            ],
            RelationCategory.HIERARCHICAL: [
                {
                    "patterns": [r"(.+)\s+(contains?|includes?|comprises?|encompasses?)\s+(.+)"],
                    "relation_types": ["contains", "includes", "comprises"]
                },
                {
                    "patterns": [r"(.+)\s+(part\s+of|component\s+of|belongs\s+to)\s+(.+)"],
                    "relation_types": ["part_of", "component_of", "belongs_to"]
                },
                {
                    "patterns": [r"(.+)\s+(manages?|supervises?|oversees?|controls?)\s+(.+)"],
                    "relation_types": ["manages", "supervises", "controls"]
                }
            ],
            RelationCategory.FUNCTIONAL: [
                {
                    "patterns": [r"(.+)\s+(uses?|utilizes?|employs?|applies?)\s+(.+)"],
                    "relation_types": ["uses", "utilizes", "employs"]
                },
                {
                    "patterns": [r"(.+)\s+(operates?|functions?\s+as|works?\s+as)\s+(.+)"],
                    "relation_types": ["operates", "functions_as"]
                },
                {
                    "patterns": [r"(.+)\s+(depends?\s+on|relies?\s+on|requires?)\s+(.+)"],
                    "relation_types": ["depends_on", "requires", "relies_on"]
                }
            ],
            RelationCategory.MEDICAL: [
                {
                    "patterns": [r"(.+)\s+(treats?|cures?|heals?|therapies?)\s+(.+)"],
                    "relation_types": ["treats", "cures", "provides_therapy_for"]
                },
                {
                    "patterns": [r"(.+)\s+(diagnoses?|detects?|identifies?)\s+(.+)"],
                    "relation_types": ["diagnoses", "detects", "identifies"]
                },
                {
                    "patterns": [r"(.+)\s+(symptoms?\s+of|indicates?|manifests?\s+as)\s+(.+)"],
                    "relation_types": ["symptom_of", "indicates", "manifests_as"]
                },
                {
                    "patterns": [r"(.+)\s+(interacts?\s+with|affects?|influences?)\s+(.+)"],
                    "relation_types": ["interacts_with", "affects", "influences"]
                }
            ],
            RelationCategory.TECHNICAL: [
                {
                    "patterns": [r"(.+)\s+(implements?|executes?|runs?)\s+(.+)"],
                    "relation_types": ["implements", "executes", "runs"]
                },
                {
                    "patterns": [r"(.+)\s+(connects?\s+to|interfaces?\s+with|communicates?\s+with)\s+(.+)"],
                    "relation_types": ["connects_to", "interfaces_with", "communicates_with"]
                },
                {
                    "patterns": [r"(.+)\s+(processes?|transforms?|converts?)\s+(.+)"],
                    "relation_types": ["processes", "transforms", "converts"]
                }
            ],
            RelationCategory.KNOWLEDGE: [
                {
                    "patterns": [r"(.+)\s+(teaches?|explains?|describes?|defines?)\s+(.+)"],
                    "relation_types": ["teaches", "explains", "describes", "defines"]
                },
                {
                    "patterns": [r"(.+)\s+(learns?\s+from|studies?|researches?)\s+(.+)"],
                    "relation_types": ["learns_from", "studies", "researches"]
                },
                {
                    "patterns": [r"(.+)\s+(demonstrates?|shows?|proves?|validates?)\s+(.+)"],
                    "relation_types": ["demonstrates", "proves", "validates"]
                }
            ],
            RelationCategory.CREATION: [
                {
                    "patterns": [r"(.+)\s+(creates?|produces?|generates?|builds?)\s+(.+)"],
                    "relation_types": ["creates", "produces", "generates", "builds"]
                },
                {
                    "patterns": [r"(.+)\s+(develops?|designs?|constructs?)\s+(.+)"],
                    "relation_types": ["develops", "designs", "constructs"]
                },
                {
                    "patterns": [r"(.+)\s+(originates?\s+from|derives?\s+from|based\s+on)\s+(.+)"],
                    "relation_types": ["originates_from", "derives_from", "based_on"]
                }
            ]
        }
    
    def _build_domain_indicators(self) -> Dict[str, List[str]]:
        """Build indicators for different domains."""
        return {
            "medical": [
                "patient", "treatment", "therapy", "diagnosis", "symptom", "disease", 
                "medication", "drug", "clinical", "medical", "health", "doctor", 
                "physician", "nurse", "hospital", "clinic", "surgery", "procedure"
            ],
            "technical": [
                "system", "software", "hardware", "algorithm", "code", "program",
                "application", "database", "server", "network", "protocol", "api",
                "framework", "library", "technology", "platform", "interface"
            ],
            "business": [
                "company", "organization", "business", "market", "customer", "client",
                "revenue", "profit", "investment", "strategy", "management", "team",
                "department", "project", "product", "service", "sales", "marketing"
            ],
            "academic": [
                "research", "study", "analysis", "experiment", "theory", "hypothesis",
                "methodology", "findings", "conclusion", "paper", "publication",
                "university", "professor", "student", "course", "education", "learning"
            ]
        }

    @lru_cache(maxsize=500)
    def enhance_relation_type(self, source_entity: str, target_entity: str, 
                            context: str, base_relation_type: str = "relates_to") -> str:
        """
        Enhance a basic relation type with semantic information.
        
        Args:
            source_entity: Source entity name
            target_entity: Target entity name
            context: Context where the relation was found
            base_relation_type: Base relation type to enhance
            
        Returns:
            Enhanced relation type with semantic meaning
        """
        # If we already have a specific relation type, check if we can enhance it further
        if base_relation_type and base_relation_type not in ["relates_to", "mentions", "associated_with"]:
            return base_relation_type
            
        # Detect domain context
        domain = self._detect_domain(context, source_entity, target_entity)
        
        # Try to extract specific relation from context
        specific_relation = self._extract_specific_relation(context, source_entity, target_entity, domain)
        
        if specific_relation:
            return specific_relation
            
        # If no specific relation found, return enhanced generic relation
        if domain and domain != "general":
            return f"{domain}_{base_relation_type}"
        
        return base_relation_type

    def _detect_domain(self, context: str, source_entity: str, target_entity: str) -> str:
        """Detect the domain of the relation based on context and entities."""
        context_lower = context.lower()
        entities_lower = f"{source_entity} {target_entity}".lower()
        combined_text = f"{context_lower} {entities_lower}"
        
        domain_scores = {}
        for domain, indicators in self._domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in combined_text)
            if score > 0:
                domain_scores[domain] = score
                
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return "general"

    def _extract_specific_relation(self, context: str, source_entity: str, 
                                 target_entity: str, domain: str) -> Optional[str]:
        """Extract specific relation type from context."""
        context_lower = context.lower()
        
        # Try domain-specific patterns first
        if domain in ["medical", "technical", "business", "academic"]:
            category = RelationCategory(domain)
            if category in self._relation_patterns:
                for pattern_group in self._relation_patterns[category]:
                    for pattern in pattern_group["patterns"]:
                        if re.search(pattern, context_lower):
                            # Extract the verb/relation indicator from the match
                            match = re.search(pattern, context_lower)
                            if match and len(match.groups()) >= 2:
                                relation_indicator = match.group(2).strip()
                                return self._normalize_relation_type(relation_indicator)
        
        # Try general patterns
        for category, pattern_groups in self._relation_patterns.items():
            if category.value in ["medical", "technical", "business", "academic"]:
                continue  # Already tried domain-specific
                
            for pattern_group in pattern_groups:
                for pattern in pattern_group["patterns"]:
                    if re.search(pattern, context_lower):
                        match = re.search(pattern, context_lower)
                        if match and len(match.groups()) >= 2:
                            relation_indicator = match.group(2).strip()
                            return self._normalize_relation_type(relation_indicator)
        
        return None

    def _normalize_relation_type(self, relation_indicator: str) -> str:
        """Normalize relation indicator to a standard relation type."""
        # Clean up the relation indicator
        normalized = relation_indicator.lower().strip()
        
        # Remove common words and normalize
        normalized = re.sub(r'\s+(to|with|of|as|in|on|at|by|for)\s*$', '', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        
        # Handle common variations
        relation_mappings = {
            "cause": "causes",
            "lead_to": "leads_to",
            "result_in": "results_in",
            "prevent": "prevents",
            "enable": "enables",
            "contain": "contains",
            "include": "includes",
            "manage": "manages",
            "use": "uses",
            "treat": "treats",
            "diagnose": "diagnoses",
            "create": "creates",
            "produce": "produces",
            "develop": "develops",
            "implement": "implements",
            "connect": "connects_to",
            "interface": "interfaces_with",
            "communicate": "communicates_with",
            "teach": "teaches",
            "explain": "explains",
            "learn_from": "learns_from",
            "study": "studies",
        }
        
        return relation_mappings.get(normalized, normalized)

    def get_relation_suggestions(self, domain: str) -> List[str]:
        """Get suggested relation types for a specific domain."""
        if domain in self._relation_patterns:
            suggestions = []
            for pattern_group in self._relation_patterns[RelationCategory(domain)]:
                suggestions.extend(pattern_group["relation_types"])
            return list(set(suggestions))
        
        # Return general suggestions
        return [
            "causes", "prevents", "enables", "contains", "part_of", "uses",
            "manages", "creates", "affects", "influences", "depends_on",
            "precedes", "follows", "occurs_during", "connects_to", "implements"
        ]

    def validate_relation_type(self, relation_type: str) -> bool:
        """Validate if a relation type is meaningful and well-formed."""
        if not relation_type or len(relation_type) < 2:
            return False
            
        # Check for basic formatting
        if not re.match(r'^[a-z][a-z0-9_]*$', relation_type):
            return False
            
        # Avoid overly generic types
        generic_types = {"relates_to", "associated_with", "connected_to", "linked_to"}
        if relation_type in generic_types:
            return False
            
        return True
