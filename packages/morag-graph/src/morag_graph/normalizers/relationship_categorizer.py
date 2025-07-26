"""Relationship type mapping and categorization for OpenIE predicates."""

import asyncio
from typing import List, Dict, Any, Optional, NamedTuple, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from .predicate_normalizer import RelationshipType, NormalizedPredicate
from .entity_linker import LinkedTriplet

logger = structlog.get_logger(__name__)


class RelationshipCategory(Enum):
    """High-level relationship categories for knowledge graph organization."""
    STRUCTURAL = "structural"       # Core structural relationships (is, has, part_of)
    FUNCTIONAL = "functional"       # Functional relationships (works_at, manages, teaches)
    SPATIAL = "spatial"            # Spatial relationships (located_in, near, contains)
    TEMPORAL = "temporal"          # Temporal relationships (before, after, during)
    CAUSAL = "causal"             # Causal relationships (causes, leads_to, results_in)
    SOCIAL = "social"             # Social relationships (knows, friends_with, related_to)
    INFORMATIONAL = "informational" # Information relationships (says, writes, publishes)
    TRANSACTIONAL = "transactional" # Transaction relationships (buys, sells, pays)
    COMPARATIVE = "comparative"    # Comparative relationships (similar_to, better_than)
    DESCRIPTIVE = "descriptive"   # Descriptive relationships (describes, characterizes)


@dataclass
class CategorizedRelationship:
    """Represents a categorized relationship with metadata."""
    predicate: str
    relationship_type: RelationshipType
    relationship_category: RelationshipCategory
    confidence: float
    semantic_weight: float
    directionality: str  # "directed", "undirected", "bidirectional"
    domain_specificity: str  # "general", "domain_specific", "technical"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RelationshipTaxonomy:
    """Hierarchical taxonomy of relationship types and categories."""
    
    def __init__(self):
        self.taxonomy = self._build_taxonomy()
    
    def _build_taxonomy(self) -> Dict[RelationshipCategory, Dict[RelationshipType, Dict[str, Any]]]:
        """Build the relationship taxonomy hierarchy."""
        return {
            RelationshipCategory.STRUCTURAL: {
                RelationshipType.IDENTITY: {
                    "predicates": ["is", "are", "was", "were", "becomes"],
                    "semantic_weight": 0.9,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Identity and classification relationships"
                },
                RelationshipType.POSSESSION: {
                    "predicates": ["has", "owns", "possesses", "contains"],
                    "semantic_weight": 0.8,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Ownership and possession relationships"
                },
                RelationshipType.MEMBERSHIP: {
                    "predicates": ["member_of", "part_of", "belongs_to", "component_of"],
                    "semantic_weight": 0.8,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Membership and containment relationships"
                }
            },
            RelationshipCategory.FUNCTIONAL: {
                RelationshipType.EMPLOYMENT: {
                    "predicates": ["works_at", "employed_by", "staff_at"],
                    "semantic_weight": 0.9,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Employment and work relationships"
                },
                RelationshipType.MANAGEMENT: {
                    "predicates": ["manages", "leads", "supervises", "directs", "heads"],
                    "semantic_weight": 0.8,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Management and leadership relationships"
                },
                RelationshipType.EDUCATION: {
                    "predicates": ["teaches", "studies", "learns", "researches", "instructs"],
                    "semantic_weight": 0.7,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Educational and learning relationships"
                },
                RelationshipType.CREATION: {
                    "predicates": ["creates", "develops", "builds", "produces", "manufactures"],
                    "semantic_weight": 0.8,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Creation and production relationships"
                }
            },
            RelationshipCategory.SPATIAL: {
                RelationshipType.LOCATION: {
                    "predicates": ["located_in", "based_in", "situated_in", "resides_in"],
                    "semantic_weight": 0.8,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Location and spatial positioning relationships"
                }
            },
            RelationshipCategory.TEMPORAL: {
                RelationshipType.TEMPORAL: {
                    "predicates": ["before", "after", "during", "since", "until"],
                    "semantic_weight": 0.7,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Temporal ordering and duration relationships"
                }
            },
            RelationshipCategory.CAUSAL: {
                RelationshipType.CAUSAL: {
                    "predicates": ["causes", "leads_to", "results_in", "triggers", "enables"],
                    "semantic_weight": 0.9,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Causal and consequence relationships"
                }
            },
            RelationshipCategory.INFORMATIONAL: {
                RelationshipType.COMMUNICATION: {
                    "predicates": ["says", "tells", "speaks", "writes", "publishes", "communicates"],
                    "semantic_weight": 0.6,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Communication and information transfer relationships"
                }
            },
            RelationshipCategory.DESCRIPTIVE: {
                RelationshipType.ACTION: {
                    "predicates": ["performs", "executes", "does", "carries_out"],
                    "semantic_weight": 0.7,
                    "directionality": "directed",
                    "domain_specificity": "general",
                    "description": "Action and activity relationships"
                },
                RelationshipType.RELATIONSHIP: {
                    "predicates": ["related_to", "connected_to", "associated_with", "linked_to"],
                    "semantic_weight": 0.5,
                    "directionality": "undirected",
                    "domain_specificity": "general",
                    "description": "General association relationships"
                }
            }
        }


class RelationshipCategorizer:
    """Categorizes and maps relationships based on predicate taxonomy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize relationship categorizer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Configuration
        self.enable_semantic_weighting = self.config.get('enable_semantic_weighting', True)
        self.enable_domain_detection = self.config.get('enable_domain_detection', True)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        self.prefer_specific_categories = self.config.get('prefer_specific_categories', True)
        
        # Initialize taxonomy
        self.taxonomy = RelationshipTaxonomy()
        
        # Build reverse lookup maps
        self._predicate_to_type = self._build_predicate_lookup()
        self._type_to_category = self._build_type_category_lookup()
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rel_categorizer")
        
        logger.info(
            "Relationship categorizer initialized",
            enable_semantic_weighting=self.enable_semantic_weighting,
            enable_domain_detection=self.enable_domain_detection,
            min_confidence_threshold=self.min_confidence_threshold
        )
    
    def _build_predicate_lookup(self) -> Dict[str, Tuple[RelationshipType, RelationshipCategory]]:
        """Build lookup map from predicates to types and categories."""
        lookup = {}
        
        for category, types in self.taxonomy.taxonomy.items():
            for rel_type, type_info in types.items():
                for predicate in type_info["predicates"]:
                    lookup[predicate] = (rel_type, category)
        
        return lookup
    
    def _build_type_category_lookup(self) -> Dict[RelationshipType, RelationshipCategory]:
        """Build lookup map from relationship types to categories."""
        lookup = {}
        
        for category, types in self.taxonomy.taxonomy.items():
            for rel_type in types.keys():
                lookup[rel_type] = category
        
        return lookup
    
    async def categorize_relationships(
        self, 
        normalized_predicates: List[NormalizedPredicate],
        source_doc_id: Optional[str] = None
    ) -> List[CategorizedRelationship]:
        """Categorize normalized predicates into relationship types and categories.
        
        Args:
            normalized_predicates: List of normalized predicates
            source_doc_id: Optional source document ID
            
        Returns:
            List of categorized relationships
            
        Raises:
            ProcessingError: If categorization fails
        """
        if not normalized_predicates:
            return []
        
        try:
            logger.debug(
                "Starting relationship categorization",
                predicate_count=len(normalized_predicates),
                source_doc_id=source_doc_id
            )
            
            # Categorize predicates
            categorized_relationships = []
            for predicate in normalized_predicates:
                try:
                    categorized = await self._categorize_single_predicate(predicate)
                    if categorized:
                        categorized_relationships.append(categorized)
                except Exception as e:
                    logger.warning(
                        "Failed to categorize predicate",
                        predicate=predicate.canonical_form,
                        error=str(e)
                    )
            
            logger.info(
                "Relationship categorization completed",
                input_predicates=len(normalized_predicates),
                categorized_relationships=len(categorized_relationships),
                source_doc_id=source_doc_id
            )
            
            return categorized_relationships
            
        except Exception as e:
            logger.error(
                "Relationship categorization failed",
                error=str(e),
                error_type=type(e).__name__,
                predicate_count=len(normalized_predicates),
                source_doc_id=source_doc_id
            )
            raise ProcessingError(f"Relationship categorization failed: {e}")
    
    async def _categorize_single_predicate(self, predicate: NormalizedPredicate) -> Optional[CategorizedRelationship]:
        """Categorize a single normalized predicate."""
        def categorize_sync():
            # Try direct lookup first
            canonical = predicate.canonical_form
            if canonical in self._predicate_to_type:
                rel_type, category = self._predicate_to_type[canonical]
                
                # Get taxonomy information
                taxonomy_info = self.taxonomy.taxonomy[category][rel_type]
                
                return CategorizedRelationship(
                    predicate=canonical,
                    relationship_type=rel_type,
                    relationship_category=category,
                    confidence=predicate.confidence,
                    semantic_weight=taxonomy_info["semantic_weight"],
                    directionality=taxonomy_info["directionality"],
                    domain_specificity=taxonomy_info["domain_specificity"],
                    metadata={
                        "original_predicate": predicate.original,
                        "normalized_predicate": predicate.normalized,
                        "language": predicate.language,
                        "taxonomy_description": taxonomy_info["description"],
                        "categorization_method": "direct_lookup"
                    }
                )
            
            # Try using the relationship type from predicate normalizer
            if predicate.relationship_type != RelationshipType.OTHER:
                category = self._type_to_category.get(predicate.relationship_type, RelationshipCategory.DESCRIPTIVE)
                
                # Get or estimate taxonomy information
                if category in self.taxonomy.taxonomy and predicate.relationship_type in self.taxonomy.taxonomy[category]:
                    taxonomy_info = self.taxonomy.taxonomy[category][predicate.relationship_type]
                else:
                    # Provide default values
                    taxonomy_info = {
                        "semantic_weight": 0.6,
                        "directionality": "directed",
                        "domain_specificity": "general",
                        "description": f"Relationship of type {predicate.relationship_type.value}"
                    }
                
                return CategorizedRelationship(
                    predicate=canonical,
                    relationship_type=predicate.relationship_type,
                    relationship_category=category,
                    confidence=predicate.confidence * 0.9,  # Slight penalty for indirect mapping
                    semantic_weight=taxonomy_info["semantic_weight"],
                    directionality=taxonomy_info["directionality"],
                    domain_specificity=taxonomy_info["domain_specificity"],
                    metadata={
                        "original_predicate": predicate.original,
                        "normalized_predicate": predicate.normalized,
                        "language": predicate.language,
                        "taxonomy_description": taxonomy_info["description"],
                        "categorization_method": "type_mapping"
                    }
                )
            
            # Fallback: classify as OTHER with low confidence
            return CategorizedRelationship(
                predicate=canonical,
                relationship_type=RelationshipType.OTHER,
                relationship_category=RelationshipCategory.DESCRIPTIVE,
                confidence=predicate.confidence * 0.5,  # Penalty for unknown categorization
                semantic_weight=0.3,
                directionality="directed",
                domain_specificity="general",
                metadata={
                    "original_predicate": predicate.original,
                    "normalized_predicate": predicate.normalized,
                    "language": predicate.language,
                    "taxonomy_description": "Uncategorized relationship",
                    "categorization_method": "fallback"
                }
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, categorize_sync
        )
    
    def get_category_statistics(self, categorized_relationships: List[CategorizedRelationship]) -> Dict[str, Any]:
        """Get statistics about relationship categories."""
        if not categorized_relationships:
            return {}
        
        # Count by category
        category_counts = {}
        type_counts = {}
        confidence_by_category = {}
        semantic_weights = []
        
        for rel in categorized_relationships:
            # Category counts
            category = rel.relationship_category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Type counts
            rel_type = rel.relationship_type.value
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
            
            # Confidence by category
            if category not in confidence_by_category:
                confidence_by_category[category] = []
            confidence_by_category[category].append(rel.confidence)
            
            # Semantic weights
            semantic_weights.append(rel.semantic_weight)
        
        # Calculate averages
        avg_confidence_by_category = {
            category: sum(confidences) / len(confidences)
            for category, confidences in confidence_by_category.items()
        }
        
        return {
            "total_relationships": len(categorized_relationships),
            "category_distribution": category_counts,
            "type_distribution": type_counts,
            "average_confidence_by_category": avg_confidence_by_category,
            "overall_average_confidence": sum(rel.confidence for rel in categorized_relationships) / len(categorized_relationships),
            "average_semantic_weight": sum(semantic_weights) / len(semantic_weights),
            "directionality_distribution": {
                direction: sum(1 for rel in categorized_relationships if rel.directionality == direction)
                for direction in ["directed", "undirected", "bidirectional"]
            }
        }
    
    def get_taxonomy_info(self) -> Dict[str, Any]:
        """Get information about the relationship taxonomy."""
        taxonomy_info = {
            "categories": {},
            "total_types": 0,
            "total_predicates": 0
        }
        
        for category, types in self.taxonomy.taxonomy.items():
            category_info = {
                "types": {},
                "type_count": len(types),
                "predicate_count": 0
            }
            
            for rel_type, type_info in types.items():
                category_info["types"][rel_type.value] = {
                    "predicates": type_info["predicates"],
                    "predicate_count": len(type_info["predicates"]),
                    "semantic_weight": type_info["semantic_weight"],
                    "directionality": type_info["directionality"],
                    "domain_specificity": type_info["domain_specificity"],
                    "description": type_info["description"]
                }
                category_info["predicate_count"] += len(type_info["predicates"])
            
            taxonomy_info["categories"][category.value] = category_info
            taxonomy_info["total_types"] += category_info["type_count"]
            taxonomy_info["total_predicates"] += category_info["predicate_count"]
        
        return taxonomy_info
    
    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("Relationship categorizer closed")
        except Exception as e:
            logger.warning("Error during relationship categorizer cleanup", error=str(e))
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
