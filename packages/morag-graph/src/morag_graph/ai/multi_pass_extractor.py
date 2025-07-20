"""Multi-pass relation extraction system with semantic enhancement."""

import asyncio
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.ai import Relation
from ..models import Entity as GraphEntity, Relation as GraphRelation
from .enhanced_relation_agent import EnhancedRelationExtractionAgent
from .semantic_analyzer import SemanticRelationAnalyzer, RelationEnhancement
from .domain_extractors import DomainExtractorFactory, BaseDomainExtractor

logger = structlog.get_logger(__name__)


class ExtractionPass(str, Enum):
    """Extraction pass types."""
    BASIC = "basic"
    SEMANTIC = "semantic"
    DOMAIN = "domain"
    CONTEXTUAL = "contextual"
    VALIDATION = "validation"


@dataclass
class ExtractionResult:
    """Result from a single extraction pass."""
    pass_type: ExtractionPass
    relations: List[Relation]
    enhancements: List[RelationEnhancement]
    metadata: Dict[str, any]


@dataclass
class MultiPassResult:
    """Complete multi-pass extraction result."""
    final_relations: List[GraphRelation]
    pass_results: List[ExtractionResult]
    statistics: Dict[str, any]
    domain: str
    confidence_distribution: Dict[str, int]


class MultiPassRelationExtractor:
    """Multi-pass relation extraction system with progressive enhancement."""

    def __init__(
        self,
        min_confidence: float = 0.6,
        enable_semantic_analysis: bool = True,
        enable_domain_extraction: bool = True,
        enable_contextual_enhancement: bool = True,
        max_relations_per_pass: int = 100,
        language: Optional[str] = None
    ):
        """Initialize the multi-pass extractor."""
        self.min_confidence = min_confidence
        self.enable_semantic_analysis = enable_semantic_analysis
        self.enable_domain_extraction = enable_domain_extraction
        self.enable_contextual_enhancement = enable_contextual_enhancement
        self.max_relations_per_pass = max_relations_per_pass
        self.language = language or "en"
        
        self.logger = structlog.get_logger(__name__)
        
        # Initialize components
        self.enhanced_agent = EnhancedRelationExtractionAgent(
            min_confidence=min_confidence * 0.8,  # Lower threshold for initial extraction
            enable_semantic_analysis=enable_semantic_analysis,
            enable_domain_detection=enable_domain_extraction,
            language=language
        )
        
        if enable_semantic_analysis:
            self.semantic_analyzer = SemanticRelationAnalyzer()
        else:
            self.semantic_analyzer = None

    async def extract_relations_multi_pass(
        self,
        text: str,
        entities: Optional[List[GraphEntity]] = None,
        source_doc_id: Optional[str] = None,
        domain_hint: Optional[str] = None
    ) -> MultiPassResult:
        """Extract relations using multi-pass approach."""
        
        if not text or not text.strip():
            return MultiPassResult(
                final_relations=[],
                pass_results=[],
                statistics={'total_passes': 0, 'total_relations': 0},
                domain='unknown',
                confidence_distribution={}
            )

        self.logger.info(
            "Starting multi-pass relation extraction",
            text_length=len(text),
            num_entities=len(entities) if entities else 0,
            domain_hint=domain_hint
        )

        pass_results = []
        all_relations = []
        
        try:
            # Pass 1: Basic Enhanced Extraction
            basic_result = await self._pass_1_basic_extraction(
                text, entities, source_doc_id, domain_hint
            )
            pass_results.append(basic_result)
            all_relations.extend(basic_result.relations)
            
            detected_domain = basic_result.metadata.get('domain', 'general')
            
            # Pass 2: Semantic Enhancement
            if self.enable_semantic_analysis and self.semantic_analyzer:
                semantic_result = await self._pass_2_semantic_enhancement(
                    text, all_relations, entities
                )
                pass_results.append(semantic_result)
                # Replace relations with enhanced versions
                all_relations = semantic_result.relations
            
            # Pass 3: Domain-Specific Extraction
            if self.enable_domain_extraction and detected_domain != 'general':
                domain_result = await self._pass_3_domain_extraction(
                    text, entities, all_relations, detected_domain
                )
                pass_results.append(domain_result)
                all_relations.extend(domain_result.relations)
            
            # Pass 4: Contextual Enhancement
            if self.enable_contextual_enhancement:
                contextual_result = await self._pass_4_contextual_enhancement(
                    text, all_relations, entities
                )
                pass_results.append(contextual_result)
                all_relations = contextual_result.relations
            
            # Pass 5: Final Validation and Filtering
            validation_result = await self._pass_5_validation(
                all_relations, entities, source_doc_id
            )
            pass_results.append(validation_result)
            
            # Convert to GraphRelation objects
            final_graph_relations = []
            for relation in validation_result.relations:
                if relation.confidence >= self.min_confidence:
                    graph_relation = self._convert_to_graph_relation(
                        relation, entities, source_doc_id, detected_domain
                    )
                    if graph_relation:
                        final_graph_relations.append(graph_relation)
            
            # Generate statistics
            statistics = self._generate_statistics(pass_results, final_graph_relations)
            confidence_dist = self._calculate_confidence_distribution(final_graph_relations)
            
            self.logger.info(
                "Multi-pass extraction completed",
                total_passes=len(pass_results),
                final_relations=len(final_graph_relations),
                domain=detected_domain
            )
            
            return MultiPassResult(
                final_relations=final_graph_relations,
                pass_results=pass_results,
                statistics=statistics,
                domain=detected_domain,
                confidence_distribution=confidence_dist
            )
            
        except Exception as e:
            self.logger.error(
                "Multi-pass extraction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _pass_1_basic_extraction(
        self,
        text: str,
        entities: Optional[List[GraphEntity]],
        source_doc_id: Optional[str],
        domain_hint: Optional[str]
    ) -> ExtractionResult:
        """Pass 1: Basic enhanced extraction."""
        self.logger.debug("Starting Pass 1: Basic extraction")
        
        relations = await self.enhanced_agent.extract_enhanced_relations(
            text=text,
            entities=entities,
            source_doc_id=source_doc_id,
            domain_hint=domain_hint
        )
        
        # Convert GraphRelation back to Relation for processing
        relation_objects = []
        detected_domain = 'general'
        
        for graph_rel in relations:
            # Extract domain from first relation's attributes
            if not detected_domain or detected_domain == 'general':
                detected_domain = graph_rel.attributes.get('domain', 'general')
            
            relation = Relation(
                source_entity=graph_rel.attributes.get('source_entity_name', ''),
                target_entity=graph_rel.attributes.get('target_entity_name', ''),
                relation_type=graph_rel.type,
                confidence=graph_rel.confidence,
                context=graph_rel.attributes.get('description', '')
            )
            relation_objects.append(relation)
        
        return ExtractionResult(
            pass_type=ExtractionPass.BASIC,
            relations=relation_objects,
            enhancements=[],
            metadata={
                'domain': detected_domain,
                'extraction_method': 'enhanced_agent',
                'relations_count': len(relation_objects)
            }
        )

    async def _pass_2_semantic_enhancement(
        self,
        text: str,
        relations: List[Relation],
        entities: Optional[List[GraphEntity]]
    ) -> ExtractionResult:
        """Pass 2: Semantic enhancement of relations."""
        self.logger.debug("Starting Pass 2: Semantic enhancement")
        
        enhanced_relations = []
        enhancements = []
        
        for relation in relations:
            enhancement = self.semantic_analyzer.analyze_relation_context(
                relation, text, entities
            )
            enhancements.append(enhancement)
            
            # Create enhanced relation
            enhanced_relation = Relation(
                source_entity=relation.source_entity,
                target_entity=relation.target_entity,
                relation_type=enhancement.enhanced_type,
                confidence=min(1.0, relation.confidence + enhancement.confidence_adjustment),
                context=relation.context
            )
            enhanced_relations.append(enhanced_relation)
        
        return ExtractionResult(
            pass_type=ExtractionPass.SEMANTIC,
            relations=enhanced_relations,
            enhancements=enhancements,
            metadata={
                'enhancements_applied': len([e for e in enhancements if e.enhanced_type != e.original_type]),
                'avg_confidence_change': sum(e.confidence_adjustment for e in enhancements) / len(enhancements) if enhancements else 0
            }
        )

    async def _pass_3_domain_extraction(
        self,
        text: str,
        entities: Optional[List[GraphEntity]],
        existing_relations: List[Relation],
        domain: str
    ) -> ExtractionResult:
        """Pass 3: Domain-specific extraction."""
        self.logger.debug(f"Starting Pass 3: Domain extraction for {domain}")
        
        domain_extractor = DomainExtractorFactory.create_extractor(domain)
        if not domain_extractor or not entities:
            return ExtractionResult(
                pass_type=ExtractionPass.DOMAIN,
                relations=[],
                enhancements=[],
                metadata={'domain': domain, 'extractor_available': False}
            )
        
        domain_relations = domain_extractor.extract_domain_relations(
            text, entities, existing_relations
        )
        
        return ExtractionResult(
            pass_type=ExtractionPass.DOMAIN,
            relations=domain_relations,
            enhancements=[],
            metadata={
                'domain': domain,
                'extractor_available': True,
                'domain_relations_found': len(domain_relations)
            }
        )

    async def _pass_4_contextual_enhancement(
        self,
        text: str,
        relations: List[Relation],
        entities: Optional[List[GraphEntity]]
    ) -> ExtractionResult:
        """Pass 4: Contextual enhancement."""
        self.logger.debug("Starting Pass 4: Contextual enhancement")
        
        # For now, this is a placeholder for future contextual analysis
        # Could include: co-reference resolution, entity linking, context propagation
        
        enhanced_relations = []
        for relation in relations:
            # Apply contextual enhancements here
            enhanced_relations.append(relation)
        
        return ExtractionResult(
            pass_type=ExtractionPass.CONTEXTUAL,
            relations=enhanced_relations,
            enhancements=[],
            metadata={'contextual_enhancements': 0}
        )

    async def _pass_5_validation(
        self,
        relations: List[Relation],
        entities: Optional[List[GraphEntity]],
        source_doc_id: Optional[str]
    ) -> ExtractionResult:
        """Pass 5: Final validation and filtering."""
        self.logger.debug("Starting Pass 5: Validation")
        
        validated_relations = []
        
        # Remove duplicates
        seen_relations = set()
        for relation in relations:
            relation_key = (
                relation.source_entity.lower(),
                relation.target_entity.lower(),
                relation.relation_type
            )
            
            if relation_key not in seen_relations:
                seen_relations.add(relation_key)
                validated_relations.append(relation)
        
        # Sort by confidence
        validated_relations.sort(key=lambda r: r.confidence, reverse=True)
        
        # Limit number of relations if needed
        if len(validated_relations) > self.max_relations_per_pass:
            validated_relations = validated_relations[:self.max_relations_per_pass]
        
        return ExtractionResult(
            pass_type=ExtractionPass.VALIDATION,
            relations=validated_relations,
            enhancements=[],
            metadata={
                'duplicates_removed': len(relations) - len(validated_relations),
                'final_count': len(validated_relations)
            }
        )

    def _convert_to_graph_relation(
        self,
        relation: Relation,
        entities: Optional[List[GraphEntity]],
        source_doc_id: Optional[str],
        domain: str
    ) -> Optional[GraphRelation]:
        """Convert Relation to GraphRelation."""
        # Create entity name to ID mapping
        entity_name_to_id = {}
        if entities:
            entity_name_to_id = {
                entity.name.lower().strip(): entity.id for entity in entities
            }

        # Resolve entity IDs
        source_key = relation.source_entity.lower().strip()
        target_key = relation.target_entity.lower().strip()

        source_entity_id = entity_name_to_id.get(source_key)
        target_entity_id = entity_name_to_id.get(target_key)

        if not source_entity_id or not target_entity_id:
            return None

        # Enhanced attributes
        attributes = {
            'description': relation.context or f"Relation: {relation.relation_type}",
            'evidence_text': relation.context,
            'domain': domain,
            'extraction_method': 'multi_pass_enhanced',
            'source_entity_name': relation.source_entity,
            'target_entity_name': relation.target_entity
        }

        return GraphRelation(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            type=relation.relation_type,
            confidence=relation.confidence,
            source_doc_id=source_doc_id,
            attributes=attributes
        )

    def _generate_statistics(
        self,
        pass_results: List[ExtractionResult],
        final_relations: List[GraphRelation]
    ) -> Dict[str, any]:
        """Generate extraction statistics."""
        stats = {
            'total_passes': len(pass_results),
            'total_relations': len(final_relations),
            'relations_by_pass': {},
            'enhancements_by_pass': {},
            'avg_confidence': 0.0,
            'relation_types': {}
        }

        # Statistics by pass
        for result in pass_results:
            pass_name = result.pass_type.value
            stats['relations_by_pass'][pass_name] = len(result.relations)
            stats['enhancements_by_pass'][pass_name] = len(result.enhancements)

        # Overall statistics
        if final_relations:
            stats['avg_confidence'] = sum(r.confidence for r in final_relations) / len(final_relations)

            # Count relation types
            for relation in final_relations:
                rel_type = relation.type
                stats['relation_types'][rel_type] = stats['relation_types'].get(rel_type, 0) + 1

        return stats

    def _calculate_confidence_distribution(
        self,
        relations: List[GraphRelation]
    ) -> Dict[str, int]:
        """Calculate confidence distribution."""
        distribution = {
            'very_high': 0,  # 0.9+
            'high': 0,       # 0.8-0.89
            'medium': 0,     # 0.6-0.79
            'low': 0         # <0.6
        }

        for relation in relations:
            conf = relation.confidence
            if conf >= 0.9:
                distribution['very_high'] += 1
            elif conf >= 0.8:
                distribution['high'] += 1
            elif conf >= 0.6:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1

        return distribution

    def get_extraction_summary(self, result: MultiPassResult) -> str:
        """Get a human-readable summary of the extraction."""
        summary_lines = [
            f"Multi-Pass Relation Extraction Summary",
            f"=====================================",
            f"Domain: {result.domain}",
            f"Total Relations: {result.statistics['total_relations']}",
            f"Passes Completed: {result.statistics['total_passes']}",
            f"Average Confidence: {result.statistics['avg_confidence']:.3f}",
            "",
            "Relations by Pass:",
        ]

        for pass_name, count in result.statistics['relations_by_pass'].items():
            summary_lines.append(f"  {pass_name}: {count}")

        summary_lines.extend([
            "",
            "Confidence Distribution:",
        ])

        for level, count in result.confidence_distribution.items():
            summary_lines.append(f"  {level}: {count}")

        if result.statistics['relation_types']:
            summary_lines.extend([
                "",
                "Top Relation Types:",
            ])

            sorted_types = sorted(
                result.statistics['relation_types'].items(),
                key=lambda x: x[1],
                reverse=True
            )

            for rel_type, count in sorted_types[:5]:
                summary_lines.append(f"  {rel_type}: {count}")

        return "\n".join(summary_lines)
