"""Enhanced relation extraction agent with semantic depth and domain awareness."""

import asyncio
from typing import Type, List, Optional, Dict, Any, Tuple, Set
from enum import Enum
import structlog

from morag_core.ai import MoRAGBaseAgent, RelationExtractionResult, Relation, ConfidenceLevel
from ..models import Entity as GraphEntity, Relation as GraphRelation

logger = structlog.get_logger(__name__)


class RelationCategory(str, Enum):
    """Categories of relations for semantic analysis."""
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
    DOMAIN_SPECIFIC = "domain_specific"


class DomainType(str, Enum):
    """Domain types for specialized relation extraction."""
    MEDICAL = "medical"
    TECHNICAL = "technical"
    BUSINESS = "business"
    ACADEMIC = "academic"
    LEGAL = "legal"
    GENERAL = "general"


class EnhancedRelationExtractionAgent(MoRAGBaseAgent[RelationExtractionResult]):
    """Enhanced PydanticAI agent for extracting deep, meaningful relations."""

    def __init__(
        self,
        min_confidence: float = 0.6,
        enable_semantic_analysis: bool = True,
        enable_domain_detection: bool = True,
        enable_causal_analysis: bool = True,
        enable_temporal_analysis: bool = True,
        language: Optional[str] = None,
        **kwargs
    ):
        """Initialize the enhanced relation extraction agent."""
        super().__init__(**kwargs)
        self.min_confidence = min_confidence
        self.enable_semantic_analysis = enable_semantic_analysis
        self.enable_domain_detection = enable_domain_detection
        self.enable_causal_analysis = enable_causal_analysis
        self.enable_temporal_analysis = enable_temporal_analysis
        self.language = language or "en"
        self.logger = structlog.get_logger(__name__)

        # Relation type mappings by category
        self.relation_categories = self._build_relation_categories()
        self.domain_patterns = self._build_domain_patterns()

    def _build_relation_categories(self) -> Dict[RelationCategory, List[str]]:
        """Build mapping of relation categories - now empty to allow full AI determination."""
        # Return empty categories to let AI determine relation types dynamically
        return {
            RelationCategory.CAUSAL: [],
            RelationCategory.TEMPORAL: [],
            RelationCategory.HIERARCHICAL: [],
            RelationCategory.FUNCTIONAL: [],
            RelationCategory.SPATIAL: [],
            RelationCategory.SEMANTIC: [],
            RelationCategory.COLLABORATIVE: [],
            RelationCategory.KNOWLEDGE: [],
            RelationCategory.CREATION: [],
            RelationCategory.COMMUNICATION: []
        }

    def _build_domain_patterns(self) -> Dict[DomainType, Dict[str, List[str]]]:
        """Build domain-specific patterns for relation extraction - now only indicators."""
        return {
            DomainType.MEDICAL: {
                "relations": [],  # No predefined relations - let AI decide
                "indicators": [
                    "patient", "doctor", "medicine", "treatment", "diagnosis",
                    "symptom", "disease", "therapy", "medication", "clinical"
                ]
            },
            DomainType.TECHNICAL: {
                "relations": [],  # No predefined relations - let AI decide
                "indicators": [
                    "software", "system", "code", "function", "class",
                    "method", "API", "framework", "library", "algorithm"
                ]
            },
            DomainType.BUSINESS: {
                "relations": [],  # No predefined relations - let AI decide
                "indicators": [
                    "company", "business", "market", "customer", "revenue",
                    "profit", "investment", "acquisition", "partnership"
                ]
            },
            DomainType.ACADEMIC: {
                "relations": [],  # No predefined relations - let AI decide
                "indicators": [
                    "research", "study", "analysis", "experiment", "theory",
                    "hypothesis", "methodology", "findings", "conclusion"
                ]
            }
        }

    def get_system_prompt(self) -> str:
        """Get enhanced system prompt for deep relation extraction."""
        base_prompt = f"""You are an expert relation extraction agent with advanced semantic understanding. Your task is to identify meaningful, deep relationships between entities that go beyond surface-level mentions.

ENHANCED EXTRACTION PRINCIPLES:
1. SEMANTIC DEPTH: Look for underlying meanings, not just surface connections
2. CAUSAL UNDERSTANDING: Identify cause-effect relationships and dependencies
3. TEMPORAL AWARENESS: Recognize sequence, timing, and temporal relationships
4. DOMAIN EXPERTISE: Apply domain-specific knowledge for specialized relationships
5. CONTEXTUAL ANALYSIS: Use surrounding context to infer implicit relationships

DYNAMIC RELATION TYPE DETERMINATION:
You must determine the most appropriate relation type based ENTIRELY on the context and semantic meaning.
DO NOT use predefined relation types. Instead, analyze the text and determine what kind of relationship
actually exists between the entities.

RELATION CATEGORIES TO CONSIDER (but determine specific types dynamically):
- CAUSAL: What causes what? What prevents what? What enables what?
- TEMPORAL: What happens before/after what? What occurs during what?
- HIERARCHICAL: What manages/owns/contains what? What belongs to what?
- FUNCTIONAL: What operates/controls/transforms what? How do things work together?
- SPATIAL: Where are things located? What connects to what?
- SEMANTIC: What is similar/different? What specializes/generalizes what?
- COLLABORATIVE: What works with what? What competes with what?
- KNOWLEDGE: What teaches/explains what? What proves/contradicts what?
- CREATION: What creates/produces/builds what?
- COMMUNICATION: What informs/discusses what?

EXTRACTION GUIDELINES:
1. For each relation, determine:
   - source_entity: The entity initiating or being the subject of the relationship
   - target_entity: The entity receiving or being the object of the relationship
   - relation_type: A descriptive, context-specific relation type that captures the exact nature of the relationship
   - confidence: Your confidence (0.0-1.0) based on textual evidence
   - context: The specific text supporting this relationship

2. RELATION TYPE CREATION:
   - Create relation types that precisely describe the relationship
   - Use clear, descriptive language that captures the semantic meaning
   - Be specific about the nature of the relationship (e.g., "therapeutically_treats" vs "surgically_treats")
   - Consider the direction and strength of the relationship
   - Use verbs that accurately represent the action or connection

3. PRIORITIZE MEANINGFUL RELATIONS:
   - Focus on relationships that add semantic value
   - Look for causal, functional, and hierarchical connections
   - Identify temporal sequences and dependencies
   - Consider domain-specific relationship patterns

4. AVOID SHALLOW RELATIONS:
   - Don't use generic types like "MENTIONS" or "RELATED_TO" unless absolutely necessary
   - Look for the WHY and HOW behind connections
   - Consider the direction and nature of influence
   - Be specific about the type of relationship

5. EVIDENCE-BASED EXTRACTION:
   - Only extract relations with clear textual support
   - Higher confidence for explicit statements
   - Lower confidence for inferred relationships
   - Include the supporting text snippet

LANGUAGE REQUIREMENTS:
- Relation types: ALWAYS use English
- Descriptions: ALWAYS use English
- Entity names: Use normalized forms matching known entities
- Be creative but precise in relation type naming
"""

        return base_prompt

    def _get_category_guidance(self, category: RelationCategory) -> str:
        """Get specific guidance for each relation category."""
        guidance = {
            RelationCategory.CAUSAL: "- Look for cause-effect language: 'because', 'due to', 'results in', 'leads to'\n- Identify prevention, enablement, and triggering relationships\n",
            RelationCategory.TEMPORAL: "- Identify sequence indicators: 'before', 'after', 'during', 'while'\n- Look for process steps and chronological relationships\n",
            RelationCategory.HIERARCHICAL: "- Identify organizational structures and ownership\n- Look for management, supervision, and containment relationships\n",
            RelationCategory.FUNCTIONAL: "- Focus on operational relationships and dependencies\n- Identify control, regulation, and transformation processes\n",
            RelationCategory.SPATIAL: "- Look for location and positioning relationships\n- Identify physical connections and separations\n",
            RelationCategory.SEMANTIC: "- Identify conceptual relationships and categorizations\n- Look for similarity, opposition, and specialization\n",
            RelationCategory.COLLABORATIVE: "- Focus on partnership and cooperation patterns\n- Identify support, competition, and assistance relationships\n",
            RelationCategory.KNOWLEDGE: "- Look for learning, teaching, and validation relationships\n- Identify proof, contradiction, and explanation patterns\n",
            RelationCategory.CREATION: "- Focus on production and development relationships\n- Identify invention, discovery, and building processes\n",
            RelationCategory.COMMUNICATION: "- Look for information exchange patterns\n- Identify notification, discussion, and response relationships\n"
        }
        return guidance.get(category, "")

    async def extract_enhanced_relations(
        self,
        text: str,
        entities: Optional[List[GraphEntity]] = None,
        chunk_size: int = 3000,
        source_doc_id: Optional[str] = None,
        domain_hint: Optional[str] = None
    ) -> List[GraphRelation]:
        """Extract enhanced relations with semantic depth and domain awareness."""
        if not text or not text.strip():
            return []

        entity_names = [entity.name for entity in entities] if entities else []
        entity_types = {entity.name: entity.type for entity in entities} if entities else {}

        self.logger.info(
            "Starting enhanced relation extraction",
            text_length=len(text),
            num_entities=len(entity_names),
            domain_hint=domain_hint,
            semantic_analysis=self.enable_semantic_analysis
        )

        try:
            # Step 1: Detect domain if not provided
            detected_domain = domain_hint or await self._detect_domain(text, entity_types)

            # Step 2: Extract relations with domain context
            if len(text) <= chunk_size:
                relations = await self._extract_enhanced_single_chunk(
                    text, entity_names, entity_types, detected_domain
                )
            else:
                relations = await self._extract_enhanced_chunked(
                    text, entity_names, entity_types, detected_domain, chunk_size
                )

            # Step 3: Post-process and enhance relations
            enhanced_relations = await self._post_process_relations(
                relations, entities, text, detected_domain
            )

            # Step 4: Convert to GraphRelation objects and filter by confidence
            graph_relations = []
            for relation in enhanced_relations:
                if relation.confidence >= self.min_confidence:
                    graph_relation = self._convert_to_enhanced_graph_relation(
                        relation, entities, source_doc_id, detected_domain
                    )
                    if graph_relation:
                        graph_relations.append(graph_relation)

            # Step 5: Deduplicate and rank relations
            graph_relations = self._deduplicate_and_rank_relations(graph_relations)

            self.logger.info(
                "Enhanced relation extraction completed",
                total_relations=len(graph_relations),
                domain=detected_domain,
                min_confidence=self.min_confidence
            )

            return graph_relations

        except Exception as e:
            self.logger.error(
                "Enhanced relation extraction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _detect_domain(
        self,
        text: str,
        entity_types: Dict[str, str]
    ) -> str:
        """Detect the domain of the text for specialized relation extraction."""
        if not self.enable_domain_detection:
            return DomainType.GENERAL.value

        # Simple domain detection based on keywords and entity types
        text_lower = text.lower()
        domain_scores = {}

        for domain, patterns in self.domain_patterns.items():
            score = 0
            # Check for domain indicators in text
            for indicator in patterns["indicators"]:
                score += text_lower.count(indicator.lower())

            # Check for domain-specific entity types
            for entity_type in entity_types.values():
                if domain == DomainType.MEDICAL and entity_type in ["SUBSTANCE", "PERSON", "ORGANIZATION"]:
                    score += 1
                elif domain == DomainType.TECHNICAL and entity_type in ["TECHNOLOGY", "SOFTWARE", "ALGORITHM"]:
                    score += 1
                elif domain == DomainType.BUSINESS and entity_type in ["ORGANIZATION", "PERSON", "LOCATION"]:
                    score += 1
                elif domain == DomainType.ACADEMIC and entity_type in ["CONCEPT", "METHODOLOGY", "DOCUMENT"]:
                    score += 1

            domain_scores[domain] = score

        # Return domain with highest score, or general if no clear winner
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain.value

        return DomainType.GENERAL.value

    async def _extract_enhanced_single_chunk(
        self,
        text: str,
        entity_names: List[str],
        entity_types: Dict[str, str],
        domain: str
    ) -> List[Relation]:
        """Extract relations from a single chunk with enhanced context."""
        # Build enhanced context
        entity_context = ""
        if entity_names:
            entity_list = []
            for name in entity_names:
                entity_type = entity_types.get(name, "UNKNOWN")
                entity_list.append(f"{name} ({entity_type})")
            entity_context = f"\n\nKnown entities with types: {', '.join(entity_list)}"

        domain_context = f"\n\nDetected domain: {domain.upper()}"
        if domain != DomainType.GENERAL.value:
            domain_indicators = self.domain_patterns.get(
                DomainType(domain), {}
            ).get("indicators", [])
            if domain_indicators:
                domain_context += f"\nDomain context indicators: {', '.join(domain_indicators[:5])}"
                domain_context += f"\nConsider domain-specific relationship patterns typical in {domain} contexts."

        analysis_context = ""
        if self.enable_causal_analysis:
            analysis_context += "\n\nFOCUS ON CAUSAL RELATIONSHIPS: Look for cause-effect patterns, dependencies, and influences."
        if self.enable_temporal_analysis:
            analysis_context += "\n\nFOCUS ON TEMPORAL RELATIONSHIPS: Identify sequences, timing, and chronological patterns."

        prompt = f"""Extract meaningful relationships between entities in the following text.

CONTEXT:{entity_context}{domain_context}{analysis_context}

TEXT:
{text}

INSTRUCTIONS:
1. Prioritize deep, meaningful relationships over surface mentions
2. Use domain-specific relation types when appropriate
3. Include causal and temporal relationships where evident
4. Provide confidence scores based on textual evidence
5. Include the specific text snippet supporting each relation"""

        result = await self.run(prompt)
        return result.relations

    async def _extract_enhanced_chunked(
        self,
        text: str,
        entity_names: List[str],
        entity_types: Dict[str, str],
        domain: str,
        chunk_size: int
    ) -> List[Relation]:
        """Extract relations from chunked text with enhanced processing."""
        # Split text into overlapping chunks to preserve context
        chunks = self._create_overlapping_chunks(text, chunk_size, overlap=200)
        all_relations = []

        # Process chunks with limited concurrency
        semaphore = asyncio.Semaphore(3)

        async def process_chunk(chunk_idx: int, chunk: str) -> List[Relation]:
            async with semaphore:
                try:
                    self.logger.debug(f"Processing enhanced chunk {chunk_idx + 1}/{len(chunks)}")
                    return await self._extract_enhanced_single_chunk(
                        chunk, entity_names, entity_types, domain
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to process enhanced chunk {chunk_idx + 1}",
                        error=str(e)
                    )
                    return []

        # Process all chunks
        chunk_results = await asyncio.gather(
            *[process_chunk(i, chunk) for i, chunk in enumerate(chunks)],
            return_exceptions=True
        )

        # Collect results
        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                self.logger.warning(f"Enhanced chunk {i + 1} failed: {result}")
            else:
                all_relations.extend(result)

        return all_relations

    def _create_overlapping_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 200
    ) -> List[str]:
        """Create overlapping chunks to preserve context across boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(end - 200, start)
                sentence_end = -1

                for i in range(end - 1, search_start - 1, -1):
                    if text[i] in '.!?':
                        sentence_end = i + 1
                        break

                if sentence_end > start:
                    end = sentence_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + chunk_size - overlap, end)

            if start >= len(text):
                break

        return chunks

    async def _post_process_relations(
        self,
        relations: List[Relation],
        entities: Optional[List[GraphEntity]],
        text: str,
        domain: str
    ) -> List[Relation]:
        """Post-process relations to enhance semantic understanding."""
        if not self.enable_semantic_analysis:
            return relations

        enhanced_relations = []

        for relation in relations:
            # Enhance relation with semantic analysis
            enhanced_relation = await self._enhance_relation_semantics(
                relation, entities, text, domain
            )
            enhanced_relations.append(enhanced_relation)

        return enhanced_relations

    async def _enhance_relation_semantics(
        self,
        relation: Relation,
        entities: Optional[List[GraphEntity]],
        text: str,
        domain: str
    ) -> Relation:
        """Enhance a single relation with semantic analysis."""
        # For now, return the relation as-is
        # This can be extended with more sophisticated semantic analysis
        return relation

    def _convert_to_enhanced_graph_relation(
        self,
        relation: Relation,
        entities: Optional[List[GraphEntity]],
        source_doc_id: Optional[str],
        domain: str
    ) -> Optional[GraphRelation]:
        """Convert AI relation to enhanced graph relation."""
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
            self.logger.debug(
                "Skipping relation due to unresolved entities",
                source_entity=relation.source_entity,
                target_entity=relation.target_entity,
                available_entities=list(entity_name_to_id.keys())
            )
            return None

        # Normalize relation type
        graph_type = self._normalize_relation_type(relation.relation_type)

        # Enhanced attributes
        attributes = {
            'description': relation.context or f"Relation: {relation.relation_type}",
            'evidence_text': getattr(relation, 'context', ''),
            'semantic_category': self._get_relation_category(graph_type),
            'domain': domain,
            'extraction_method': 'enhanced_semantic'
        }

        # Add entity names for fallback
        attributes['source_entity_name'] = relation.source_entity
        attributes['target_entity_name'] = relation.target_entity

        return GraphRelation(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            type=graph_type,
            confidence=relation.confidence,
            source_doc_id=source_doc_id,
            attributes=attributes
        )

    def _normalize_relation_type(self, relation_type: str) -> str:
        """Normalize relation type - now keeps AI-generated types as-is."""
        # Simply clean up the relation type without forcing it to match predefined schema
        normalized = relation_type.strip()

        # Convert to lowercase with underscores for consistency
        normalized = normalized.lower().replace(' ', '_').replace('-', '_')

        # Remove any special characters except underscores
        import re
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)

        # Ensure it starts with a letter
        if normalized and not normalized[0].isalpha():
            normalized = f"rel_{normalized}"

        # Ensure it's not empty
        if not normalized:
            normalized = "relates_to"

        return normalized

    def _get_relation_category(self, relation_type: str) -> str:
        """Get the semantic category for a relation type based on keywords."""
        relation_lower = relation_type.lower()

        # Analyze the relation type to determine its category
        if any(word in relation_lower for word in ['cause', 'prevent', 'enable', 'trigger', 'result', 'lead']):
            return RelationCategory.CAUSAL.value
        elif any(word in relation_lower for word in ['before', 'after', 'during', 'precede', 'follow', 'initiate']):
            return RelationCategory.TEMPORAL.value
        elif any(word in relation_lower for word in ['manage', 'own', 'belong', 'supervise', 'control']):
            return RelationCategory.HIERARCHICAL.value
        elif any(word in relation_lower for word in ['operate', 'transform', 'modify', 'implement', 'use']):
            return RelationCategory.FUNCTIONAL.value
        elif any(word in relation_lower for word in ['locate', 'connect', 'adjacent', 'surround']):
            return RelationCategory.SPATIAL.value
        elif any(word in relation_lower for word in ['similar', 'different', 'opposite', 'equivalent']):
            return RelationCategory.SEMANTIC.value
        elif any(word in relation_lower for word in ['collaborate', 'partner', 'compete', 'support']):
            return RelationCategory.COLLABORATIVE.value
        elif any(word in relation_lower for word in ['teach', 'learn', 'explain', 'prove', 'demonstrate']):
            return RelationCategory.KNOWLEDGE.value
        elif any(word in relation_lower for word in ['create', 'produce', 'generate', 'build', 'develop']):
            return RelationCategory.CREATION.value
        elif any(word in relation_lower for word in ['communicate', 'inform', 'discuss', 'notify']):
            return RelationCategory.COMMUNICATION.value
        else:
            return RelationCategory.SEMANTIC.value

    def _deduplicate_and_rank_relations(
        self,
        relations: List[GraphRelation]
    ) -> List[GraphRelation]:
        """Deduplicate and rank relations by confidence and semantic value."""
        # Create a mapping to track duplicates
        relation_map = {}

        for relation in relations:
            # Create a key for deduplication
            key = (
                relation.source_entity_id,
                relation.target_entity_id,
                relation.type
            )

            # Keep the relation with higher confidence
            if key not in relation_map or relation.confidence > relation_map[key].confidence:
                relation_map[key] = relation

        # Convert back to list and sort by confidence and semantic value
        unique_relations = list(relation_map.values())

        # Sort by semantic value (non-generic relations first) and confidence
        def relation_score(rel):
            # Prefer specific relations over generic ones
            generic_penalty = 0.1 if rel.type in ['MENTIONS', 'RELATED_TO'] else 0
            semantic_bonus = 0.1 if rel.attributes.get('semantic_category') in [
                'causal', 'functional', 'hierarchical'
            ] else 0
            return rel.confidence + semantic_bonus - generic_penalty

        unique_relations.sort(key=relation_score, reverse=True)

        return unique_relations
