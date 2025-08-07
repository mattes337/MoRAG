"""Migration utilities for converting legacy facts to hybrid format."""

import logging
from typing import List, Dict, Any, Optional
from ..models.fact import Fact, StructuredMetadata


class HybridFactMigrator:
    """Utility for migrating legacy facts to hybrid format."""
    
    def __init__(self):
        """Initialize the migrator."""
        self.logger = logging.getLogger(__name__)
    
    def migrate_legacy_fact(self, legacy_fact_data: Dict[str, Any]) -> Fact:
        """Migrate a legacy fact to hybrid format.
        
        Args:
            legacy_fact_data: Dictionary containing legacy fact data
            
        Returns:
            Migrated Fact object in hybrid format
        """
        # Extract legacy structured fields
        subject = legacy_fact_data.get('subject', '')
        obj = legacy_fact_data.get('object', '')
        approach = legacy_fact_data.get('approach', '')
        solution = legacy_fact_data.get('solution', '')
        condition = legacy_fact_data.get('condition', '')
        remarks = legacy_fact_data.get('remarks', '')
        
        # Construct fact_text from legacy fields
        fact_text = self._construct_fact_text(subject, obj, approach, solution, condition, remarks)
        
        # Create structured metadata with legacy fields preserved
        structured_metadata = StructuredMetadata(
            primary_entities=self._extract_primary_entities(subject, obj),
            relationships=self._extract_relationships(approach, solution),
            domain_concepts=self._extract_domain_concepts(approach, solution, remarks),
            # Preserve legacy fields for backward compatibility
            subject=subject if subject else None,
            object=obj if obj else None,
            approach=approach if approach else None,
            solution=solution if solution else None
        )
        
        # Create hybrid fact
        fact_data = {
            'fact_text': fact_text,
            'structured_metadata': structured_metadata,
            'source_chunk_id': legacy_fact_data.get('source_chunk_id', ''),
            'source_document_id': legacy_fact_data.get('source_document_id', ''),
            'extraction_confidence': legacy_fact_data.get('extraction_confidence', 0.8),
            'fact_type': legacy_fact_data.get('fact_type', 'definition'),
            'domain': legacy_fact_data.get('domain'),
            'keywords': legacy_fact_data.get('keywords', []),
            'language': legacy_fact_data.get('language', 'en'),
            # Preserve all other metadata
            'query_relevance': legacy_fact_data.get('query_relevance'),
            'evidence_strength': legacy_fact_data.get('evidence_strength'),
            'source_span': legacy_fact_data.get('source_span'),
            'source_file_path': legacy_fact_data.get('source_file_path'),
            'source_file_name': legacy_fact_data.get('source_file_name'),
            'page_number': legacy_fact_data.get('page_number'),
            'chapter_title': legacy_fact_data.get('chapter_title'),
            'chapter_index': legacy_fact_data.get('chapter_index'),
            'paragraph_index': legacy_fact_data.get('paragraph_index'),
            'timestamp_start': legacy_fact_data.get('timestamp_start'),
            'timestamp_end': legacy_fact_data.get('timestamp_end'),
            'topic_header': legacy_fact_data.get('topic_header'),
            'speaker_label': legacy_fact_data.get('speaker_label'),
            'source_text_excerpt': legacy_fact_data.get('source_text_excerpt')
        }
        
        # Remove None values
        fact_data = {k: v for k, v in fact_data.items() if v is not None}
        
        return Fact(**fact_data)
    
    def _construct_fact_text(
        self, 
        subject: str, 
        obj: str, 
        approach: str, 
        solution: str, 
        condition: str, 
        remarks: str
    ) -> str:
        """Construct self-contained fact text from legacy fields.
        
        Args:
            subject: Subject entity
            obj: Object entity
            approach: Approach/method
            solution: Solution/outcome
            condition: Condition/context
            remarks: Additional remarks
            
        Returns:
            Self-contained fact text
        """
        parts = []
        
        # Start with subject-object relationship
        if subject and obj:
            parts.append(f"{subject} relates to {obj}")
        elif subject:
            parts.append(f"Regarding {subject}")
        elif obj:
            parts.append(f"Concerning {obj}")
        
        # Add approach/method
        if approach:
            if parts:
                parts.append(f"through {approach}")
            else:
                parts.append(f"The approach involves {approach}")
        
        # Add solution/outcome
        if solution:
            if parts:
                parts.append(f"resulting in {solution}")
            else:
                parts.append(f"The outcome is {solution}")
        
        # Add condition/context
        if condition:
            if parts:
                parts.append(f"when {condition}")
            else:
                parts.append(f"This applies when {condition}")
        
        # Add remarks
        if remarks:
            if parts:
                parts.append(f"Note: {remarks}")
            else:
                parts.append(remarks)
        
        # Join parts into coherent sentence
        if parts:
            fact_text = " ".join(parts)
            # Ensure it ends with a period
            if not fact_text.endswith('.'):
                fact_text += '.'
            return fact_text
        else:
            return "No specific information available."
    
    def _extract_primary_entities(self, subject: str, obj: str) -> List[str]:
        """Extract primary entities from subject and object.
        
        Args:
            subject: Subject entity
            obj: Object entity
            
        Returns:
            List of primary entities
        """
        entities = []
        if subject and subject.strip():
            entities.append(subject.strip())
        if obj and obj.strip() and obj.strip() != subject.strip():
            entities.append(obj.strip())
        return entities
    
    def _extract_relationships(self, approach: str, solution: str) -> List[str]:
        """Extract relationships from approach and solution fields.
        
        Args:
            approach: Approach/method
            solution: Solution/outcome
            
        Returns:
            List of relationship terms
        """
        relationships = []
        
        # Common relationship indicators
        relationship_indicators = [
            'treats', 'cures', 'prevents', 'reduces', 'increases', 'improves',
            'causes', 'leads to', 'results in', 'affects', 'influences',
            'helps', 'supports', 'enhances', 'optimizes', 'manages'
        ]
        
        # Check approach and solution for relationship indicators
        for text in [approach, solution]:
            if text:
                text_lower = text.lower()
                for indicator in relationship_indicators:
                    if indicator in text_lower:
                        relationships.append(indicator)
        
        # Default relationships if none found
        if not relationships:
            if approach:
                relationships.append('involves')
            if solution:
                relationships.append('results in')
        
        return list(set(relationships))  # Remove duplicates
    
    def _extract_domain_concepts(self, approach: str, solution: str, remarks: str) -> List[str]:
        """Extract domain-specific concepts from text fields.
        
        Args:
            approach: Approach/method
            solution: Solution/outcome
            remarks: Additional remarks
            
        Returns:
            List of domain concepts
        """
        concepts = []
        
        # Common domain concept patterns
        concept_patterns = [
            'dosage', 'treatment', 'therapy', 'medication', 'extract', 'supplement',
            'technique', 'method', 'procedure', 'protocol', 'strategy', 'approach',
            'optimization', 'configuration', 'implementation', 'algorithm', 'system',
            'compliance', 'regulation', 'standard', 'guideline', 'requirement'
        ]
        
        # Extract concepts from all text fields
        all_text = ' '.join([text for text in [approach, solution, remarks] if text])
        if all_text:
            all_text_lower = all_text.lower()
            for pattern in concept_patterns:
                if pattern in all_text_lower:
                    concepts.append(pattern)
        
        return list(set(concepts))  # Remove duplicates
    
    def migrate_fact_batch(self, legacy_facts: List[Dict[str, Any]]) -> List[Fact]:
        """Migrate a batch of legacy facts to hybrid format.
        
        Args:
            legacy_facts: List of legacy fact dictionaries
            
        Returns:
            List of migrated Fact objects
        """
        migrated_facts = []
        
        for i, legacy_fact in enumerate(legacy_facts):
            try:
                migrated_fact = self.migrate_legacy_fact(legacy_fact)
                migrated_facts.append(migrated_fact)
                self.logger.debug(f"Successfully migrated fact {i+1}/{len(legacy_facts)}")
            except Exception as e:
                self.logger.error(f"Failed to migrate fact {i+1}: {e}")
                continue
        
        self.logger.info(f"Migrated {len(migrated_facts)}/{len(legacy_facts)} facts successfully")
        return migrated_facts
    
    def validate_migration(self, original_fact: Dict[str, Any], migrated_fact: Fact) -> bool:
        """Validate that migration preserved essential information.
        
        Args:
            original_fact: Original legacy fact data
            migrated_fact: Migrated hybrid fact
            
        Returns:
            True if migration is valid
        """
        # Check that essential fields are preserved
        if original_fact.get('subject') and original_fact['subject'] not in migrated_fact.fact_text:
            return False
        
        if original_fact.get('object') and original_fact['object'] not in migrated_fact.fact_text:
            return False
        
        # Check that metadata is preserved
        if (original_fact.get('subject') and 
            migrated_fact.structured_metadata.subject != original_fact['subject']):
            return False
        
        if (original_fact.get('object') and 
            migrated_fact.structured_metadata.object != original_fact['object']):
            return False
        
        # Check that fact_text is not empty
        if not migrated_fact.fact_text or migrated_fact.fact_text.strip() == "No specific information available.":
            return False
        
        return True


def migrate_legacy_facts_to_hybrid(legacy_facts: List[Dict[str, Any]]) -> List[Fact]:
    """Convenience function to migrate legacy facts to hybrid format.
    
    Args:
        legacy_facts: List of legacy fact dictionaries
        
    Returns:
        List of migrated hybrid facts
    """
    migrator = HybridFactMigrator()
    return migrator.migrate_fact_batch(legacy_facts)
