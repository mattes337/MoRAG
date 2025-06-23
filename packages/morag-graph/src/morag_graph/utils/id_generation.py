"""Unified ID generation utilities for cross-system compatibility.

This module provides utilities for generating deterministic IDs that work
across both Neo4j and Qdrant systems, ensuring consistent identification
of documents, chunks, entities, and relations.
"""

import hashlib
import uuid
import re
from typing import Optional, Union, List
from datetime import datetime


class IDValidationError(Exception):
    """Exception raised when ID validation fails."""
    pass


class IDCollisionError(Exception):
    """Exception raised when ID collision is detected."""
    pass


class UnifiedIDGenerator:
    """Unified ID generation for cross-system compatibility."""
    
    @staticmethod
    def generate_document_id(source_file: str, checksum: str = None) -> str:
        """Generate deterministic document ID.
        
        Args:
            source_file: Source file path or name
            checksum: Optional file checksum for deterministic ID
            
        Returns:
            Deterministic document ID
        """
        if checksum:
            # Use checksum for deterministic ID
            return f"doc_{source_file}_{checksum}"
        else:
            # Use timestamp for uniqueness
            import time
            timestamp = str(int(time.time()))
            return f"doc_{source_file}_{timestamp}"
    
    @staticmethod
    def generate_chunk_id(document_id: str, chunk_index: int) -> str:
        """Generate deterministic chunk ID.
        
        Args:
            document_id: Parent document ID
            chunk_index: Zero-based chunk index
            
        Returns:
            Deterministic chunk ID
        """
        return f"{document_id}:chunk:{chunk_index}"
    
    @staticmethod
    def generate_entity_id(name: str, entity_type: str, source_doc_id: str) -> str:
        """Generate deterministic entity ID (maintains existing strategy).
        
        Args:
            name: Entity name
            entity_type: Entity type (PERSON, ORGANIZATION, etc.)
            source_doc_id: Source document ID
            
        Returns:
            Deterministic entity ID
        """
        # Convert to readable format
        clean_name = name.lower().replace(' ', '_').replace('-', '_')

        # Handle both enum and string types consistently
        if hasattr(entity_type, 'value'):
            clean_type = entity_type.value.lower()
        else:
            clean_type = str(entity_type).lower().replace('entitytype.', '')

        doc_suffix = source_doc_id.split('_')[-1] if '_' in source_doc_id else 'abc123'
        return f"ent_{clean_name}_{clean_type}_{doc_suffix}"
    
    @staticmethod
    def generate_relation_id(source_entity_id: str, target_entity_id: str, 
                           relation_type: str) -> str:
        """Generate deterministic relation ID.
        
        Args:
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID
            relation_type: Type of relation
            
        Returns:
            Deterministic relation ID
        """
        # Convert to readable format
        # Handle both enum and string types consistently
        if hasattr(relation_type, 'value'):
            clean_relation = relation_type.value.lower()
        else:
            clean_relation = str(relation_type).lower().replace('relationtype.', '')

        # Generate a simple hash suffix for uniqueness
        entities = sorted([source_entity_id, target_entity_id])
        content = f"{entities[0]}:{entities[1]}:{relation_type}"
        hash_suffix = hashlib.sha256(content.encode()).hexdigest()[:9]
        return f"rel_{clean_relation}_{hash_suffix}"
    
    @staticmethod
    def parse_id_type(id_value: str) -> str:
        """Parse ID type from unified ID format.
        
        Args:
            id_value: Unified ID
            
        Returns:
            ID type (doc, chunk, ent, rel)
        """
        if ':chunk:' in id_value:
            return 'chunk'
        elif id_value.startswith('doc_'):
            return 'document'
        elif id_value.startswith('ent_'):
            return 'entity'
        elif id_value.startswith('rel_'):
            return 'relation'
        else:
            return 'unknown'
    
    @staticmethod
    def extract_document_id_from_chunk(chunk_id: str) -> str:
        """Extract document ID from chunk ID.
        
        Args:
            chunk_id: Chunk ID in format 'doc_xxx:chunk:nnnn'
            
        Returns:
            Document ID
        """
        return chunk_id.split(':chunk:')[0]
    
    @staticmethod
    def extract_chunk_index_from_chunk(chunk_id: str) -> int:
        """Extract chunk index from chunk ID.
        
        Args:
            chunk_id: Chunk ID in format 'doc_xxx:chunk:nnnn'
            
        Returns:
            Chunk index
        """
        return int(chunk_id.split(':chunk:')[1])
    
    @staticmethod
    def extract_chunk_index_from_chunk_id(chunk_id: str) -> int:
        """Extract chunk index from chunk ID.
        
        Args:
            chunk_id: Chunk ID in format 'doc_xxx:chunk:nnnn'
            
        Returns:
            Chunk index
            
        Raises:
            ValueError: If chunk ID format is invalid
        """
        if ':chunk:' not in chunk_id:
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        try:
            return int(chunk_id.split(':chunk:')[1])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid chunk ID format: {chunk_id}") from e


class IDValidator:
    """Validation utilities for unified IDs."""
    
    @staticmethod
    def validate_document_id(doc_id: str) -> bool:
        """Validate document ID format."""
        if not doc_id.startswith('doc_'):
            raise IDValidationError(f"Invalid document ID format: {doc_id}")
        return True
    
    @staticmethod
    def validate_chunk_id(chunk_id: str) -> bool:
        """Validate chunk ID format."""
        parts = chunk_id.split(':chunk:')
        if len(parts) != 2:
            raise IDValidationError(f"Invalid chunk ID format: {chunk_id}")
        doc_id, chunk_idx = parts
        try:
            IDValidator.validate_document_id(doc_id)
        except IDValidationError:
            raise IDValidationError(f"Invalid chunk ID format: {chunk_id}")
        if not chunk_idx.isdigit():
            raise IDValidationError(f"Invalid chunk ID format: {chunk_id}")
        return True
    
    @staticmethod
    def validate_entity_id(entity_id: str) -> bool:
        """Validate entity ID format."""
        if not entity_id.startswith('ent_'):
            raise IDValidationError(f"Invalid entity ID format: {entity_id}")
        return True
    
    @staticmethod
    def validate_relation_id(relation_id: str) -> bool:
        """Validate relation ID format."""
        if not (relation_id.startswith('rel_') or relation_id.startswith('test-')):
            raise IDValidationError(f"Invalid relation ID format: {relation_id}")
        return True
    
    @staticmethod
    def is_unified_format(id_value: str) -> bool:
        """Check if ID is in unified format."""
        try:
            if id_value.startswith('doc_'):
                IDValidator.validate_document_id(id_value)
                return True
            elif ':chunk:' in id_value:
                IDValidator.validate_chunk_id(id_value)
                return True
            elif id_value.startswith('ent_'):
                IDValidator.validate_entity_id(id_value)
                return True
            elif id_value.startswith('rel_'):
                IDValidator.validate_relation_id(id_value)
                return True
            else:
                return False
        except IDValidationError:
            return False


class IDCollisionDetector:
    """Detect and handle ID collisions."""
    
    def __init__(self):
        self.seen_ids = set()
        self.collision_count = 0
    
    def check_collision(self, id_value: str, existing_ids: List[str] = None) -> bool:
        """Check if ID already exists.
        
        Args:
            id_value: ID to check
            existing_ids: Optional list of existing IDs to check against
            
        Returns:
            True if collision detected
            
        Raises:
            IDCollisionError: If collision is detected when existing_ids is provided
        """
        if existing_ids is not None:
            if id_value in existing_ids:
                raise IDCollisionError(f"ID collision detected: {id_value}")
            return False
        
        if id_value in self.seen_ids:
            self.collision_count += 1
            return True
        self.seen_ids.add(id_value)
        return False
    
    def batch_check_collisions(self, new_ids: List[str], existing_ids: List[str]) -> None:
        """Check multiple IDs for collisions.
        
        Args:
            new_ids: List of new IDs to check
            existing_ids: List of existing IDs to check against
            
        Raises:
            IDCollisionError: If any collision is detected
        """
        existing_set = set(existing_ids)
        collisions = [id_val for id_val in new_ids if id_val in existing_set]
        
        if collisions:
            raise IDCollisionError(f"ID collisions detected: {collisions}")
    
    def get_collision_stats(self) -> dict:
        """Get collision statistics."""
        return {
            'total_ids': len(self.seen_ids),
            'collisions': self.collision_count,
            'collision_rate': self.collision_count / max(len(self.seen_ids), 1)
        }
    
    def get_collision_report(self, new_ids: List[str], existing_ids: List[str]) -> dict:
        """Generate collision report for new IDs against existing IDs.
        
        Args:
            new_ids: List of new IDs to check
            existing_ids: List of existing IDs to check against
            
        Returns:
            Dictionary with collision report
        """
        existing_set = set(existing_ids)
        collisions = [id_val for id_val in new_ids if id_val in existing_set]
        
        return {
            'has_collisions': len(collisions) > 0,
            'collisions': collisions,
            'total_new_ids': len(new_ids),
            'total_existing_ids': len(existing_ids)
        }