"""Entity and Relation adapters for MoRAG-Graphiti integration."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from graphiti_core.nodes import EntityType, RelationType
    GRAPHITI_AVAILABLE = True
except ImportError:
    # Graceful degradation when graphiti-core is not installed
    class EntityType:
        person = "person"
        organization = "organization"
        location = "location"
        concept = "concept"
        event = "event"
        other = "other"
    
    class RelationType:
        related_to = "related_to"
        part_of = "part_of"
        located_in = "located_in"
        works_for = "works_for"
        other = "other"
    
    GRAPHITI_AVAILABLE = False

from morag_graph.models import Entity, Relation
from .core import BaseAdapter, ConversionResult, ConversionDirection, ValidationError

logger = logging.getLogger(__name__)


class EntityAdapter(BaseAdapter[Entity, Dict[str, Any]]):
    """Adapter for converting Entities between MoRAG and Graphiti formats."""
    
    def __init__(self, strict_validation: bool = True):
        super().__init__(strict_validation)
        
        # Mapping between MoRAG entity types and Graphiti entity types
        self.entity_type_mapping = {
            "PERSON": EntityType.person,
            "ORGANIZATION": EntityType.organization,
            "LOCATION": EntityType.location,
            "CONCEPT": EntityType.concept,
            "EVENT": EntityType.event,
            "OTHER": EntityType.other,
            # Add more mappings as needed
        }
        
        # Reverse mapping for conversion back
        self.reverse_entity_type_mapping = {
            v: k for k, v in self.entity_type_mapping.items()
        }
    
    def to_graphiti(self, morag_model: Entity) -> ConversionResult:
        """Convert MoRAG Entity to Graphiti entity format.
        
        Args:
            morag_model: MoRAG Entity instance
            
        Returns:
            ConversionResult with Graphiti entity data
        """
        warnings = []
        
        try:
            # Validate input
            validation_errors = self.validate_input(morag_model, ConversionDirection.MORAG_TO_GRAPHITI)
            if validation_errors and self.strict_validation:
                self._record_conversion(False)
                return ConversionResult(
                    success=False,
                    error=f"Validation failed: {'; '.join(validation_errors)}"
                )
            elif validation_errors:
                warnings.extend(validation_errors)
            
            # Map entity type
            graphiti_entity_type = self.entity_type_mapping.get(
                morag_model.type.value if hasattr(morag_model.type, 'value') else str(morag_model.type),
                EntityType.other
            )
            
            if graphiti_entity_type == EntityType.other and morag_model.type:
                warnings.append(f"Unknown entity type '{morag_model.type}', mapped to 'other'")
            
            # Create Graphiti entity data
            entity_data = {
                "name": morag_model.name,
                "entity_type": graphiti_entity_type,
                "description": morag_model.description or f"Entity: {morag_model.name}",
                "properties": self._convert_entity_properties(morag_model),
                "metadata": {
                    "morag_entity_id": morag_model.id,
                    "original_type": str(morag_model.type),
                    "confidence": getattr(morag_model, 'confidence', None),
                    "source_document_id": getattr(morag_model, 'source_document_id', None),
                    "extraction_method": getattr(morag_model, 'extraction_method', None),
                    "conversion_timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Add aliases if present
            if hasattr(morag_model, 'aliases') and morag_model.aliases:
                entity_data["aliases"] = morag_model.aliases
            
            # Add coordinates if it's a location
            if hasattr(morag_model, 'coordinates') and morag_model.coordinates:
                entity_data["coordinates"] = morag_model.coordinates
            
            self._record_conversion(True, len(warnings))
            return ConversionResult(
                success=True,
                data=entity_data,
                warnings=warnings,
                metadata={
                    "original_entity_id": morag_model.id,
                    "conversion_timestamp": datetime.utcnow().isoformat(),
                    "entity_type_mapped": str(graphiti_entity_type)
                }
            )
            
        except Exception as e:
            self._record_conversion(False)
            logger.error(f"Entity conversion failed: {str(e)}")
            return ConversionResult(
                success=False,
                error=f"Conversion error: {str(e)}"
            )
    
    def from_graphiti(self, graphiti_data: Dict[str, Any]) -> ConversionResult:
        """Convert Graphiti entity data to MoRAG Entity.
        
        Args:
            graphiti_data: Graphiti entity data
            
        Returns:
            ConversionResult with MoRAG Entity
        """
        warnings = []
        
        try:
            # Validate input
            validation_errors = self.validate_input(graphiti_data, ConversionDirection.GRAPHITI_TO_MORAG)
            if validation_errors and self.strict_validation:
                self._record_conversion(False)
                return ConversionResult(
                    success=False,
                    error=f"Validation failed: {'; '.join(validation_errors)}"
                )
            elif validation_errors:
                warnings.extend(validation_errors)
            
            # Extract metadata
            metadata = graphiti_data.get("metadata", {})
            
            # Map entity type back to MoRAG format
            graphiti_entity_type = graphiti_data.get("entity_type", EntityType.other)
            morag_entity_type = self.reverse_entity_type_mapping.get(
                graphiti_entity_type,
                metadata.get("original_type", "OTHER")
            )
            
            # Create Entity data
            entity_data = {
                "id": metadata.get("morag_entity_id", graphiti_data.get("name", "unknown")),
                "name": graphiti_data.get("name", ""),
                "type": morag_entity_type,
                "description": graphiti_data.get("description"),
                "confidence": metadata.get("confidence"),
                "source_document_id": metadata.get("source_document_id"),
                "extraction_method": metadata.get("extraction_method")
            }
            
            # Add properties if present
            if "properties" in graphiti_data:
                entity_data.update(self._extract_entity_properties(graphiti_data["properties"]))
            
            # Add aliases if present
            if "aliases" in graphiti_data:
                entity_data["aliases"] = graphiti_data["aliases"]
            
            # Add coordinates if present
            if "coordinates" in graphiti_data:
                entity_data["coordinates"] = graphiti_data["coordinates"]
            
            # Create Entity instance (placeholder - would need proper constructor)
            entity = Entity(**entity_data)
            
            self._record_conversion(True, len(warnings))
            return ConversionResult(
                success=True,
                data=entity,
                warnings=warnings,
                metadata={
                    "conversion_timestamp": datetime.utcnow().isoformat(),
                    "source_entity_name": graphiti_data.get("name"),
                    "entity_type_mapped": morag_entity_type
                }
            )
            
        except Exception as e:
            self._record_conversion(False)
            logger.error(f"Graphiti to Entity conversion failed: {str(e)}")
            return ConversionResult(
                success=False,
                error=f"Conversion error: {str(e)}"
            )
    
    def validate_input(self, data: Any, direction: ConversionDirection) -> List[str]:
        """Validate input data for entity conversion."""
        errors = super().validate_input(data, direction)
        
        if direction == ConversionDirection.MORAG_TO_GRAPHITI:
            if not isinstance(data, Entity):
                errors.append("Input must be an Entity instance")
            elif not data.name:
                errors.append("Entity must have a name")
        
        elif direction == ConversionDirection.GRAPHITI_TO_MORAG:
            if not isinstance(data, dict):
                errors.append("Input must be a dictionary")
            elif "name" not in data:
                errors.append("Entity data must contain 'name' field")
        
        return errors
    
    def _convert_entity_properties(self, entity: Entity) -> Dict[str, Any]:
        """Convert MoRAG entity properties to Graphiti format."""
        properties = {}
        
        # Add standard properties
        if hasattr(entity, 'properties') and entity.properties:
            properties.update(entity.properties)
        
        # Add specific attributes as properties
        for attr in ['confidence', 'extraction_method', 'source_document_id']:
            if hasattr(entity, attr):
                value = getattr(entity, attr)
                if value is not None:
                    properties[attr] = value
        
        return properties
    
    def _extract_entity_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entity properties from Graphiti format."""
        # Filter out metadata that shouldn't be in the main entity
        filtered_properties = {}
        
        for key, value in properties.items():
            if key not in ['conversion_timestamp', 'morag_entity_id']:
                filtered_properties[key] = value
        
        return {"properties": filtered_properties} if filtered_properties else {}


class RelationAdapter(BaseAdapter[Relation, Dict[str, Any]]):
    """Adapter for converting Relations between MoRAG and Graphiti formats."""
    
    def __init__(self, strict_validation: bool = True):
        super().__init__(strict_validation)
        
        # Mapping between MoRAG relation types and Graphiti relation types
        self.relation_type_mapping = {
            "RELATED_TO": RelationType.related_to,
            "PART_OF": RelationType.part_of,
            "LOCATED_IN": RelationType.located_in,
            "WORKS_FOR": RelationType.works_for,
            "OTHER": RelationType.other,
            # Add more mappings as needed
        }
        
        # Reverse mapping for conversion back
        self.reverse_relation_type_mapping = {
            v: k for k, v in self.relation_type_mapping.items()
        }
    
    def to_graphiti(self, morag_model: Relation) -> ConversionResult:
        """Convert MoRAG Relation to Graphiti relation format."""
        warnings = []
        
        try:
            # Validate input
            validation_errors = self.validate_input(morag_model, ConversionDirection.MORAG_TO_GRAPHITI)
            if validation_errors and self.strict_validation:
                self._record_conversion(False)
                return ConversionResult(
                    success=False,
                    error=f"Validation failed: {'; '.join(validation_errors)}"
                )
            elif validation_errors:
                warnings.extend(validation_errors)
            
            # Map relation type
            graphiti_relation_type = self.relation_type_mapping.get(
                morag_model.type.value if hasattr(morag_model.type, 'value') else str(morag_model.type),
                RelationType.other
            )
            
            if graphiti_relation_type == RelationType.other and morag_model.type:
                warnings.append(f"Unknown relation type '{morag_model.type}', mapped to 'other'")
            
            # Create Graphiti relation data
            relation_data = {
                "source_entity": morag_model.source_entity_id,
                "target_entity": morag_model.target_entity_id,
                "relation_type": graphiti_relation_type,
                "description": morag_model.description or f"Relation: {morag_model.type}",
                "properties": self._convert_relation_properties(morag_model),
                "metadata": {
                    "morag_relation_id": morag_model.id,
                    "original_type": str(morag_model.type),
                    "confidence": getattr(morag_model, 'confidence', None),
                    "source_document_id": getattr(morag_model, 'source_document_id', None),
                    "extraction_method": getattr(morag_model, 'extraction_method', None),
                    "conversion_timestamp": datetime.utcnow().isoformat()
                }
            }
            
            self._record_conversion(True, len(warnings))
            return ConversionResult(
                success=True,
                data=relation_data,
                warnings=warnings,
                metadata={
                    "original_relation_id": morag_model.id,
                    "conversion_timestamp": datetime.utcnow().isoformat(),
                    "relation_type_mapped": str(graphiti_relation_type)
                }
            )
            
        except Exception as e:
            self._record_conversion(False)
            logger.error(f"Relation conversion failed: {str(e)}")
            return ConversionResult(
                success=False,
                error=f"Conversion error: {str(e)}"
            )
    
    def from_graphiti(self, graphiti_data: Dict[str, Any]) -> ConversionResult:
        """Convert Graphiti relation data to MoRAG Relation."""
        warnings = []
        
        try:
            # Validate input
            validation_errors = self.validate_input(graphiti_data, ConversionDirection.GRAPHITI_TO_MORAG)
            if validation_errors and self.strict_validation:
                self._record_conversion(False)
                return ConversionResult(
                    success=False,
                    error=f"Validation failed: {'; '.join(validation_errors)}"
                )
            elif validation_errors:
                warnings.extend(validation_errors)
            
            # Extract metadata
            metadata = graphiti_data.get("metadata", {})
            
            # Map relation type back to MoRAG format
            graphiti_relation_type = graphiti_data.get("relation_type", RelationType.other)
            morag_relation_type = self.reverse_relation_type_mapping.get(
                graphiti_relation_type,
                metadata.get("original_type", "OTHER")
            )
            
            # Create Relation data
            relation_data = {
                "id": metadata.get("morag_relation_id", f"rel_{datetime.utcnow().timestamp()}"),
                "source_entity_id": graphiti_data.get("source_entity"),
                "target_entity_id": graphiti_data.get("target_entity"),
                "type": morag_relation_type,
                "description": graphiti_data.get("description"),
                "confidence": metadata.get("confidence"),
                "source_document_id": metadata.get("source_document_id"),
                "extraction_method": metadata.get("extraction_method")
            }
            
            # Add properties if present
            if "properties" in graphiti_data:
                relation_data.update(self._extract_relation_properties(graphiti_data["properties"]))
            
            # Create Relation instance (placeholder - would need proper constructor)
            relation = Relation(**relation_data)
            
            self._record_conversion(True, len(warnings))
            return ConversionResult(
                success=True,
                data=relation,
                warnings=warnings,
                metadata={
                    "conversion_timestamp": datetime.utcnow().isoformat(),
                    "relation_type_mapped": morag_relation_type
                }
            )
            
        except Exception as e:
            self._record_conversion(False)
            logger.error(f"Graphiti to Relation conversion failed: {str(e)}")
            return ConversionResult(
                success=False,
                error=f"Conversion error: {str(e)}"
            )
    
    def validate_input(self, data: Any, direction: ConversionDirection) -> List[str]:
        """Validate input data for relation conversion."""
        errors = super().validate_input(data, direction)
        
        if direction == ConversionDirection.MORAG_TO_GRAPHITI:
            if not isinstance(data, Relation):
                errors.append("Input must be a Relation instance")
            elif not data.source_entity_id or not data.target_entity_id:
                errors.append("Relation must have both source and target entity IDs")
        
        elif direction == ConversionDirection.GRAPHITI_TO_MORAG:
            if not isinstance(data, dict):
                errors.append("Input must be a dictionary")
            elif not data.get("source_entity") or not data.get("target_entity"):
                errors.append("Relation data must contain both 'source_entity' and 'target_entity' fields")
        
        return errors
    
    def _convert_relation_properties(self, relation: Relation) -> Dict[str, Any]:
        """Convert MoRAG relation properties to Graphiti format."""
        properties = {}
        
        # Add standard properties
        if hasattr(relation, 'properties') and relation.properties:
            properties.update(relation.properties)
        
        # Add specific attributes as properties
        for attr in ['confidence', 'extraction_method', 'source_document_id']:
            if hasattr(relation, attr):
                value = getattr(relation, attr)
                if value is not None:
                    properties[attr] = value
        
        return properties
    
    def _extract_relation_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relation properties from Graphiti format."""
        # Filter out metadata that shouldn't be in the main relation
        filtered_properties = {}
        
        for key, value in properties.items():
            if key not in ['conversion_timestamp', 'morag_relation_id']:
                filtered_properties[key] = value
        
        return {"properties": filtered_properties} if filtered_properties else {}
