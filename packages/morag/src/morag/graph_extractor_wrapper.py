"""
Graph extractor wrapper for MoRAG ingestion system.

This module provides a unified interface for extracting entities and relations
from text content using the morag-graph package.
"""

import os
from typing import Dict, List, Any, Optional
from morag_graph.extraction.entity_extractor import EntityExtractor
from morag_graph.extraction.relation_extractor import RelationExtractor
from morag_graph.extraction.base import LLMConfig

import structlog

logger = structlog.get_logger(__name__)


class GraphExtractor:
    """Unified graph extractor that combines entity and relation extraction."""
    
    def __init__(self):
        """Initialize the graph extractor."""
        self.entity_extractor = None
        self.relation_extractor = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the extractors with LLM configuration."""
        if self._initialized:
            return
            
        # Get LLM configuration from environment
        llm_config = LLMConfig(
            provider="gemini",
            api_key=os.getenv('GEMINI_API_KEY'),
            model=os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'),
            temperature=0.1,
            max_tokens=2000
        )
        
        self.entity_extractor = EntityExtractor(llm_config)
        self.relation_extractor = RelationExtractor(llm_config)
        self._initialized = True
        
        logger.info("Graph extractor initialized")
        
    async def extract_entities_and_relations(
        self,
        content: str,
        source_path: str
    ) -> Dict[str, Any]:
        """
        Extract entities and relations from text content.
        
        Args:
            content: Text content to extract from
            source_path: Source file path or identifier
            
        Returns:
            Dictionary containing entities, relations, and metadata
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Extract entities
            logger.info("Extracting entities from content", 
                       content_length=len(content),
                       source_path=source_path)
            
            entities = await self.entity_extractor.extract(
                text=content,
                source_doc_id=source_path
            )
            
            logger.info("Entities extracted", count=len(entities))
            
            # Extract relations
            logger.info("Extracting relations from content")
            
            relations = await self.relation_extractor.extract_with_entities(
                text=content,
                entities=entities,
                source_doc_id=source_path
            )
            
            logger.info("Relations extracted", count=len(relations))
            
            # Convert to serializable format
            entities_data = []
            for entity in entities:
                entity_data = {
                    'id': entity.id,
                    'name': entity.name,
                    'type': entity.type.value if hasattr(entity.type, 'value') else str(entity.type),
                    'description': entity.attributes.get('context', ''),
                    'attributes': entity.attributes or {},
                    'confidence': entity.confidence,
                    'source_doc_id': entity.source_doc_id
                }
                entities_data.append(entity_data)
                
            relations_data = []
            for relation in relations:
                relation_data = {
                    'id': relation.id,
                    'source_entity_id': relation.source_entity_id,
                    'target_entity_id': relation.target_entity_id,
                    'relation_type': relation.type.value if hasattr(relation.type, 'value') else str(relation.type),
                    'description': relation.attributes.get('context', ''),
                    'attributes': relation.attributes or {},
                    'confidence': relation.confidence,
                    'source_doc_id': relation.source_doc_id
                }
                relations_data.append(relation_data)
                
            return {
                'entities': entities_data,
                'relations': relations_data,
                'metadata': {
                    'entity_count': len(entities),
                    'relation_count': len(relations),
                    'source_path': source_path,
                    'content_length': len(content)
                }
            }
            
        except Exception as e:
            logger.error("Failed to extract graph data", 
                        error=str(e),
                        source_path=source_path)
            return {
                'entities': [],
                'relations': [],
                'metadata': {
                    'error': str(e),
                    'entity_count': 0,
                    'relation_count': 0,
                    'source_path': source_path,
                    'content_length': len(content)
                }
            }
