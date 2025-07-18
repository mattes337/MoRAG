# Step 11: Data Migration Strategy

**Duration**: 4-5 days  
**Phase**: Migration and Production  
**Prerequisites**: Steps 1-10 completed, hybrid search working

## Objective

Develop comprehensive migration strategies and tools for transferring existing Neo4j data to Graphiti, ensuring data integrity, minimal downtime, and rollback capabilities.

## Deliverables

1. Migration assessment and planning tools
2. Data extraction utilities from existing Neo4j
3. Graphiti data import and validation tools
4. Rollback and recovery procedures
5. Migration monitoring and progress tracking

## Implementation

### 1. Create Migration Assessment Tool

**File**: `packages/morag-graph/src/morag_graph/graphiti/migration/assessment.py`

```python
"""Migration assessment tool for analyzing existing Neo4j data."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MigrationAssessment:
    """Assessment results for migration planning."""
    total_documents: int
    total_chunks: int
    total_entities: int
    total_relations: int
    entity_type_distribution: Dict[str, int]
    relation_type_distribution: Dict[str, int]
    data_quality_issues: List[str]
    estimated_migration_time: float  # hours
    storage_requirements: Dict[str, Any]
    complexity_score: float  # 0-10 scale


class MigrationAssessmentTool:
    """Tool for assessing existing Neo4j data for migration to Graphiti."""
    
    def __init__(self, neo4j_storage):
        self.neo4j_storage = neo4j_storage
        
    async def assess_migration_feasibility(self) -> MigrationAssessment:
        """Assess the feasibility and complexity of migrating to Graphiti.
        
        Returns:
            MigrationAssessment with detailed analysis
        """
        logger.info("Starting migration assessment...")
        
        # Count existing data
        document_count = await self._count_documents()
        chunk_count = await self._count_chunks()
        entity_count = await self._count_entities()
        relation_count = await self._count_relations()
        
        # Analyze data distribution
        entity_distribution = await self._analyze_entity_types()
        relation_distribution = await self._analyze_relation_types()
        
        # Identify data quality issues
        quality_issues = await self._identify_quality_issues()
        
        # Estimate migration time and complexity
        migration_time = self._estimate_migration_time(
            document_count, chunk_count, entity_count, relation_count
        )
        complexity_score = self._calculate_complexity_score(
            entity_count, relation_count, len(quality_issues)
        )
        
        # Calculate storage requirements
        storage_reqs = self._estimate_storage_requirements(
            document_count, chunk_count, entity_count, relation_count
        )
        
        assessment = MigrationAssessment(
            total_documents=document_count,
            total_chunks=chunk_count,
            total_entities=entity_count,
            total_relations=relation_count,
            entity_type_distribution=entity_distribution,
            relation_type_distribution=relation_distribution,
            data_quality_issues=quality_issues,
            estimated_migration_time=migration_time,
            storage_requirements=storage_reqs,
            complexity_score=complexity_score
        )
        
        logger.info(f"Migration assessment completed. Complexity score: {complexity_score}/10")
        return assessment
    
    async def _count_documents(self) -> int:
        """Count total documents in Neo4j."""
        try:
            # This would use the actual Neo4j storage methods
            # For now, we'll simulate the count
            query = "MATCH (d:Document) RETURN count(d) as count"
            result = await self.neo4j_storage._execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
    
    async def _count_chunks(self) -> int:
        """Count total chunks in Neo4j."""
        try:
            query = "MATCH (c:Chunk) RETURN count(c) as count"
            result = await self.neo4j_storage._execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to count chunks: {e}")
            return 0
    
    async def _count_entities(self) -> int:
        """Count total entities in Neo4j."""
        try:
            query = "MATCH (e:Entity) RETURN count(e) as count"
            result = await self.neo4j_storage._execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to count entities: {e}")
            return 0
    
    async def _count_relations(self) -> int:
        """Count total relations in Neo4j."""
        try:
            query = "MATCH ()-[r:RELATION]->() RETURN count(r) as count"
            result = await self.neo4j_storage._execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to count relations: {e}")
            return 0
    
    async def _analyze_entity_types(self) -> Dict[str, int]:
        """Analyze distribution of entity types."""
        try:
            query = """
            MATCH (e:Entity) 
            RETURN e.type as entity_type, count(e) as count 
            ORDER BY count DESC
            """
            results = await self.neo4j_storage._execute_query(query)
            return {row['entity_type']: row['count'] for row in results}
        except Exception as e:
            logger.error(f"Failed to analyze entity types: {e}")
            return {}
    
    async def _analyze_relation_types(self) -> Dict[str, int]:
        """Analyze distribution of relation types."""
        try:
            query = """
            MATCH ()-[r:RELATION]->() 
            RETURN r.relation_type as relation_type, count(r) as count 
            ORDER BY count DESC
            """
            results = await self.neo4j_storage._execute_query(query)
            return {row['relation_type']: row['count'] for row in results}
        except Exception as e:
            logger.error(f"Failed to analyze relation types: {e}")
            return {}
    
    async def _identify_quality_issues(self) -> List[str]:
        """Identify potential data quality issues."""
        issues = []
        
        try:
            # Check for entities without names
            query = "MATCH (e:Entity) WHERE e.name IS NULL OR e.name = '' RETURN count(e) as count"
            result = await self.neo4j_storage._execute_query(query)
            if result and result[0]['count'] > 0:
                issues.append(f"Found {result[0]['count']} entities without names")
            
            # Check for relations without types
            query = "MATCH ()-[r:RELATION]->() WHERE r.relation_type IS NULL RETURN count(r) as count"
            result = await self.neo4j_storage._execute_query(query)
            if result and result[0]['count'] > 0:
                issues.append(f"Found {result[0]['count']} relations without types")
            
            # Check for orphaned chunks
            query = """
            MATCH (c:Chunk) 
            WHERE NOT (c)-[:BELONGS_TO]->(:Document) 
            RETURN count(c) as count
            """
            result = await self.neo4j_storage._execute_query(query)
            if result and result[0]['count'] > 0:
                issues.append(f"Found {result[0]['count']} orphaned chunks")
            
            # Check for duplicate entities
            query = """
            MATCH (e1:Entity), (e2:Entity) 
            WHERE e1.name = e2.name AND e1.type = e2.type AND id(e1) < id(e2)
            RETURN count(*) as count
            """
            result = await self.neo4j_storage._execute_query(query)
            if result and result[0]['count'] > 0:
                issues.append(f"Found {result[0]['count']} potential duplicate entities")
                
        except Exception as e:
            logger.error(f"Failed to identify quality issues: {e}")
            issues.append(f"Quality assessment failed: {str(e)}")
        
        return issues
    
    def _estimate_migration_time(
        self, 
        documents: int, 
        chunks: int, 
        entities: int, 
        relations: int
    ) -> float:
        """Estimate migration time in hours."""
        # Base processing rates (items per hour)
        doc_rate = 1000  # documents per hour
        chunk_rate = 5000  # chunks per hour
        entity_rate = 10000  # entities per hour
        relation_rate = 8000  # relations per hour
        
        # Calculate time for each component
        doc_time = documents / doc_rate
        chunk_time = chunks / chunk_rate
        entity_time = entities / entity_rate
        relation_time = relations / relation_rate
        
        # Add overhead for validation and error handling (20%)
        total_time = (doc_time + chunk_time + entity_time + relation_time) * 1.2
        
        return max(1.0, total_time)  # Minimum 1 hour
    
    def _calculate_complexity_score(
        self, 
        entities: int, 
        relations: int, 
        quality_issues: int
    ) -> float:
        """Calculate migration complexity score (0-10)."""
        # Base complexity from data volume
        volume_score = min(5.0, (entities + relations) / 100000 * 5)
        
        # Quality issues add complexity
        quality_score = min(3.0, quality_issues * 0.5)
        
        # Relationship density adds complexity
        if entities > 0:
            density = relations / entities
            density_score = min(2.0, density / 10 * 2)
        else:
            density_score = 0.0
        
        return min(10.0, volume_score + quality_score + density_score)
    
    def _estimate_storage_requirements(
        self, 
        documents: int, 
        chunks: int, 
        entities: int, 
        relations: int
    ) -> Dict[str, Any]:
        """Estimate storage requirements for Graphiti."""
        # Average sizes in bytes
        avg_document_size = 50000  # 50KB per document episode
        avg_chunk_size = 2000     # 2KB per chunk episode
        avg_entity_size = 1000    # 1KB per entity episode
        avg_relation_size = 500   # 500B per relation episode
        
        # Calculate total storage
        document_storage = documents * avg_document_size
        chunk_storage = chunks * avg_chunk_size
        entity_storage = entities * avg_entity_size
        relation_storage = relations * avg_relation_size
        
        total_storage = document_storage + chunk_storage + entity_storage + relation_storage
        
        # Add overhead for indexes and metadata (30%)
        total_with_overhead = total_storage * 1.3
        
        return {
            'documents_mb': document_storage / (1024 * 1024),
            'chunks_mb': chunk_storage / (1024 * 1024),
            'entities_mb': entity_storage / (1024 * 1024),
            'relations_mb': relation_storage / (1024 * 1024),
            'total_mb': total_with_overhead / (1024 * 1024),
            'total_gb': total_with_overhead / (1024 * 1024 * 1024),
            'recommended_disk_space_gb': (total_with_overhead * 2) / (1024 * 1024 * 1024)  # 2x for safety
        }
    
    async def generate_migration_report(self, assessment: MigrationAssessment) -> str:
        """Generate a detailed migration report."""
        report = f"""
# MoRAG to Graphiti Migration Assessment Report

**Generated**: {datetime.now().isoformat()}

## Data Overview
- **Documents**: {assessment.total_documents:,}
- **Chunks**: {assessment.total_chunks:,}
- **Entities**: {assessment.total_entities:,}
- **Relations**: {assessment.total_relations:,}

## Entity Type Distribution
"""
        for entity_type, count in assessment.entity_type_distribution.items():
            report += f"- **{entity_type}**: {count:,}\n"
        
        report += "\n## Relation Type Distribution\n"
        for relation_type, count in assessment.relation_type_distribution.items():
            report += f"- **{relation_type}**: {count:,}\n"
        
        report += f"""
## Migration Estimates
- **Estimated Time**: {assessment.estimated_migration_time:.1f} hours
- **Complexity Score**: {assessment.complexity_score:.1f}/10
- **Storage Required**: {assessment.storage_requirements['total_gb']:.2f} GB
- **Recommended Disk Space**: {assessment.storage_requirements['recommended_disk_space_gb']:.2f} GB

## Data Quality Issues
"""
        if assessment.data_quality_issues:
            for issue in assessment.data_quality_issues:
                report += f"- ⚠️ {issue}\n"
        else:
            report += "- ✅ No significant data quality issues detected\n"
        
        report += f"""
## Recommendations

### Migration Strategy
"""
        if assessment.complexity_score <= 3:
            report += "- **Low Complexity**: Direct migration recommended\n"
            report += "- **Downtime**: Minimal (< 1 hour)\n"
        elif assessment.complexity_score <= 6:
            report += "- **Medium Complexity**: Phased migration recommended\n"
            report += "- **Downtime**: Moderate (1-4 hours)\n"
        else:
            report += "- **High Complexity**: Careful planning and testing required\n"
            report += "- **Downtime**: Extended (4+ hours)\n"
        
        report += f"""
### Pre-Migration Steps
1. **Backup**: Create full backup of existing Neo4j database
2. **Quality**: Address data quality issues identified above
3. **Testing**: Test migration with subset of data
4. **Resources**: Ensure sufficient disk space and processing capacity

### Migration Phases
1. **Phase 1**: Migrate documents and chunks
2. **Phase 2**: Migrate entities with deduplication
3. **Phase 3**: Migrate relations and validate integrity
4. **Phase 4**: Verify search functionality and performance

### Rollback Plan
- Maintain original Neo4j database until migration is validated
- Document rollback procedures for each migration phase
- Test rollback procedures before starting migration
"""
        
        return report
```

### 2. Create Data Extraction Tool

**File**: `packages/morag-graph/src/morag_graph/graphiti/migration/extractor.py`

```python
"""Data extraction tool for migrating from Neo4j to Graphiti."""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Iterator, AsyncIterator
from datetime import datetime
from dataclasses import dataclass, asdict

from morag_graph.models import Document, DocumentChunk, Entity, Relation

logger = logging.getLogger(__name__)


@dataclass
class ExtractionBatch:
    """Batch of extracted data for migration."""
    documents: List[Document]
    chunks: List[DocumentChunk]
    entities: List[Entity]
    relations: List[Relation]
    batch_id: str
    extraction_timestamp: datetime


class Neo4jDataExtractor:
    """Tool for extracting data from Neo4j for migration to Graphiti."""
    
    def __init__(self, neo4j_storage, batch_size: int = 1000):
        self.neo4j_storage = neo4j_storage
        self.batch_size = batch_size
        
    async def extract_all_data(
        self, 
        output_dir: str = "migration_data",
        validate_integrity: bool = True
    ) -> Dict[str, Any]:
        """Extract all data from Neo4j in batches.
        
        Args:
            output_dir: Directory to save extracted data
            validate_integrity: Whether to validate data integrity
            
        Returns:
            Extraction summary
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        extraction_summary = {
            "start_time": datetime.now().isoformat(),
            "batches_created": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_relations": 0,
            "validation_errors": [],
            "output_files": []
        }
        
        logger.info("Starting data extraction from Neo4j...")
        
        try:
            # Extract data in batches
            async for batch in self.extract_data_batches():
                # Save batch to file
                batch_file = os.path.join(output_dir, f"batch_{batch.batch_id}.json")
                await self._save_batch_to_file(batch, batch_file)
                
                # Update summary
                extraction_summary["batches_created"] += 1
                extraction_summary["total_documents"] += len(batch.documents)
                extraction_summary["total_chunks"] += len(batch.chunks)
                extraction_summary["total_entities"] += len(batch.entities)
                extraction_summary["total_relations"] += len(batch.relations)
                extraction_summary["output_files"].append(batch_file)
                
                # Validate batch if requested
                if validate_integrity:
                    validation_errors = await self._validate_batch_integrity(batch)
                    extraction_summary["validation_errors"].extend(validation_errors)
                
                logger.info(f"Extracted batch {batch.batch_id}: "
                          f"{len(batch.documents)} docs, {len(batch.chunks)} chunks, "
                          f"{len(batch.entities)} entities, {len(batch.relations)} relations")
            
            extraction_summary["end_time"] = datetime.now().isoformat()
            extraction_summary["success"] = True
            
            # Save extraction summary
            summary_file = os.path.join(output_dir, "extraction_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(extraction_summary, f, indent=2, default=str)
            
            logger.info(f"Data extraction completed. {extraction_summary['batches_created']} batches created.")
            
        except Exception as e:
            extraction_summary["error"] = str(e)
            extraction_summary["success"] = False
            logger.error(f"Data extraction failed: {e}")
        
        return extraction_summary
    
    async def extract_data_batches(self) -> AsyncIterator[ExtractionBatch]:
        """Extract data in batches for memory-efficient processing."""
        batch_id = 0
        
        # Get total counts for progress tracking
        total_documents = await self._count_total_documents()
        
        # Extract documents in batches
        for offset in range(0, total_documents, self.batch_size):
            batch_id += 1
            
            # Extract documents for this batch
            documents = await self._extract_documents_batch(offset, self.batch_size)
            
            if not documents:
                break
            
            # Extract related data for these documents
            document_ids = [doc.id for doc in documents]
            chunks = await self._extract_chunks_for_documents(document_ids)
            entities = await self._extract_entities_for_documents(document_ids)
            relations = await self._extract_relations_for_documents(document_ids)
            
            batch = ExtractionBatch(
                documents=documents,
                chunks=chunks,
                entities=entities,
                relations=relations,
                batch_id=f"{batch_id:04d}",
                extraction_timestamp=datetime.now()
            )
            
            yield batch
    
    async def _extract_documents_batch(self, offset: int, limit: int) -> List[Document]:
        """Extract a batch of documents from Neo4j."""
        try:
            query = """
            MATCH (d:Document)
            RETURN d
            ORDER BY d.id
            SKIP $offset LIMIT $limit
            """
            
            results = await self.neo4j_storage._execute_query(
                query, {"offset": offset, "limit": limit}
            )
            
            documents = []
            for row in results:
                doc_node = row['d']
                document = Document(
                    id=doc_node.get('id'),
                    name=doc_node.get('name'),
                    source_file=doc_node.get('source_file'),
                    file_name=doc_node.get('file_name'),
                    file_size=doc_node.get('file_size'),
                    checksum=doc_node.get('checksum'),
                    mime_type=doc_node.get('mime_type'),
                    summary=doc_node.get('summary'),
                    metadata=doc_node.get('metadata', {}),
                    ingestion_timestamp=self._parse_timestamp(doc_node.get('ingestion_timestamp')),
                    last_modified=self._parse_timestamp(doc_node.get('last_modified')),
                    model=doc_node.get('model')
                )
                documents.append(document)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to extract documents batch: {e}")
            return []
    
    async def _extract_chunks_for_documents(self, document_ids: List[str]) -> List[DocumentChunk]:
        """Extract chunks for specific documents."""
        try:
            query = """
            MATCH (c:Chunk)-[:BELONGS_TO]->(d:Document)
            WHERE d.id IN $document_ids
            RETURN c, d.id as document_id
            ORDER BY c.chunk_index
            """
            
            results = await self.neo4j_storage._execute_query(
                query, {"document_ids": document_ids}
            )
            
            chunks = []
            for row in results:
                chunk_node = row['c']
                chunk = DocumentChunk(
                    id=chunk_node.get('id'),
                    document_id=row['document_id'],
                    chunk_index=chunk_node.get('chunk_index', 0),
                    text=chunk_node.get('text', ''),
                    metadata=chunk_node.get('metadata', {})
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to extract chunks: {e}")
            return []
    
    async def _extract_entities_for_documents(self, document_ids: List[str]) -> List[Entity]:
        """Extract entities for specific documents."""
        try:
            query = """
            MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
            WHERE d.id IN $document_ids
            RETURN DISTINCT e
            """
            
            results = await self.neo4j_storage._execute_query(
                query, {"document_ids": document_ids}
            )
            
            entities = []
            for row in results:
                entity_node = row['e']
                
                # Convert entity type
                from morag_graph.models import EntityType
                entity_type = EntityType.UNKNOWN
                try:
                    entity_type = EntityType(entity_node.get('type', 'UNKNOWN'))
                except ValueError:
                    pass
                
                entity = Entity(
                    id=entity_node.get('id'),
                    name=entity_node.get('name'),
                    type=entity_type,
                    confidence=entity_node.get('confidence', 0.5),
                    attributes=entity_node.get('attributes', {}),
                    source_doc_id=entity_node.get('source_doc_id'),
                    mentioned_in_chunks=entity_node.get('mentioned_in_chunks', [])
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []
    
    async def _extract_relations_for_documents(self, document_ids: List[str]) -> List[Relation]:
        """Extract relations for specific documents."""
        try:
            query = """
            MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
            WHERE r.source_doc_id IN $document_ids
            RETURN r, e1.id as source_id, e2.id as target_id
            """
            
            results = await self.neo4j_storage._execute_query(
                query, {"document_ids": document_ids}
            )
            
            relations = []
            for row in results:
                rel_node = row['r']
                
                # Convert relation type
                from morag_graph.models import RelationType
                relation_type = RelationType.UNKNOWN
                try:
                    relation_type = RelationType(rel_node.get('relation_type', 'UNKNOWN'))
                except ValueError:
                    pass
                
                relation = Relation(
                    id=rel_node.get('id'),
                    source_entity_id=row['source_id'],
                    target_entity_id=row['target_id'],
                    relation_type=relation_type,
                    confidence=rel_node.get('confidence', 0.5),
                    attributes=rel_node.get('attributes', {}),
                    source_doc_id=rel_node.get('source_doc_id'),
                    mentioned_in_chunks=rel_node.get('mentioned_in_chunks', [])
                )
                relations.append(relation)
            
            return relations
            
        except Exception as e:
            logger.error(f"Failed to extract relations: {e}")
            return []
    
    async def _count_total_documents(self) -> int:
        """Count total documents for progress tracking."""
        try:
            query = "MATCH (d:Document) RETURN count(d) as count"
            result = await self.neo4j_storage._execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
    
    async def _save_batch_to_file(self, batch: ExtractionBatch, file_path: str):
        """Save extraction batch to JSON file."""
        try:
            batch_data = {
                "batch_id": batch.batch_id,
                "extraction_timestamp": batch.extraction_timestamp.isoformat(),
                "documents": [asdict(doc) for doc in batch.documents],
                "chunks": [asdict(chunk) for chunk in batch.chunks],
                "entities": [asdict(entity) for entity in batch.entities],
                "relations": [asdict(relation) for relation in batch.relations]
            }
            
            with open(file_path, 'w') as f:
                json.dump(batch_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save batch to file: {e}")
            raise
    
    async def _validate_batch_integrity(self, batch: ExtractionBatch) -> List[str]:
        """Validate integrity of extracted batch."""
        errors = []
        
        # Check for missing required fields
        for doc in batch.documents:
            if not doc.id or not doc.name:
                errors.append(f"Document missing required fields: {doc.id}")
        
        for chunk in batch.chunks:
            if not chunk.id or not chunk.document_id or not chunk.text:
                errors.append(f"Chunk missing required fields: {chunk.id}")
        
        for entity in batch.entities:
            if not entity.id or not entity.name:
                errors.append(f"Entity missing required fields: {entity.id}")
        
        for relation in batch.relations:
            if not relation.id or not relation.source_entity_id or not relation.target_entity_id:
                errors.append(f"Relation missing required fields: {relation.id}")
        
        # Check referential integrity
        document_ids = {doc.id for doc in batch.documents}
        for chunk in batch.chunks:
            if chunk.document_id not in document_ids:
                errors.append(f"Chunk {chunk.id} references non-existent document {chunk.document_id}")
        
        entity_ids = {entity.id for entity in batch.entities}
        for relation in batch.relations:
            if relation.source_entity_id not in entity_ids:
                errors.append(f"Relation {relation.id} references non-existent source entity")
            if relation.target_entity_id not in entity_ids:
                errors.append(f"Relation {relation.id} references non-existent target entity")
        
        return errors
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        if not timestamp_str:
            return None
        
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            logger.warning(f"Failed to parse timestamp: {timestamp_str}")
            return None
```

### 3. Create Migration Execution Tool

**File**: `packages/morag-graph/src/morag_graph/graphiti/migration/executor.py`

```python
"""Migration execution tool for importing data into Graphiti."""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from ..integration_service import GraphitiIntegrationService
from .extractor import ExtractionBatch

logger = logging.getLogger(__name__)


class MigrationExecutor:
    """Tool for executing migration from extracted Neo4j data to Graphiti."""
    
    def __init__(self, graphiti_service: GraphitiIntegrationService):
        self.graphiti_service = graphiti_service
        
    async def execute_migration(
        self,
        extraction_dir: str,
        dry_run: bool = False,
        validate_results: bool = True
    ) -> Dict[str, Any]:
        """Execute migration from extracted data.
        
        Args:
            extraction_dir: Directory containing extracted data batches
            dry_run: If True, validate but don't actually migrate
            validate_results: Whether to validate migrated data
            
        Returns:
            Migration results summary
        """
        migration_results = {
            "start_time": datetime.now().isoformat(),
            "dry_run": dry_run,
            "batches_processed": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_relations": 0,
            "successful_documents": 0,
            "successful_chunks": 0,
            "successful_entities": 0,
            "successful_relations": 0,
            "errors": [],
            "validation_results": {}
        }
        
        try:
            # Find all batch files
            batch_files = list(Path(extraction_dir).glob("batch_*.json"))
            batch_files.sort()
            
            logger.info(f"Found {len(batch_files)} batch files for migration")
            
            for batch_file in batch_files:
                logger.info(f"Processing batch: {batch_file.name}")
                
                try:
                    # Load batch data
                    batch = await self._load_batch_from_file(batch_file)
                    
                    # Process batch
                    batch_results = await self._process_batch(batch, dry_run)
                    
                    # Update results
                    migration_results["batches_processed"] += 1
                    migration_results["total_documents"] += len(batch.documents)
                    migration_results["total_chunks"] += len(batch.chunks)
                    migration_results["total_entities"] += len(batch.entities)
                    migration_results["total_relations"] += len(batch.relations)
                    
                    migration_results["successful_documents"] += batch_results["successful_documents"]
                    migration_results["successful_chunks"] += batch_results["successful_chunks"]
                    migration_results["successful_entities"] += batch_results["successful_entities"]
                    migration_results["successful_relations"] += batch_results["successful_relations"]
                    
                    migration_results["errors"].extend(batch_results["errors"])
                    
                except Exception as e:
                    error_msg = f"Failed to process batch {batch_file.name}: {str(e)}"
                    migration_results["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Validate results if requested
            if validate_results and not dry_run:
                validation_results = await self._validate_migration_results(migration_results)
                migration_results["validation_results"] = validation_results
            
            migration_results["end_time"] = datetime.now().isoformat()
            migration_results["success"] = len(migration_results["errors"]) == 0
            
            logger.info(f"Migration completed. Processed {migration_results['batches_processed']} batches.")
            
        except Exception as e:
            migration_results["error"] = str(e)
            migration_results["success"] = False
            logger.error(f"Migration failed: {e}")
        
        return migration_results
    
    async def _load_batch_from_file(self, batch_file: Path) -> ExtractionBatch:
        """Load extraction batch from JSON file."""
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
        
        # Convert dictionaries back to model objects
        from morag_graph.models import Document, DocumentChunk, Entity, Relation, EntityType, RelationType
        
        documents = []
        for doc_data in batch_data["documents"]:
            doc = Document(**doc_data)
            documents.append(doc)
        
        chunks = []
        for chunk_data in batch_data["chunks"]:
            chunk = DocumentChunk(**chunk_data)
            chunks.append(chunk)
        
        entities = []
        for entity_data in batch_data["entities"]:
            # Handle enum conversion
            if isinstance(entity_data["type"], str):
                entity_data["type"] = EntityType(entity_data["type"])
            entity = Entity(**entity_data)
            entities.append(entity)
        
        relations = []
        for relation_data in batch_data["relations"]:
            # Handle enum conversion
            if isinstance(relation_data["relation_type"], str):
                relation_data["relation_type"] = RelationType(relation_data["relation_type"])
            relation = Relation(**relation_data)
            relations.append(relation)
        
        return ExtractionBatch(
            documents=documents,
            chunks=chunks,
            entities=entities,
            relations=relations,
            batch_id=batch_data["batch_id"],
            extraction_timestamp=datetime.fromisoformat(batch_data["extraction_timestamp"])
        )
    
    async def _process_batch(self, batch: ExtractionBatch, dry_run: bool) -> Dict[str, Any]:
        """Process a single batch of migration data."""
        batch_results = {
            "successful_documents": 0,
            "successful_chunks": 0,
            "successful_entities": 0,
            "successful_relations": 0,
            "errors": []
        }
        
        if dry_run:
            # For dry run, just validate the data structure
            batch_results["successful_documents"] = len(batch.documents)
            batch_results["successful_chunks"] = len(batch.chunks)
            batch_results["successful_entities"] = len(batch.entities)
            batch_results["successful_relations"] = len(batch.relations)
            return batch_results
        
        # Process documents with their chunks
        for document in batch.documents:
            try:
                # Get chunks for this document
                doc_chunks = [chunk for chunk in batch.chunks if chunk.document_id == document.id]
                
                # Get entities and relations for this document
                doc_entities = [entity for entity in batch.entities if entity.source_doc_id == document.id]
                doc_relations = [relation for relation in batch.relations if relation.source_doc_id == document.id]
                
                # Ingest using Graphiti
                result = await self.graphiti_service.ingest_document_with_graph_data(
                    document, doc_chunks, doc_entities, doc_relations
                )
                
                if result.success:
                    batch_results["successful_documents"] += 1
                    batch_results["successful_chunks"] += len(doc_chunks)
                    batch_results["successful_entities"] += len(doc_entities)
                    batch_results["successful_relations"] += len(doc_relations)
                else:
                    batch_results["errors"].append(f"Document {document.id}: {result.error}")
                
            except Exception as e:
                error_msg = f"Failed to process document {document.id}: {str(e)}"
                batch_results["errors"].append(error_msg)
                logger.error(error_msg)
        
        return batch_results
    
    async def _validate_migration_results(self, migration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate migration results by checking data in Graphiti."""
        validation = {
            "documents_found": 0,
            "entities_found": 0,
            "relations_found": 0,
            "search_test_passed": False,
            "validation_errors": []
        }
        
        try:
            # Test basic search functionality
            search_results = await self.graphiti_service.search_with_backend(
                "test", limit=10
            )
            
            if search_results and "results" in search_results:
                validation["search_test_passed"] = True
                validation["documents_found"] = len([
                    r for r in search_results["results"] 
                    if r.get("document_id")
                ])
            
            # Additional validation could include:
            # - Counting episodes by type
            # - Testing entity search
            # - Validating relationship integrity
            
        except Exception as e:
            validation["validation_errors"].append(f"Validation failed: {str(e)}")
        
        return validation
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_migration_tools.py`

```python
"""Unit tests for migration tools."""

import pytest
import tempfile
import json
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from morag_graph.graphiti.migration.assessment import MigrationAssessmentTool, MigrationAssessment
from morag_graph.graphiti.migration.extractor import Neo4jDataExtractor, ExtractionBatch
from morag_graph.models import Document, DocumentChunk, Entity, Relation, EntityType, RelationType


class TestMigrationAssessmentTool:
    """Test migration assessment functionality."""
    
    @pytest.fixture
    def mock_neo4j_storage(self):
        """Create mock Neo4j storage."""
        storage = Mock()
        storage._execute_query = AsyncMock()
        return storage
    
    @pytest.fixture
    def assessment_tool(self, mock_neo4j_storage):
        """Create assessment tool."""
        return MigrationAssessmentTool(mock_neo4j_storage)
    
    @pytest.mark.asyncio
    async def test_count_documents(self, assessment_tool, mock_neo4j_storage):
        """Test document counting."""
        mock_neo4j_storage._execute_query.return_value = [{"count": 100}]
        
        count = await assessment_tool._count_documents()
        
        assert count == 100
        mock_neo4j_storage._execute_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_entity_types(self, assessment_tool, mock_neo4j_storage):
        """Test entity type analysis."""
        mock_neo4j_storage._execute_query.return_value = [
            {"entity_type": "PERSON", "count": 50},
            {"entity_type": "ORGANIZATION", "count": 30},
            {"entity_type": "LOCATION", "count": 20}
        ]
        
        distribution = await assessment_tool._analyze_entity_types()
        
        assert distribution["PERSON"] == 50
        assert distribution["ORGANIZATION"] == 30
        assert distribution["LOCATION"] == 20
    
    def test_estimate_migration_time(self, assessment_tool):
        """Test migration time estimation."""
        time_estimate = assessment_tool._estimate_migration_time(
            documents=1000,
            chunks=5000,
            entities=2000,
            relations=1500
        )
        
        assert time_estimate >= 1.0  # Minimum 1 hour
        assert isinstance(time_estimate, float)
    
    def test_calculate_complexity_score(self, assessment_tool):
        """Test complexity score calculation."""
        score = assessment_tool._calculate_complexity_score(
            entities=10000,
            relations=15000,
            quality_issues=5
        )
        
        assert 0.0 <= score <= 10.0
        assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_full_assessment(self, assessment_tool, mock_neo4j_storage):
        """Test complete migration assessment."""
        # Mock all the query results
        mock_neo4j_storage._execute_query.side_effect = [
            [{"count": 100}],  # documents
            [{"count": 500}],  # chunks
            [{"count": 200}],  # entities
            [{"count": 150}],  # relations
            [{"entity_type": "PERSON", "count": 100}],  # entity types
            [{"relation_type": "MENTIONS", "count": 75}],  # relation types
            [{"count": 0}],  # entities without names
            [{"count": 0}],  # relations without types
            [{"count": 0}],  # orphaned chunks
            [{"count": 0}]   # duplicate entities
        ]
        
        assessment = await assessment_tool.assess_migration_feasibility()
        
        assert isinstance(assessment, MigrationAssessment)
        assert assessment.total_documents == 100
        assert assessment.total_chunks == 500
        assert assessment.total_entities == 200
        assert assessment.total_relations == 150
        assert assessment.complexity_score >= 0.0


class TestNeo4jDataExtractor:
    """Test data extraction functionality."""
    
    @pytest.fixture
    def mock_neo4j_storage(self):
        """Create mock Neo4j storage."""
        storage = Mock()
        storage._execute_query = AsyncMock()
        return storage
    
    @pytest.fixture
    def extractor(self, mock_neo4j_storage):
        """Create data extractor."""
        return Neo4jDataExtractor(mock_neo4j_storage, batch_size=10)
    
    @pytest.mark.asyncio
    async def test_extract_documents_batch(self, extractor, mock_neo4j_storage):
        """Test document batch extraction."""
        mock_neo4j_storage._execute_query.return_value = [
            {
                "d": {
                    "id": "doc_1",
                    "name": "Test Document",
                    "source_file": "/test/doc.pdf",
                    "file_name": "doc.pdf"
                }
            }
        ]
        
        documents = await extractor._extract_documents_batch(0, 10)
        
        assert len(documents) == 1
        assert documents[0].id == "doc_1"
        assert documents[0].name == "Test Document"
    
    @pytest.mark.asyncio
    async def test_extract_chunks_for_documents(self, extractor, mock_neo4j_storage):
        """Test chunk extraction for documents."""
        mock_neo4j_storage._execute_query.return_value = [
            {
                "c": {
                    "id": "chunk_1",
                    "chunk_index": 0,
                    "text": "Test chunk content"
                },
                "document_id": "doc_1"
            }
        ]
        
        chunks = await extractor._extract_chunks_for_documents(["doc_1"])
        
        assert len(chunks) == 1
        assert chunks[0].id == "chunk_1"
        assert chunks[0].document_id == "doc_1"
        assert chunks[0].text == "Test chunk content"
    
    @pytest.mark.asyncio
    async def test_batch_validation(self, extractor):
        """Test batch integrity validation."""
        # Create test batch with valid data
        batch = ExtractionBatch(
            documents=[Document(id="doc_1", name="Test Doc", file_name="test.txt")],
            chunks=[DocumentChunk(id="chunk_1", document_id="doc_1", chunk_index=0, text="Test")],
            entities=[Entity(id="entity_1", name="Test Entity", type=EntityType.PERSON)],
            relations=[Relation(
                id="rel_1", 
                source_entity_id="entity_1", 
                target_entity_id="entity_1",
                relation_type=RelationType.MENTIONS
            )],
            batch_id="test_batch",
            extraction_timestamp=datetime.now()
        )
        
        errors = await extractor._validate_batch_integrity(batch)
        
        # Should have no errors for valid data
        assert len(errors) == 0
        
        # Test with invalid data
        invalid_batch = ExtractionBatch(
            documents=[Document(id="", name="", file_name="")],  # Missing required fields
            chunks=[],
            entities=[],
            relations=[],
            batch_id="invalid_batch",
            extraction_timestamp=datetime.now()
        )
        
        errors = await extractor._validate_batch_integrity(invalid_batch)
        assert len(errors) > 0
```

## Validation Checklist

- [ ] Migration assessment accurately analyzes existing data
- [ ] Data extraction preserves all information from Neo4j
- [ ] Batch processing handles large datasets efficiently
- [ ] Data validation catches integrity issues
- [ ] Migration execution successfully imports to Graphiti
- [ ] Rollback procedures are documented and tested
- [ ] Progress monitoring provides meaningful feedback
- [ ] Error handling gracefully manages failures
- [ ] Unit tests cover all migration components
- [ ] Integration tests validate end-to-end migration

## Success Criteria

1. **Data Integrity**: All data migrates without loss or corruption
2. **Performance**: Migration completes within estimated timeframes
3. **Reliability**: Process handles errors gracefully with rollback options
4. **Monitoring**: Clear progress tracking and error reporting
5. **Validation**: Comprehensive verification of migrated data

## Next Steps

After completing this step:
1. Test migration with sample data
2. Validate migration performance and accuracy
3. Document rollback and recovery procedures
4. Proceed to [Step 12: Production Deployment and Cleanup](./step-12-production-deployment.md)

## Performance Considerations

- Batch processing reduces memory usage for large datasets
- Parallel processing where possible to improve speed
- Progress tracking for long-running migrations
- Efficient data validation to minimize overhead
