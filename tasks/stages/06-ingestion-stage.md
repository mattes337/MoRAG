# Task 6: Implement ingestor Stage

## Overview
Implement the ingestor stage that performs actual database ingestion into Qdrant, Neo4j, and other databases. This completely replaces all existing ingestion logic.

## Objectives
- **COMPLETELY REPLACE** ingestion coordinator with new stage interface
- Add multi-database configuration support
- Implement ingestion result tracking
- Add conflict resolution and deduplication
- Support configurable database targets
- **REMOVE ALL LEGACY INGESTION CODE**

## Deliverables

### 1. ingestor Stage Implementation (Complete Replacement)
```python
from morag_stages.models import Stage, StageType, StageResult, StageContext, StageStatus
from morag_graph.models.database_config import DatabaseConfig, DatabaseType
from morag_qdrant import QdrantService
from morag_graph import Neo4jService
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import json

class IngestorStage(Stage):
    def __init__(self,
                 qdrant_service: QdrantService,
                 neo4j_service: Neo4jService):
        super().__init__(StageType.INGESTOR)
        self.qdrant_service = qdrant_service
        self.neo4j_service = neo4j_service

    async def execute(self,
                     input_files: List[Path],
                     context: StageContext) -> StageResult:
        """Perform database ingestion from chunks and facts."""
        start_time = time.time()

        try:
            # Validate inputs
            chunks_file, facts_file = self._validate_and_parse_inputs(input_files)

            # Load data
            chunks_data = self._load_json_file(chunks_file)
            facts_data = self._load_json_file(facts_file)

            # Configure databases
            config = context.config.get('stage5', {})
            database_configs = self._configure_databases(config)

            # Prepare ingestion data
            ingestion_data = self._prepare_ingestion_data(
                chunks_data, facts_data, context
            )

            # Perform ingestion
            ingestion_results = await self._perform_ingestion(
                ingestion_data, database_configs, config
            )

            # Generate output files
            output_file = self._generate_ingestion_output(
                ingestion_results, chunks_file, context
            )

            # Create ingestion report
            report_file = self._create_ingestion_report(
                ingestion_results, chunks_file, context
            )

            execution_time = time.time() - start_time

            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file, report_file],
                metadata={
                    "databases_used": [db.type.value for db in database_configs],
                    "chunks_ingested": len(chunks_data.get('chunks', [])),
                    "facts_ingested": len(facts_data.get('facts', [])),
                    "entities_ingested": len(facts_data.get('entities', [])),
                    "relations_ingested": len(facts_data.get('relations', [])),
                    "processing_time": execution_time
                },
                execution_time=execution_time
            )

        except Exception as e:
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )

    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for Stage 5."""
        if len(input_files) != 2:
            return False

        # Check for chunks.json and facts.json files
        file_types = set()
        for file in input_files:
            if file.name.endswith('.chunks.json'):
                file_types.add('chunks')
            elif file.name.endswith('.facts.json'):
                file_types.add('facts')

        return len(file_types) == 2 and 'chunks' in file_types and 'facts' in file_types

    def get_dependencies(self) -> List[StageType]:
        """ingestor depends on chunker and fact-generator."""
        return [StageType.CHUNKER, StageType.FACT_GENERATOR]

    def _validate_and_parse_inputs(self, input_files: List[Path]) -> tuple[Path, Path]:
        """Validate and identify chunks and facts files."""
        chunks_file = None
        facts_file = None

        for file in input_files:
            if file.name.endswith('.chunks.json'):
                chunks_file = file
            elif file.name.endswith('.facts.json'):
                facts_file = file

        if not chunks_file or not facts_file:
            raise ValueError("Stage 5 requires both .chunks.json and .facts.json files")

        return chunks_file, facts_file

    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON data from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _configure_databases(self, config: Dict[str, Any]) -> List[DatabaseConfig]:
        """Configure database connections based on config."""
        database_configs = []

        databases = config.get('databases', ['qdrant'])

        for db_name in databases:
            if db_name.lower() == 'qdrant':
                database_configs.append(DatabaseConfig(
                    type=DatabaseType.QDRANT,
                    collection_name=config.get('collection_name', 'morag_collection'),
                    config={
                        'host': config.get('qdrant_host', 'localhost'),
                        'port': config.get('qdrant_port', 6333),
                        'api_key': config.get('qdrant_api_key')
                    }
                ))
            elif db_name.lower() == 'neo4j':
                database_configs.append(DatabaseConfig(
                    type=DatabaseType.NEO4J,
                    collection_name=config.get('database_name', 'morag'),
                    config={
                        'uri': config.get('neo4j_uri', 'bolt://localhost:7687'),
                        'username': config.get('neo4j_username', 'neo4j'),
                        'password': config.get('neo4j_password', 'password')
                    }
                ))

        return database_configs

    def _prepare_ingestion_data(self,
                               chunks_data: Dict[str, Any],
                               facts_data: Dict[str, Any],
                               context: StageContext) -> Dict[str, Any]:
        """Prepare data for ingestion."""

        # Extract document metadata
        document_metadata = chunks_data.get('document_metadata', {})

        # Prepare chunks for vector storage
        chunks = chunks_data.get('chunks', [])
        for chunk in chunks:
            # Ensure chunk has required fields
            if 'embedding' not in chunk or not chunk['embedding']:
                logger.warning(f"Chunk {chunk.get('index', 'unknown')} missing embedding")

        # Prepare facts for graph storage
        facts = facts_data.get('facts', [])
        entities = facts_data.get('entities', [])
        relations = facts_data.get('relations', [])

        return {
            'document_metadata': document_metadata,
            'chunks': chunks,
            'facts': facts,
            'entities': entities,
            'relations': relations,
            'summary': chunks_data.get('summary'),
            'source_file': context.source_path
        }

    async def _perform_ingestion(self,
                                ingestion_data: Dict[str, Any],
                                database_configs: List[DatabaseConfig],
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ingestion into configured databases."""

        results = {
            'databases': {},
            'total_chunks': len(ingestion_data['chunks']),
            'total_facts': len(ingestion_data['facts']),
            'total_entities': len(ingestion_data['entities']),
            'total_relations': len(ingestion_data['relations'])
        }

        for db_config in database_configs:
            try:
                if db_config.type == DatabaseType.QDRANT:
                    result = await self._ingest_to_qdrant(
                        ingestion_data, db_config, config
                    )
                elif db_config.type == DatabaseType.NEO4J:
                    result = await self._ingest_to_neo4j(
                        ingestion_data, db_config, config
                    )
                else:
                    result = {'status': 'unsupported', 'error': f'Unsupported database type: {db_config.type}'}

                results['databases'][db_config.type.value] = result

            except Exception as e:
                logger.error(f"Failed to ingest to {db_config.type.value}: {e}")
                results['databases'][db_config.type.value] = {
                    'status': 'failed',
                    'error': str(e)
                }

        return results

    async def _ingest_to_qdrant(self,
                               ingestion_data: Dict[str, Any],
                               db_config: DatabaseConfig,
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest chunks to Qdrant vector database."""

        chunks = ingestion_data['chunks']
        collection_name = db_config.collection_name

        # Ensure collection exists
        await self.qdrant_service.ensure_collection_exists(
            collection_name,
            vector_size=len(chunks[0]['embedding']) if chunks and chunks[0].get('embedding') else 1536
        )

        # Prepare points for Qdrant
        points = []
        for chunk in chunks:
            if chunk.get('embedding'):
                point = {
                    'id': f"{ingestion_data['source_file'].stem}_{chunk['index']}",
                    'vector': chunk['embedding'],
                    'payload': {
                        'content': chunk['content'],
                        'chunk_index': chunk['index'],
                        'source_file': str(ingestion_data['source_file']),
                        'timestamp': chunk.get('timestamp'),
                        'token_count': chunk.get('token_count'),
                        'source_metadata': chunk.get('source_metadata', {})
                    }
                }
                points.append(point)

        # Batch insert points
        batch_size = config.get('qdrant_batch_size', 100)
        inserted_count = 0

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                await self.qdrant_service.upsert_points(collection_name, batch)
                inserted_count += len(batch)
            except Exception as e:
                logger.error(f"Failed to insert batch {i}: {e}")

        return {
            'status': 'success',
            'collection_name': collection_name,
            'points_inserted': inserted_count,
            'total_points': len(points)
        }

    async def _ingest_to_neo4j(self,
                              ingestion_data: Dict[str, Any],
                              db_config: DatabaseConfig,
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest facts, entities, and relations to Neo4j graph database."""

        entities = ingestion_data['entities']
        relations = ingestion_data['relations']
        chunks = ingestion_data['chunks']

        # Create document node
        document_id = f"doc_{ingestion_data['source_file'].stem}"
        await self.neo4j_service.create_document_node(
            document_id,
            ingestion_data['document_metadata']
        )

        # Create chunk nodes and link to document
        chunk_nodes_created = 0
        for chunk in chunks:
            try:
                chunk_id = f"{document_id}_chunk_{chunk['index']}"
                await self.neo4j_service.create_chunk_node(
                    chunk_id, chunk, document_id
                )
                chunk_nodes_created += 1
            except Exception as e:
                logger.error(f"Failed to create chunk node {chunk['index']}: {e}")

        # Create entity nodes
        entity_nodes_created = 0
        for entity in entities:
            try:
                await self.neo4j_service.create_entity_node(
                    entity['normalized_name'],
                    entity
                )
                entity_nodes_created += 1
            except Exception as e:
                logger.error(f"Failed to create entity node {entity['name']}: {e}")

        # Create relations
        relations_created = 0
        for relation in relations:
            try:
                await self.neo4j_service.create_relation(
                    relation['subject'],
                    relation['predicate'],
                    relation['object'],
                    relation
                )
                relations_created += 1
            except Exception as e:
                logger.error(f"Failed to create relation: {e}")

        # Link entities to chunks
        entity_chunk_links = 0
        for chunk in chunks:
            chunk_id = f"{document_id}_chunk_{chunk['index']}"

            # Find entities mentioned in this chunk
            chunk_content_lower = chunk['content'].lower()
            for entity in entities:
                if entity['normalized_name'].lower() in chunk_content_lower:
                    try:
                        await self.neo4j_service.link_entity_to_chunk(
                            entity['normalized_name'], chunk_id
                        )
                        entity_chunk_links += 1
                    except Exception as e:
                        logger.error(f"Failed to link entity to chunk: {e}")

        return {
            'status': 'success',
            'database_name': db_config.collection_name,
            'chunk_nodes_created': chunk_nodes_created,
            'entity_nodes_created': entity_nodes_created,
            'relations_created': relations_created,
            'entity_chunk_links': entity_chunk_links
        }

    def _generate_ingestion_output(self,
                                  ingestion_results: Dict[str, Any],
                                  chunks_file: Path,
                                  context: StageContext) -> Path:
        """Generate ingestion results JSON file."""
        from morag_stages.file_manager import FileNamingConvention

        output_filename = FileNamingConvention.get_stage_output_filename(
            chunks_file.with_suffix(''), StageType.INGESTOR
        )
        output_path = context.output_dir / output_filename

        # Create comprehensive ingestion results
        output_data = {
            'stage': 'ingestor',
            'stage_name': 'ingestor',
            'source_files': {
                'chunks_file': str(chunks_file),
                'facts_file': str(chunks_file.with_name(chunks_file.name.replace('.chunks.', '.facts.')))
            },
            'ingestion_results': ingestion_results,
            'processed_at': time.time(),
            'success': all(
                result.get('status') == 'success'
                for result in ingestion_results.get('databases', {}).values()
            )
        }

        # Write results file
        output_path.write_text(json.dumps(output_data, indent=2), encoding='utf-8')

        return output_path
```

## Implementation Steps

1. **Create ingestor package structure**
2. **Implement IngestorStage class**
3. **Add multi-database configuration support**
4. **Implement Qdrant ingestion logic**
5. **Implement Neo4j ingestion logic**
6. **Add conflict resolution and deduplication**
7. **Create ingestion result tracking**
8. **REMOVE ALL LEGACY INGESTION CODE**
9. **Add comprehensive error handling**
10. **Implement performance monitoring and comprehensive testing**

## Testing Requirements

- Unit tests for ingestion logic
- Database integration tests
- Multi-database configuration tests
- Error handling and rollback tests
- Performance tests for large datasets
- Conflict resolution validation

## Files to Create

- `packages/morag-stages/src/morag_stages/ingestor/__init__.py`
- `packages/morag-stages/src/morag_stages/ingestor/implementation.py`
- `packages/morag-stages/src/morag_stages/ingestor/database_ingestion.py`
- `packages/morag-stages/tests/test_ingestor.py`

## Success Criteria

- Data is ingested correctly into all configured databases
- Ingestion results are tracked and reported accurately
- Error handling prevents partial ingestion states
- Performance is acceptable for typical datasets
- Multi-database configuration works reliably
- **ALL LEGACY INGESTION CODE IS REMOVED**
- All tests pass with good coverage
