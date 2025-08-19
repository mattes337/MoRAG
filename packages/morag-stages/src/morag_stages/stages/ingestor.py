"""Ingestor stage implementation."""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import structlog

from ..models import Stage, StageType, StageStatus, StageResult, StageContext, StageMetadata
from ..exceptions import StageExecutionError, StageValidationError

# Import storage services with graceful fallback
try:
    from morag_graph.storage import Neo4jStorage, QdrantStorage, Neo4jConfig, QdrantConfig
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    Neo4jStorage = None
    QdrantStorage = None
    Neo4jConfig = None
    QdrantConfig = None

logger = structlog.get_logger(__name__)


class IngestorStage(Stage):
    """Stage that ingests data into databases with deduplication."""
    
    def __init__(self, stage_type: StageType = StageType.INGESTOR):
        """Initialize ingestor stage."""
        super().__init__(stage_type)
        
        if not STORAGE_AVAILABLE:
            logger.warning("Storage services not available for ingestion")
        
        self.neo4j_storage = None
        self.qdrant_storage = None
    
    async def execute(self, 
                     input_files: List[Path], 
                     context: StageContext) -> StageResult:
        """Execute ingestion on input files.
        
        Args:
            input_files: List of input files (chunks.json and facts.json)
            context: Stage execution context
            
        Returns:
            Stage execution result
        """
        if len(input_files) < 1 or len(input_files) > 2:
            raise StageValidationError(
                "Ingestor stage requires 1-2 input files (chunks.json and optionally facts.json)",
                stage_type=self.stage_type.value,
                invalid_files=[str(f) for f in input_files]
            )
        
        config = context.get_stage_config(self.stage_type)
        
        logger.info("Starting ingestion", 
                   input_files=[str(f) for f in input_files],
                   config=config)
        
        try:
            # Find chunks and facts files
            chunks_file = None
            facts_file = None
            
            for file in input_files:
                if file.name.endswith('.chunks.json'):
                    chunks_file = file
                elif file.name.endswith('.facts.json'):
                    facts_file = file
            
            if not chunks_file:
                raise StageValidationError(
                    "No chunks.json file found in input files",
                    stage_type=self.stage_type.value,
                    invalid_files=[str(f) for f in input_files]
                )
            
            # Generate output filename
            base_name = chunks_file.stem.replace('.chunks', '')
            output_file = context.output_dir / f"{base_name}.ingestion.json"
            context.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load input data
            chunks_data = self._load_json_file(chunks_file)
            facts_data = self._load_json_file(facts_file) if facts_file else None
            
            # Initialize storage backends
            await self._initialize_storage(config)
            
            # Perform ingestion
            ingestion_results = await self._perform_ingestion(
                chunks_data, facts_data, config
            )
            
            # Create output data
            output_data = {
                "ingestion_results": ingestion_results,
                "metadata": {
                    "source_files": {
                        "chunks": str(chunks_file),
                        "facts": str(facts_file) if facts_file else None
                    },
                    "databases": config.get('databases', ['qdrant']),
                    "collection_name": config.get('collection_name', 'documents'),
                    "deduplication_enabled": config.get('enable_deduplication', True),
                    "created_at": datetime.now().isoformat()
                }
            }
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Create metadata
            stage_metadata = StageMetadata(
                execution_time=0.0,  # Will be set by manager
                start_time=datetime.now(),
                input_files=[str(f) for f in input_files],
                output_files=[str(output_file)],
                config_used=config,
                metrics={
                    "chunks_ingested": ingestion_results.get('chunks_ingested', 0),
                    "entities_ingested": ingestion_results.get('entities_ingested', 0),
                    "relations_ingested": ingestion_results.get('relations_ingested', 0),
                    "facts_ingested": ingestion_results.get('facts_ingested', 0),
                    "duplicates_skipped": ingestion_results.get('duplicates_skipped', 0),
                    "databases_used": len(config.get('databases', ['qdrant']))
                }
            )
            
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file],
                metadata=stage_metadata,
                data=ingestion_results
            )
            
        except Exception as e:
            logger.error("Ingestion failed", 
                        input_files=[str(f) for f in input_files], 
                        error=str(e))
            raise StageExecutionError(
                f"Ingestion failed: {e}",
                stage_type=self.stage_type.value,
                original_error=e
            )
    
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for ingestion.
        
        Args:
            input_files: List of input file paths
            
        Returns:
            True if inputs are valid
        """
        if len(input_files) < 1 or len(input_files) > 2:
            return False
        
        # Check for required chunks file
        has_chunks = any(f.name.endswith('.chunks.json') for f in input_files)
        if not has_chunks:
            return False
        
        # Validate all files exist and are JSON
        for file in input_files:
            if not file.exists():
                return False
            
            if not file.name.endswith('.json'):
                return False
            
            # Try to parse JSON
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    json.load(f)
            except (json.JSONDecodeError, IOError):
                return False
        
        return True
    
    def get_dependencies(self) -> List[StageType]:
        """Get stage dependencies.
        
        Returns:
            List containing chunker and optionally fact-generator stages
        """
        return [StageType.CHUNKER]  # fact-generator is optional
    
    def get_expected_outputs(self, input_files: List[Path], context: StageContext) -> List[Path]:
        """Get expected output file paths.
        
        Args:
            input_files: List of input file paths
            context: Stage execution context
            
        Returns:
            List of expected output file paths
        """
        if not input_files:
            return []
        
        # Find chunks file to determine base name
        chunks_file = None
        for file in input_files:
            if file.name.endswith('.chunks.json'):
                chunks_file = file
                break
        
        if not chunks_file:
            return []
        
        base_name = chunks_file.stem.replace('.chunks', '')
        output_file = context.output_dir / f"{base_name}.ingestion.json"
        return [output_file]
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file safely.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load JSON file", file=str(file_path), error=str(e))
            raise StageExecutionError(f"Failed to load {file_path}: {e}")
    
    async def _initialize_storage(self, config: Dict[str, Any]) -> None:
        """Initialize storage backends.
        
        Args:
            config: Stage configuration
        """
        if not STORAGE_AVAILABLE:
            return
        
        databases = config.get('databases', ['qdrant'])
        
        # Initialize Qdrant if requested
        if 'qdrant' in databases and QdrantStorage:
            try:
                qdrant_config = QdrantConfig(**config.get('qdrant_config', {}))
                self.qdrant_storage = QdrantStorage(qdrant_config)
                await self.qdrant_storage.initialize()
                logger.info("Qdrant storage initialized")
            except Exception as e:
                logger.error("Failed to initialize Qdrant storage", error=str(e))
                raise StageExecutionError(f"Qdrant initialization failed: {e}")
        
        # Initialize Neo4j if requested
        if 'neo4j' in databases and Neo4jStorage:
            try:
                neo4j_config = Neo4jConfig(**config.get('neo4j_config', {}))
                self.neo4j_storage = Neo4jStorage(neo4j_config)
                await self.neo4j_storage.initialize()
                logger.info("Neo4j storage initialized")
            except Exception as e:
                logger.error("Failed to initialize Neo4j storage", error=str(e))
                raise StageExecutionError(f"Neo4j initialization failed: {e}")
    
    async def _perform_ingestion(self, 
                               chunks_data: Dict[str, Any], 
                               facts_data: Optional[Dict[str, Any]], 
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual ingestion.
        
        Args:
            chunks_data: Chunks data
            facts_data: Facts data (optional)
            config: Stage configuration
            
        Returns:
            Ingestion results
        """
        results = {
            "chunks_ingested": 0,
            "entities_ingested": 0,
            "relations_ingested": 0,
            "facts_ingested": 0,
            "duplicates_skipped": 0,
            "errors": []
        }
        
        collection_name = config.get('collection_name', 'documents')
        batch_size = config.get('batch_size', 50)
        enable_dedup = config.get('enable_deduplication', True)
        
        # Ingest chunks
        chunks = chunks_data.get('chunks', [])
        if chunks:
            chunk_results = await self._ingest_chunks(
                chunks, collection_name, batch_size, enable_dedup
            )
            results.update(chunk_results)
        
        # Ingest facts if available
        if facts_data:
            fact_results = await self._ingest_facts(
                facts_data, collection_name, batch_size, enable_dedup
            )
            results.update(fact_results)
        
        return results

    async def _ingest_chunks(self,
                           chunks: List[Dict[str, Any]],
                           collection_name: str,
                           batch_size: int,
                           enable_dedup: bool) -> Dict[str, Any]:
        """Ingest chunks into vector database.

        Args:
            chunks: List of chunk data
            collection_name: Collection name
            batch_size: Batch size for ingestion
            enable_dedup: Whether to enable deduplication

        Returns:
            Ingestion results for chunks
        """
        results = {"chunks_ingested": 0, "duplicates_skipped": 0}

        if not self.qdrant_storage:
            logger.warning("Qdrant storage not available, skipping chunk ingestion")
            return results

        try:
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                # Prepare documents for ingestion
                documents = []
                for chunk in batch:
                    doc = {
                        "id": chunk.get('id'),
                        "content": chunk.get('content', ''),
                        "embedding": chunk.get('embedding'),
                        "metadata": {
                            **chunk.get('metadata', {}),
                            "chunk_type": "document_chunk",
                            "context_summary": chunk.get('context_summary', '')
                        }
                    }

                    # Check for duplicates if enabled
                    if enable_dedup:
                        content_hash = self._generate_content_hash(doc['content'])
                        if await self._is_duplicate(content_hash, collection_name):
                            results["duplicates_skipped"] += 1
                            continue
                        doc["metadata"]["content_hash"] = content_hash

                    documents.append(doc)

                # Ingest batch
                if documents:
                    await self.qdrant_storage.store_documents(documents, collection_name)
                    results["chunks_ingested"] += len(documents)

                    logger.info("Ingested chunk batch",
                               batch_size=len(documents),
                               total_ingested=results["chunks_ingested"])

        except Exception as e:
            logger.error("Chunk ingestion failed", error=str(e))
            raise StageExecutionError(f"Chunk ingestion failed: {e}")

        return results

    async def _ingest_facts(self,
                          facts_data: Dict[str, Any],
                          collection_name: str,
                          batch_size: int,
                          enable_dedup: bool) -> Dict[str, Any]:
        """Ingest facts into graph database.

        Args:
            facts_data: Facts data
            collection_name: Collection name
            batch_size: Batch size for ingestion
            enable_dedup: Whether to enable deduplication

        Returns:
            Ingestion results for facts
        """
        results = {
            "entities_ingested": 0,
            "relations_ingested": 0,
            "facts_ingested": 0
        }

        if not self.neo4j_storage:
            logger.warning("Neo4j storage not available, skipping fact ingestion")
            return results

        try:
            # Ingest entities
            entities = facts_data.get('entities', [])
            if entities:
                entity_results = await self._ingest_entities(entities, enable_dedup)
                results["entities_ingested"] = entity_results.get("ingested", 0)

            # Ingest relations
            relations = facts_data.get('relations', [])
            if relations:
                relation_results = await self._ingest_relations(relations, enable_dedup)
                results["relations_ingested"] = relation_results.get("ingested", 0)

            # Ingest facts as nodes
            facts = facts_data.get('facts', [])
            if facts:
                fact_results = await self._ingest_fact_nodes(facts, enable_dedup)
                results["facts_ingested"] = fact_results.get("ingested", 0)

        except Exception as e:
            logger.error("Fact ingestion failed", error=str(e))
            raise StageExecutionError(f"Fact ingestion failed: {e}")

        return results

    async def _ingest_entities(self, entities: List[Dict[str, Any]], enable_dedup: bool) -> Dict[str, Any]:
        """Ingest entities into Neo4j.

        Args:
            entities: List of entity data
            enable_dedup: Whether to enable deduplication

        Returns:
            Entity ingestion results
        """
        results = {"ingested": 0, "duplicates_skipped": 0}

        for entity in entities:
            try:
                # Check for duplicates if enabled
                if enable_dedup:
                    existing = await self.neo4j_storage.find_entity_by_name(
                        entity.get('normalized_name', entity.get('name', ''))
                    )
                    if existing:
                        results["duplicates_skipped"] += 1
                        continue

                # Create entity node
                await self.neo4j_storage.create_entity(
                    name=entity.get('name', ''),
                    entity_type=entity.get('type', 'Entity'),
                    normalized_name=entity.get('normalized_name', ''),
                    confidence=entity.get('confidence', 0.0),
                    source_chunks=entity.get('source_chunks', []),
                    metadata=entity.get('metadata', {})
                )

                results["ingested"] += 1

            except Exception as e:
                logger.warning("Failed to ingest entity", entity=entity.get('name'), error=str(e))

        return results

    async def _ingest_relations(self, relations: List[Dict[str, Any]], enable_dedup: bool) -> Dict[str, Any]:
        """Ingest relations into Neo4j.

        Args:
            relations: List of relation data
            enable_dedup: Whether to enable deduplication

        Returns:
            Relation ingestion results
        """
        results = {"ingested": 0, "duplicates_skipped": 0}

        for relation in relations:
            try:
                subject = relation.get('subject', '')
                predicate = relation.get('predicate', '')
                obj = relation.get('object', '')

                if not all([subject, predicate, obj]):
                    continue

                # Check for duplicates if enabled
                if enable_dedup:
                    existing = await self.neo4j_storage.find_relation(subject, predicate, obj)
                    if existing:
                        results["duplicates_skipped"] += 1
                        continue

                # Create relation
                await self.neo4j_storage.create_relation(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    confidence=relation.get('confidence', 0.0),
                    source_chunks=relation.get('source_chunks', []),
                    metadata=relation.get('metadata', {})
                )

                results["ingested"] += 1

            except Exception as e:
                logger.warning("Failed to ingest relation",
                             subject=relation.get('subject'),
                             predicate=relation.get('predicate'),
                             error=str(e))

        return results

    async def _ingest_fact_nodes(self, facts: List[Dict[str, Any]], enable_dedup: bool) -> Dict[str, Any]:
        """Ingest facts as nodes into Neo4j.

        Args:
            facts: List of fact data
            enable_dedup: Whether to enable deduplication

        Returns:
            Fact ingestion results
        """
        results = {"ingested": 0, "duplicates_skipped": 0}

        for fact in facts:
            try:
                statement = fact.get('statement', '')
                if not statement:
                    continue

                # Check for duplicates if enabled
                if enable_dedup:
                    statement_hash = self._generate_content_hash(statement)
                    existing = await self.neo4j_storage.find_fact_by_hash(statement_hash)
                    if existing:
                        results["duplicates_skipped"] += 1
                        continue

                # Create fact node
                await self.neo4j_storage.create_fact(
                    statement=statement,
                    entities=fact.get('entities', []),
                    confidence=fact.get('confidence', 0.0),
                    source_chunk=fact.get('source_chunk', ''),
                    metadata=fact.get('metadata', {})
                )

                results["ingested"] += 1

            except Exception as e:
                logger.warning("Failed to ingest fact",
                             statement=fact.get('statement', '')[:100],
                             error=str(e))

        return results

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication.

        Args:
            content: Content to hash

        Returns:
            Content hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    async def _is_duplicate(self, content_hash: str, collection_name: str) -> bool:
        """Check if content is duplicate.

        Args:
            content_hash: Content hash
            collection_name: Collection name

        Returns:
            True if duplicate exists
        """
        if not self.qdrant_storage:
            return False

        try:
            # Search for existing document with same hash
            results = await self.qdrant_storage.search_by_metadata(
                collection_name,
                {"content_hash": content_hash},
                limit=1
            )
            return len(results) > 0
        except Exception:
            return False
