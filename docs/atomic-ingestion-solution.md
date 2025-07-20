# Atomic Ingestion Solution

## Overview

This document describes the comprehensive solution implemented to address ingestion consistency issues in the MoRAG system. The solution provides atomic ingestion with pre-validation, transaction management, and error recovery.

## Problem Statement

The original ingestion process had several critical issues:

1. **No Atomic Transaction Management**: Chunks were processed sequentially without proper transaction coordination, leading to partial failures and inconsistent database states.

2. **JSON Parsing Errors**: LLM responses containing malformed JSON (e.g., "Unterminated string starting at: line 1 column 379") caused ingestion failures without proper recovery.

3. **No Rollback Mechanism**: When errors occurred during ingestion, there was no way to rollback already-written data, leaving databases in inconsistent states.

4. **Lack of Pre-validation**: The system would start writing to databases before validating all chunks, leading to wasted resources and partial ingestion.

## Solution Architecture

### 1. Enhanced JSON Parsing (`packages/morag-core/src/morag_core/utils/json_parser.py`)

**Features:**
- Multi-level error recovery for malformed JSON responses
- Extraction of JSON from markdown-wrapped text
- Common JSON issue fixes (trailing commas, missing quotes, etc.)
- Partial JSON recovery for incomplete responses
- Unterminated string handling

**Usage:**
```python
from morag_core.utils.json_parser import parse_json_response

# Automatically handles malformed JSON with fallback
result = parse_json_response(llm_response, fallback_value={})
```

### 2. Transaction Coordinator (`packages/morag/src/morag/ingestion/transaction_coordinator.py`)

**Features:**
- Atomic transaction management across multiple databases
- Two-phase commit protocol (prepare → commit)
- Operation tracking and rollback capabilities
- Transaction state management
- Cleanup of old transactions

**Transaction States:**
- `PENDING`: Transaction created, operations being added
- `PREPARING`: Validating all operations
- `PREPARED`: All operations validated, ready to commit
- `COMMITTING`: Executing operations
- `COMMITTED`: All operations completed successfully
- `ABORTING`: Rolling back operations
- `ABORTED`: Rollback completed
- `FAILED`: Transaction failed

### 3. Enhanced Transaction Coordinator (`packages/morag/src/morag/ingestion/enhanced_transaction_coordinator.py`)

**Features:**
- Actual database operation execution
- Neo4j and Qdrant operation support
- Connection pooling and caching
- Rollback implementation for both databases

**Supported Operations:**
- **Neo4j**: `create_entities`, `create_relations`
- **Qdrant**: `store_vectors`

### 4. Atomic Ingestion Service (`packages/morag/src/morag/ingestion/atomic_ingestion_service.py`)

**Features:**
- Pre-validation of all chunks before any database writes
- Enhanced JSON parsing integration
- Transaction coordination
- Comprehensive error handling
- State tracking integration

**Process Flow:**
1. **Begin Transaction**: Create transaction and state tracking
2. **Pre-validation**: Process all chunks and validate LLM responses
3. **Prepare Operations**: Set up database operations for all validated data
4. **Prepare Transaction**: Validate all operations can be executed
5. **Commit Transaction**: Execute all operations atomically
6. **Update State**: Track progress throughout the process

### 5. State Management (`packages/morag/src/morag/ingestion/state_manager.py`)

**Features:**
- Persistent state tracking for ingestion processes
- Resume/retry capabilities
- State cleanup and maintenance
- Failed state identification for retry

**State Tracking:**
- `pending`: Ingestion started
- `validating`: Pre-validation in progress
- `validated`: All chunks validated successfully
- `committing`: Database operations executing
- `committed`: Ingestion completed successfully
- `failed`: Ingestion failed (with error details)
- `aborted`: Ingestion was aborted

## Usage

### Basic Atomic Ingestion

```python
from morag.ingestion_coordinator import IngestionCoordinator

coordinator = IngestionCoordinator()
await coordinator.initialize()

# Use atomic ingestion instead of regular ingestion
result = await coordinator.ingest_content_atomic(
    content=text_content,
    source_path="document.pdf",
    content_type="document",
    metadata={"author": "John Doe"},
    processing_result=processing_result,
    chunk_size=1000,
    chunk_overlap=200,
    replace_existing=False
)

if result["success"]:
    print(f"Ingestion completed: {result['transaction_id']}")
    print(f"Chunks processed: {result['chunks_processed']}")
    print(f"Entities extracted: {result['entities_extracted']}")
else:
    print(f"Ingestion failed: {result['error']}")
```

### Direct Atomic Service Usage

```python
from morag.ingestion.atomic_ingestion_service import AtomicIngestionService
from morag_core.config import DatabaseConfig, DatabaseType

service = AtomicIngestionService()

# Configure databases
databases = [
    DatabaseConfig(type=DatabaseType.NEO4J, host="localhost", port=7687),
    DatabaseConfig(type=DatabaseType.QDRANT, host="localhost", port=6333)
]

try:
    result = await service.ingest_with_validation(
        content=content,
        source_path=source_path,
        document_id=document_id,
        database_configs=databases,
        chunk_size=1000,
        chunk_overlap=200
    )
    print("Atomic ingestion successful!")
except ValidationError as e:
    print(f"Validation failed: {e}")
except Exception as e:
    print(f"Ingestion failed: {e}")
```

### State Management

```python
from morag.ingestion.state_manager import get_state_manager

state_manager = get_state_manager()

# Check failed ingestions for retry
failed_states = await state_manager.get_failed_states(max_age_hours=24)
for state in failed_states:
    print(f"Failed ingestion: {state.transaction_id} - {state.error_message}")

# Cleanup old states
cleaned = await state_manager.cleanup_old_states(max_age_hours=168)  # 7 days
print(f"Cleaned up {cleaned} old states")
```

## Benefits

### 1. **Consistency Guarantee**
- Either all chunks are ingested successfully, or none are
- No partial ingestion states in databases
- Automatic rollback on any failure

### 2. **Error Recovery**
- Enhanced JSON parsing handles malformed LLM responses
- Graceful degradation with fallback values
- Comprehensive error logging and reporting

### 3. **Resource Efficiency**
- Pre-validation prevents wasted database operations
- Early failure detection saves processing time
- Connection pooling reduces overhead

### 4. **Observability**
- Detailed state tracking throughout the process
- Transaction logging and monitoring
- Failed state identification for debugging

### 5. **Reliability**
- Robust error handling at every step
- Automatic cleanup of failed transactions
- Resume/retry capabilities for failed ingestions

## Migration Guide

### From Regular Ingestion to Atomic Ingestion

1. **Replace method calls:**
   ```python
   # Old
   result = await coordinator.ingest_content(...)
   
   # New
   result = await coordinator.ingest_content_atomic(...)
   ```

2. **Handle validation errors:**
   ```python
   try:
       result = await coordinator.ingest_content_atomic(...)
   except ValidationError as e:
       # Handle validation failures specifically
       logger.error(f"Validation failed: {e}")
   except Exception as e:
       # Handle other ingestion failures
       logger.error(f"Ingestion failed: {e}")
   ```

3. **Monitor transaction states:**
   ```python
   from morag.ingestion.state_manager import get_state_manager
   
   state_manager = get_state_manager()
   
   # Monitor progress
   state = await state_manager.load_state(transaction_id)
   print(f"Current state: {state.state}")
   ```

## Configuration

### Environment Variables

```bash
# State management
MORAG_INGESTION_STATE_DIR=/path/to/state/directory

# Transaction timeouts
MORAG_TRANSACTION_TIMEOUT=300  # 5 minutes

# Cleanup intervals
MORAG_STATE_CLEANUP_HOURS=168  # 7 days
MORAG_TRANSACTION_CLEANUP_HOURS=24  # 1 day
```

### Database Configuration

Ensure your database configurations support transactions:

```python
# Neo4j - ensure proper transaction support
neo4j_config = DatabaseConfig(
    type=DatabaseType.NEO4J,
    host="localhost",
    port=7687,
    username="neo4j",
    password="password",
    database="morag"
)

# Qdrant - ensure collection exists
qdrant_config = DatabaseConfig(
    type=DatabaseType.QDRANT,
    host="localhost",
    port=6333,
    database_name="morag_documents"
)
```

## Monitoring and Maintenance

### Regular Maintenance Tasks

1. **State Cleanup:**
   ```python
   # Run daily
   await state_manager.cleanup_old_states(max_age_hours=168)
   ```

2. **Transaction Cleanup:**
   ```python
   # Run hourly
   await coordinator.cleanup_old_transactions(max_age_hours=24)
   ```

3. **Failed State Review:**
   ```python
   # Run daily
   failed_states = await state_manager.get_failed_states(max_age_hours=24)
   for state in failed_states:
       # Analyze and potentially retry
       pass
   ```

### Metrics to Monitor

- Transaction success/failure rates
- Average validation time
- Average commit time
- Number of rollbacks
- Failed state accumulation
- JSON parsing error rates

## Troubleshooting

### Common Issues

1. **Validation Failures:**
   - Check LLM response quality
   - Review JSON parsing errors
   - Verify chunk size settings

2. **Transaction Timeouts:**
   - Increase timeout settings
   - Check database performance
   - Review chunk size and overlap

3. **Rollback Failures:**
   - Check database connectivity
   - Verify permissions
   - Review transaction logs

### Debug Mode

Enable detailed logging:

```python
import structlog
structlog.configure(level="DEBUG")
```

This will provide detailed logs for:
- JSON parsing attempts
- Transaction state changes
- Database operations
- Error recovery steps
