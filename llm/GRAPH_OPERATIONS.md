# Neo4j Graph Operations

## Entity Operations

### Entity Storage with Deduplication
**Strategy**: Use MERGE operations for entity deduplication based on normalized names.

**Deduplication Logic**:
- **Primary Key**: Normalized name for entity matching
- **Creation**: Set all properties including timestamps when creating new entities
- **Matching**: Update properties only when new confidence is higher than existing
- **Conditional Updates**: Preserve higher-confidence entity names and types
- **Metadata Refresh**: Always update metadata and timestamps on matches
- **Return Values**: Provide entity ID, final type, and final name for confirmation

### Dynamic Entity Labels
**Conversion Process**: Transform entity types to Neo4j-compatible labels using title case formatting.

**Transformation Rules**:
- **Prefix Removal**: Remove "EntityType_" prefixes from type names
- **Underscore Handling**: Replace underscores with spaces for processing
- **Title Case**: Convert to title case for proper Neo4j label format
- **Space Removal**: Remove spaces to create valid Neo4j labels

**Examples**:
- "PERSON" → "Person"
- "ORGANIZATION" → "Organization"
- "MEDICAL_CONDITION" → "MedicalCondition"

### Entity Retrieval Patterns
**Retrieval Methods**:

**By ID**: Direct entity lookup using unique identifier for precise retrieval.

**By Type**: Filter entities by specific type labels with confidence thresholds, ordered by confidence score.

**By Name Pattern**: Search entities using name pattern matching with CONTAINS operator, returning entities and their labels ordered by confidence.

## Relationship Operations

### Dynamic Relationship Creation
**Process**: Create relationships between entities with dynamic types and metadata.

**Creation Steps**:
1. **Entity Matching**: Find source and target entities by ID
2. **Relationship Merging**: Use MERGE to avoid duplicate relationships
3. **Property Setting**: Set relationship type, confidence, and metadata
4. **Timestamp Management**: Set creation time on first creation, always update modification time
5. **ID Return**: Return relationship ID for reference

### Relationship Type Normalization
**Normalization Process**: Convert relationship types to consistent Neo4j format.

**Normalization Rules**:
- **Case Conversion**: Convert to uppercase for consistency
- **Character Replacement**: Replace spaces and hyphens with underscores
- **Prefix Removal**: Remove "RELATIONTYPE_" prefixes
- **Singular Form**: Ensure singular form for consistency (TREATS, CAUSES, PREVENTS, etc.)
- **Mapping Validation**: Use predefined mappings for common relationship types

### Relationship Queries
**Query Types**:

**All Relationships**: Get all relationships for an entity with connected entities and relationship types, ordered by confidence.

**Specific Types**: Filter relationships by specific types (TREATS, CAUSES, PREVENTS) with connected entities, ordered by confidence.

**Relationship Paths**: Find paths between specific entities within hop limits (1-3), returning path and length information ordered by path length.

## Graph Traversal Patterns

### Neighbor Discovery
**Discovery Methods**:

**Direct Neighbors**: Find entities directly connected to a starting entity, returning neighbors with relationship information, ordered by name.

**Distance-Based**: Find neighbors within specified distance (1-3 hops), returning distinct neighbors with distance information, ordered by distance and name.

### Path Finding
**Path Types**:

**Shortest Path**: Find shortest path between two entities within hop limits (1-5), returning complete path information including entities and relationships, ordered by path length.

**All Paths**: Find all paths between entities within distance limits (1-3 hops), returning path IDs and lengths, ordered by path length with result limiting (top 10).

### Breadth-First Exploration
**BFS Configuration**: Use APOC procedures for advanced breadth-first traversal with filtering capabilities.

**Configuration Options**:
- **Relationship Filtering**: Specify allowed relationship types (TREATS, CAUSES, PREVENTS)
- **Label Filtering**: Restrict to specific node labels (+Entity)
- **Level Control**: Set minimum (1) and maximum (3) traversal levels
- **Traversal Mode**: Use breadth-first search for systematic exploration
- **Uniqueness**: Ensure global node uniqueness to avoid cycles

## Fact Operations

### Fact Storage
**Storage Process**: Create fact nodes with comprehensive properties and establish relationships to source chunks and mentioned entities.

**Fact Properties**:
- **Core Information**: ID, content, subject, predicate, object
- **Quality Metrics**: Confidence score, fact type, domain classification
- **Metadata**: Creation timestamp for tracking

**Relationship Creation**:
- **Source Linking**: Connect facts to source document chunks via CONTAINS_FACT relationships
- **Entity Linking**: Connect facts to mentioned entities via MENTIONS relationships

### Fact Retrieval
**Retrieval Methods**:

**By Entity**: Find all facts that mention a specific entity, ordered by confidence score.

**By Domain**: Filter facts by domain and minimum confidence threshold, ordered by confidence.

**With Source Attribution**: Retrieve facts for an entity along with source chunk information, ordered by confidence for complete traceability.

## Document and Chunk Operations

### Document Storage
**Storage Components**:

**Document Node**: Store document with comprehensive metadata including ID, title, content type, file path, and creation timestamp.

**Document Chunk**: Create chunk nodes with document reference, chunk index, text content, position information, and metadata.

**Relationship Linking**: Establish HAS_CHUNK relationships between documents and their chunks for hierarchical organization.

### Chunk Retrieval
**Retrieval Methods**:

**By Document**: Get all chunks for a specific document, ordered by chunk index for sequential reading.

**By Entity Mentions**: Find chunks that contain facts mentioning specific entities, returning chunks with associated facts and entities, ordered by chunk index.

## Performance Optimization

### Index Creation
**Index Categories**:

**Entity Indexes**: Create indexes on entity ID, name, normalized_name, and type for fast entity lookups and searches.

**Fact Indexes**: Index fact ID, domain, and confidence for efficient fact filtering and retrieval.

**Document Indexes**: Index document ID, chunk ID, and chunk document_id for fast document and chunk operations.

### Query Optimization
**Optimization Tools**:

**EXPLAIN**: Analyze query execution plans to understand performance characteristics and identify bottlenecks.

**PROFILE**: Get detailed execution statistics including actual row counts, database hits, and timing information for performance tuning.

### Batch Operations
**Batch Processing**:

**Entity Creation**: Use UNWIND for batch entity creation with MERGE operations, setting properties and timestamps appropriately for creation and updates.

**Relationship Creation**: Batch create relationships by unwinding relationship data, matching source and target entities, and merging relationships with properties.

## Graph Analytics

### Entity Statistics
**Statistical Analysis**:

**Entity Count by Type**: Count entities grouped by their primary label/type, ordered by frequency for understanding data distribution.

**Confidence Distribution**: Categorize entities by confidence ranges (High 0.9+, Medium 0.7-0.9, Low 0.5-0.7, Very Low <0.5) to assess data quality.

### Relationship Statistics
**Relationship Analysis**:

**Count by Type**: Count relationships grouped by type, ordered by frequency to understand relationship patterns.

**Most Connected Entities**: Find entities with highest connection counts (top 10) to identify central nodes in the graph.

### Graph Connectivity
**Connectivity Analysis**:

**Connected Components**: Use Graph Data Science library to identify connected components and their sizes for understanding graph structure.

**PageRank Analysis**: Calculate PageRank scores to identify most important entities based on their position in the graph network (top 10).

## Error Handling

### Constraint Violations
**Error Handling Strategies**:

**Duplicate Entity Creation**: Use MERGE operations with ON CREATE and ON MATCH clauses to handle duplicate entity creation gracefully.

**Missing Entity References**: Validate entity existence before creating relationships using WHERE clauses to ensure both source and target entities exist.

### Transaction Management
**Transaction Handling**: Execute multiple graph operations within a single transaction for consistency and atomicity.

**Process**:
1. **Session Management**: Create Neo4j session for database connection
2. **Transaction Begin**: Start transaction for atomic operations
3. **Operation Execution**: Execute multiple operations within transaction scope
4. **Result Collection**: Gather results from all operations
5. **Commit/Rollback**: Commit on success or rollback on failure with error logging
