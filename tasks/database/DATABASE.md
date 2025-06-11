## Entities

### User
The central entity representing system users.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | String | PRIMARY KEY, UUID | Unique identifier for the user |
| `name` | String | NOT NULL | User's display name |
| `email` | String | UNIQUE, NOT NULL | User's email address (used for authentication) |
| `avatar` | String | NULLABLE | URL or path to user's avatar image |
| `role` | UserRole | DEFAULT: USER | User's permission level (ADMIN, USER, VIEWER) |
| `createdAt` | DateTime | DEFAULT: now() | Timestamp when user was created |
| `updatedAt` | DateTime | AUTO UPDATE | Timestamp when user was last updated |

**Relationships:**
- One-to-many with Database
- One-to-many with Document
- One-to-many with ApiKey
- One-to-many with Job
- One-to-many with DatabaseServer
- One-to-one with UserSettings

### UserSettings
Stores user preferences and configuration.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | String | PRIMARY KEY, UUID | Unique identifier for settings |
| `userId` | String | FOREIGN KEY, UNIQUE | Reference to User |
| `theme` | Theme | DEFAULT: LIGHT | UI theme preference (LIGHT, DARK, SYSTEM) |
| `language` | String | DEFAULT: "en" | Preferred language code |
| `notifications` | Boolean | DEFAULT: true | Whether to show notifications |
| `autoSave` | Boolean | DEFAULT: true | Whether to auto-save changes |
| `defaultDatabase` | String | NULLABLE | Default database ID for new documents |
| `createdAt` | DateTime | DEFAULT: now() | Timestamp when settings were created |
| `updatedAt` | DateTime | AUTO UPDATE | Timestamp when settings were last updated |

### Database
Represents a vector database instance where documents are stored.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | String | PRIMARY KEY, UUID | Unique identifier for the database |
| `name` | String | NOT NULL | Human-readable database name |
| `description` | String | NOT NULL | Database description or purpose |
| `documentCount` | Int | DEFAULT: 0 | Cached count of documents in this database |
| `lastUpdated` | DateTime | DEFAULT: now() | When database was last modified |
| `createdAt` | DateTime | DEFAULT: now() | Timestamp when database was created |
| `updatedAt` | DateTime | AUTO UPDATE | Timestamp when database was last updated |
| `userId` | String | FOREIGN KEY | Owner of the database |
| `serverId` | String | FOREIGN KEY | Database server hosting this database |

**Constraints:**
- Unique constraint on (name, userId) - users cannot have duplicate database names

### Document
Represents uploaded documents and their processing state.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | String | PRIMARY KEY, UUID | Unique identifier for the document |
| `name` | String | NOT NULL | Original filename or document name |
| `type` | String | NOT NULL | Document type/format (pdf, docx, txt, etc.) |
| `state` | DocumentState | DEFAULT: PENDING | Processing state of the document |
| `version` | Int | DEFAULT: 1 | Document version number |
| `chunks` | Int | DEFAULT: 0 | Number of text chunks created from document |
| `quality` | Float | DEFAULT: 0.0 | Processing quality score (0.0-1.0) |
| `uploadDate` | DateTime | DEFAULT: now() | When document was uploaded |
| `createdAt` | DateTime | DEFAULT: now() | Timestamp when record was created |
| `updatedAt` | DateTime | AUTO UPDATE | Timestamp when record was last updated |
| `userId` | String | FOREIGN KEY | User who uploaded the document |
| `databaseId` | String | FOREIGN KEY, NULLABLE | Database where document is stored |

### ApiKey
Manages API keys for external access to the system.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | String | PRIMARY KEY, UUID | Unique identifier for the API key |
| `name` | String | NOT NULL | Human-readable name for the key |
| `key` | String | UNIQUE, NOT NULL | The actual API key value |
| `created` | DateTime | DEFAULT: now() | When the key was created |
| `lastUsed` | DateTime | NULLABLE | When the key was last used |
| `createdAt` | DateTime | DEFAULT: now() | Timestamp when record was created |
| `updatedAt` | DateTime | AUTO UPDATE | Timestamp when record was last updated |
| `userId` | String | FOREIGN KEY | Owner of the API key |

### DatabaseServer
Configuration for external database servers (vector databases).

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | String | PRIMARY KEY, UUID | Unique identifier for the server |
| `name` | String | NOT NULL | Human-readable server name |
| `type` | DatabaseType | NOT NULL | Type of database server |
| `host` | String | NOT NULL | Server hostname or IP address |
| `port` | Int | NOT NULL | Server port number |
| `username` | String | NULLABLE | Authentication username |
| `password` | String | NULLABLE | Authentication password |
| `apiKey` | String | NULLABLE | API key for authentication |
| `database` | String | NULLABLE | Default database name |
| `collection` | String | NULLABLE | Default collection name |
| `isActive` | Boolean | DEFAULT: false | Whether server is currently active |
| `createdAt` | DateTime | DEFAULT: now() | Timestamp when server was added |
| `lastConnected` | DateTime | NULLABLE | When server was last successfully connected |
| `updatedAt` | DateTime | AUTO UPDATE | Timestamp when server config was updated |
| `userId` | String | FOREIGN KEY | User who configured this server |

### Job
Tracks document processing jobs and their status.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | String | PRIMARY KEY, UUID | Unique identifier for the job |
| `documentName` | String | NOT NULL | Name of document being processed |
| `documentType` | String | NOT NULL | Type of document being processed |
| `startDate` | DateTime | DEFAULT: now() | When job started |
| `endDate` | DateTime | NULLABLE | When job completed |
| `status` | JobStatus | DEFAULT: PENDING | Current job status |
| `percentage` | Int | DEFAULT: 0 | Completion percentage (0-100) |
| `summary` | String | DEFAULT: "" | Job summary or error message |
| `createdAt` | DateTime | DEFAULT: now() | Timestamp when job was created |
| `updatedAt` | DateTime | AUTO UPDATE | Timestamp when job was last updated |
| `documentId` | String | FOREIGN KEY | Document being processed |
| `userId` | String | FOREIGN KEY | User who owns the document |

## Relationships

### User Relationships
- **User → UserSettings**: One-to-one relationship. Each user has exactly one settings record.
- **User → Database**: One-to-many relationship. Users can create multiple databases.
- **User → Document**: One-to-many relationship. Users can upload multiple documents.
- **User → ApiKey**: One-to-many relationship. Users can create multiple API keys.
- **User → Job**: One-to-many relationship. Users can have multiple processing jobs.
- **User → DatabaseServer**: One-to-many relationship. Users can configure multiple database servers.

### Database Relationships
- **Database → User**: Many-to-one relationship. Each database belongs to one user.
- **Database → DatabaseServer**: Many-to-one relationship. Each database is hosted on one server.
- **Database → Document**: One-to-many relationship. Databases can contain multiple documents.

### Document Relationships
- **Document → User**: Many-to-one relationship. Each document belongs to one user.
- **Document → Database**: Many-to-one relationship (nullable). Documents can be assigned to a database.
- **Document → Job**: One-to-many relationship. Documents can have multiple processing jobs.

### Job Relationships
- **Job → Document**: Many-to-one relationship. Each job processes one document.
- **Job → User**: Many-to-one relationship. Each job belongs to one user.

## Enums

### UserRole
Defines user permission levels:
- `ADMIN`: Full system access, can manage all users and resources
- `USER`: Standard user with access to own resources
- `VIEWER`: Read-only access to assigned resources

### Theme
UI theme preferences:
- `LIGHT`: Light theme
- `DARK`: Dark theme
- `SYSTEM`: Follow system theme preference

### DocumentState
Document processing lifecycle states:
- `PENDING`: Document uploaded but not yet processed
- `INGESTING`: Document is currently being processed
- `INGESTED`: Document successfully processed and stored
- `DEPRECATED`: Document marked as outdated
- `DELETED`: Document marked for deletion

### DatabaseType
Supported vector database types:
- `QDRANT`: Qdrant vector database
- `NEO4J`: Neo4j graph database
- `PINECONE`: Pinecone vector database
- `WEAVIATE`: Weaviate vector database
- `CHROMA`: ChromaDB vector database

### JobStatus
Job processing states:
- `PENDING`: Job created but not started
- `WAITING_FOR_REMOTE_WORKER`: Waiting for processing resources
- `PROCESSING`: Job is currently running
- `FINISHED`: Job completed successfully
- `FAILED`: Job failed with errors
- `CANCELLED`: Job was cancelled by user

## Indexes and Constraints

### Primary Keys
All entities use UUID-based primary keys for better distribution and security.

### Unique Constraints
- `User.email`: Ensures unique user identification
- `ApiKey.key`: Prevents duplicate API keys
- `Database(name, userId)`: Users cannot have databases with duplicate names
- `UserSettings.userId`: One settings record per user

### Foreign Key Constraints
All relationships use CASCADE or SET NULL deletion policies:
- User deletion cascades to all owned resources
- Database deletion sets documents' databaseId to NULL
- Document deletion cascades to associated jobs

### Indexes
The schema relies on Prisma's automatic index generation for:
- Foreign key relationships
- Unique constraints
- Primary keys

## Business Logic

### Document Count Management
The `Database.documentCount` field is automatically maintained through:
- Increment on document creation
- Decrement on document deletion
- Manual recalculation via `DatabaseService.updateDocumentCount()`

### Job Lifecycle
Jobs track document processing with automatic timestamp management:
- `startDate` set on creation
- `endDate` set when status changes to FINISHED, FAILED, or CANCELLED
- `percentage` updated during processing

### User Settings
Default settings are created automatically for new users with sensible defaults:
- Light theme
- English language
- Notifications enabled
- Auto-save enabled

### Database Server Management
Servers track connection health:
- `isActive` indicates current availability
- `lastConnected` updated on successful connections
- Connection failures logged for troubleshooting

### Security Considerations
- API keys are stored as unique strings (should be hashed in production)
- Database passwords stored in plain text (should be encrypted in production)
- User roles control access to resources
- Foreign key constraints ensure data integrity

### Performance Considerations
- Document count is cached to avoid expensive COUNT queries
- Jobs are ordered by creation date for efficient pagination
- Indexes on foreign keys ensure fast relationship queries
- Soft deletion pattern used for documents (DELETED state)