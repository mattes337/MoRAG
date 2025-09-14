# Intelligent Retrieval with Custom Database Connections

The intelligent retrieval REST API now supports custom database server connections, allowing you to connect to Neo4j and Qdrant servers other than those configured in the MoRAG server environment.

## Overview

The `/api/v2/intelligent-query` endpoint supports three ways to specify database connections:

1. **Custom Server Configurations** - Provide full connection details for any Neo4j/Qdrant server
2. **Named Database Configurations** - Use pre-configured databases by name
3. **Server Default Configurations** - Use the server's default environment settings

## Custom Database Server Configurations

### Neo4j Server Configuration

```json
{
  "neo4j_server": {
    "type": "neo4j",
    "hostname": "neo4j://your-server.com:7687",
    "username": "your-username",
    "password": "your-password",
    "database_name": "your-database",
    "config_options": {
      "verify_ssl": true,
      "trust_all_certificates": false
    }
  }
}
```

### Qdrant Server Configuration

```json
{
  "qdrant_server": {
    "type": "qdrant",
    "hostname": "your-qdrant-server.com",
    "port": 6333,
    "password": "your-api-key",
    "database_name": "your-collection",
    "config_options": {
      "https": true,
      "verify_ssl": true
    }
  }
}
```

## Configuration Parameters

### Common Parameters

- `type`: Database type (`"neo4j"` or `"qdrant"`)
- `hostname`: Server hostname or full URI
- `port`: Server port (optional if included in hostname)
- `username`: Database username (Neo4j only)
- `password`: Database password or API key
- `database_name`: Database name (Neo4j) or collection name (Qdrant)
- `config_options`: Additional database-specific options

### Neo4j Specific Options

- `verify_ssl`: Whether to verify SSL certificates (default: true)
- `trust_all_certificates`: Trust all certificates including self-signed (default: false)

### Qdrant Specific Options

- `https`: Use HTTPS connection (default: false)
- `verify_ssl`: Whether to verify SSL certificates (default: true)

## Usage Examples

### Example 1: Custom External Servers

```json
{
  "query": "What are the main symptoms of ADHD?",
  "max_iterations": 5,
  "neo4j_server": {
    "type": "neo4j",
    "hostname": "https://graph.example.com",
    "username": "neo4j",
    "password": "secure-password",
    "database_name": "production",
    "config_options": {
      "verify_ssl": true
    }
  },
  "qdrant_server": {
    "type": "qdrant",
    "hostname": "vectors.example.com",
    "port": 6333,
    "password": "api-key-123",
    "database_name": "documents",
    "config_options": {
      "https": true,
      "verify_ssl": true
    }
  }
}
```

### Example 2: Named Database Configurations

```json
{
  "query": "What are the main symptoms of ADHD?",
  "max_iterations": 5,
  "neo4j_database": "production",
  "qdrant_collection": "documents_v2"
}
```

### Example 3: Server Default Configurations

```json
{
  "query": "What are the main symptoms of ADHD?",
  "max_iterations": 5
}
```

## Priority Order

When multiple database configuration methods are provided, they are processed in this priority order:

1. **Custom server configurations** (`neo4j_server`, `qdrant_server`) - highest priority
2. **Named configurations** (`neo4j_database`, `qdrant_collection`) - medium priority  
3. **Server defaults** - lowest priority (fallback)

## Error Handling

The API will return appropriate HTTP error codes for configuration issues:

- `400 Bad Request`: Invalid configuration parameters or connection failures
- `503 Service Unavailable`: Database storage not available

Example error responses:

```json
{
  "detail": "Invalid database type for Neo4j server: qdrant"
}
```

```json
{
  "detail": "Failed to create Neo4j connection: Connection refused"
}
```

## Security Considerations

- **Credentials**: Database passwords and API keys are transmitted in the request body. Use HTTPS in production.
- **SSL Verification**: Always enable SSL verification (`verify_ssl: true`) for production environments.
- **Network Access**: Ensure the MoRAG server has network access to your custom database servers.

## Testing

Use the provided test script to validate custom database configurations:

```bash
# Show example configurations
python cli/test-custom-database-retrieval.py examples

# Test REST API with custom configuration
python cli/test-custom-database-retrieval.py test-api
```

## Migration from Previous Versions

Previous versions only supported named database configurations. The new custom server configurations are fully backward compatible:

**Old way (still supported):**
```json
{
  "neo4j_database": "my-database",
  "qdrant_collection": "my-collection"
}
```

**New way (additional option):**
```json
{
  "neo4j_server": {
    "type": "neo4j",
    "hostname": "neo4j://external-server:7687",
    "username": "user",
    "password": "pass",
    "database_name": "my-database"
  }
}
```

## Limitations

- Custom server configurations create new connections for each request. For high-frequency usage, consider using named configurations with pre-configured connection pools.
- Connection timeouts and retry logic use the same settings as the server's default configurations.
- Database schema compatibility is assumed - ensure your custom databases have the same schema as expected by MoRAG.
