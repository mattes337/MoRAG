"""Search endpoints for MoRAG API."""

import structlog
from fastapi import APIRouter, HTTPException

from morag.api_models.models import SearchRequest

logger = structlog.get_logger(__name__)

search_router = APIRouter(tags=["Search"])


def setup_search_endpoints(morag_api_getter):
    """Setup search endpoints with MoRAG API getter function."""
    
    @search_router.post("/search")
    async def search_similar(request: SearchRequest):
        """Search for similar content."""
        try:
            # Handle custom database servers if provided
            if request.database_servers:
                from morag.database_factory import get_qdrant_storages
                
                # Convert database server configs to DatabaseConfig objects
                database_configs = []
                for server_config in request.database_servers:
                    from morag_graph.models.database_config import DatabaseConfig, DatabaseType
                    db_config = DatabaseConfig(
                        type=DatabaseType.QDRANT,
                        host=server_config.get('host', 'localhost'),
                        port=server_config.get('port', 6333),
                        collection_name=server_config.get('collection_name', 'default'),
                        api_key=server_config.get('api_key'),
                        url=server_config.get('url')
                    )
                    database_configs.append(db_config)
                
                # Get storages for the specified databases
                storages = get_qdrant_storages(database_configs)
                
                # Search across all specified databases
                all_results = []
                for storage in storages:
                    results = await storage.search_similar(
                        query=request.query,
                        limit=request.limit,
                        filters=request.filters
                    )
                    all_results.extend(results)
                
                # Sort by score and limit results
                all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                return all_results[:request.limit]
            else:
                # Use default database configuration
                results = await morag_api_getter().search_similar(
                    request.query,
                    request.limit,
                    request.filters
                )
                return results

        except Exception as e:
            logger.error("Search failed", query=request.query, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    return search_router
