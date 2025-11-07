"""Base operations class for Neo4j storage operations."""

import logging
from typing import Any, Dict, List, Optional

from neo4j import AsyncDriver, AsyncSession

logger = logging.getLogger(__name__)


class BaseOperations:
    """Base class for Neo4j storage operations."""

    def __init__(self, driver: AsyncDriver, database: str):
        """Initialize base operations.

        Args:
            driver: Neo4j async driver
            database: Database name
        """
        self.driver = driver
        self.database = database

    async def _execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query.

        Args:
            query: Cypher query to execute
            parameters: Query parameters

        Returns:
            List of result records as dictionaries
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, parameters or {})
            records = []
            async for record in result:
                records.append(record.data())
            return records
