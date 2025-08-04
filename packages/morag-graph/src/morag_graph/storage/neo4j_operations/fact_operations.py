"""Neo4j operations for fact storage and retrieval."""

from typing import List, Dict, Any, Optional, Tuple
import structlog
from neo4j import AsyncDriver

from ...models.fact import Fact, FactRelation
from .base_operations import BaseOperations


class FactOperations(BaseOperations):
    """Neo4j operations for fact management."""

    def __init__(self, driver: AsyncDriver, database: str):
        """Initialize with Neo4j driver and database.

        Args:
            driver: Neo4j async driver instance
            database: Database name
        """
        super().__init__(driver, database)
        self.logger = structlog.get_logger(__name__)
    
    async def store_fact(self, fact: Fact) -> str:
        """Store a fact in Neo4j.

        Args:
            fact: Fact to store

        Returns:
            Fact ID
        """
        query = """
        MERGE (f:Fact {id: $id})
        SET f += $properties
        RETURN f.id as fact_id
        """

        results = await self._execute_query(
            query,
            {
                "id": fact.id,
                "properties": fact.get_neo4j_properties()
            }
        )

        if results:
            self.logger.debug(
                "Fact stored successfully",
                fact_id=fact.id,
                subject=fact.subject[:50] + "..." if len(fact.subject) > 50 else fact.subject
            )
            return results[0]["fact_id"]

        raise RuntimeError(f"Failed to store fact {fact.id}")
    
    async def store_facts(self, facts: List[Fact]) -> List[str]:
        """Store multiple facts in Neo4j.

        Args:
            facts: List of facts to store

        Returns:
            List of fact IDs
        """
        if not facts:
            return []

        # Prepare batch data
        fact_data = [
            {
                'id': fact.id,
                'properties': fact.get_neo4j_properties()
            }
            for fact in facts
        ]

        query = """
        UNWIND $facts as fact_data
        MERGE (f:Fact {id: fact_data.id})
        SET f += fact_data.properties
        RETURN f.id as fact_id
        """

        results = await self._execute_query(query, {"facts": fact_data})
        fact_ids = [record["fact_id"] for record in results]

        self.logger.info(
            "Facts stored successfully",
            num_facts=len(facts),
            stored_ids=len(fact_ids)
        )

        return fact_ids
    
    async def store_fact_relation(self, relation: FactRelation) -> str:
        """Store a fact relationship in Neo4j.
        
        Args:
            relation: Fact relationship to store
            
        Returns:
            Relationship ID
        """
        async with self.driver.session() as session:
            query = """
            MATCH (source:Fact {id: $source_id})
            MATCH (target:Fact {id: $target_id})
            MERGE (source)-[r:FACT_RELATION {id: $rel_id}]->(target)
            SET r += $properties
            RETURN r.id as relation_id
            """
            
            result = await session.run(
                query,
                source_id=relation.source_fact_id,
                target_id=relation.target_fact_id,
                rel_id=relation.id,
                properties=relation.get_neo4j_properties()
            )
            
            record = await result.single()
            if record:
                self.logger.debug(
                    "Fact relation stored successfully",
                    relation_id=relation.id,
                    relation_type=relation.relation_type
                )
                return record["relation_id"]
            
            raise RuntimeError(f"Failed to store fact relation {relation.id}")
    
    async def store_fact_relations(self, relations: List[FactRelation]) -> List[str]:
        """Store multiple fact relationships in Neo4j.
        
        Args:
            relations: List of relationships to store
            
        Returns:
            List of relationship IDs
        """
        if not relations:
            return []
        
        async with self.driver.session() as session:
            # Prepare batch data
            relation_data = [
                {
                    'source_id': rel.source_fact_id,
                    'target_id': rel.target_fact_id,
                    'rel_id': rel.id,
                    'properties': rel.get_neo4j_properties()
                }
                for rel in relations
            ]
            
            query = """
            UNWIND $relations as rel_data
            MATCH (source:Fact {id: rel_data.source_id})
            MATCH (target:Fact {id: rel_data.target_id})
            MERGE (source)-[r:FACT_RELATION {id: rel_data.rel_id}]->(target)
            SET r += rel_data.properties
            RETURN r.id as relation_id
            """
            
            result = await session.run(query, relations=relation_data)
            relation_ids = [record["relation_id"] async for record in result]
            
            self.logger.info(
                "Fact relations stored successfully",
                num_relations=len(relations),
                stored_ids=len(relation_ids)
            )
            
            return relation_ids
    
    async def get_fact_by_id(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a fact by ID.
        
        Args:
            fact_id: Fact ID to retrieve
            
        Returns:
            Fact object or None if not found
        """
        async with self.driver.session() as session:
            query = """
            MATCH (f:Fact {id: $fact_id})
            RETURN f
            """
            
            result = await session.run(query, fact_id=fact_id)
            record = await result.single()
            
            if record:
                fact_data = dict(record["f"])
                return self._neo4j_to_fact(fact_data)
            
            return None
    
    async def get_facts_by_document(self, document_id: str) -> List[Fact]:
        """Get all facts from a specific document.
        
        Args:
            document_id: Document ID to search for
            
        Returns:
            List of facts from the document
        """
        async with self.driver.session() as session:
            query = """
            MATCH (f:Fact)
            WHERE f.source_document_id = $document_id
            RETURN f
            ORDER BY f.created_at
            """
            
            result = await session.run(query, document_id=document_id)
            facts = []
            
            async for record in result:
                fact_data = dict(record["f"])
                fact = self._neo4j_to_fact(fact_data)
                if fact:
                    facts.append(fact)
            
            return facts
    
    async def get_facts_by_domain(self, domain: str, limit: int = 100) -> List[Fact]:
        """Get facts by domain.
        
        Args:
            domain: Domain to search for
            limit: Maximum number of facts to return
            
        Returns:
            List of facts in the domain
        """
        async with self.driver.session() as session:
            query = """
            MATCH (f:Fact)
            WHERE f.domain = $domain
            RETURN f
            ORDER BY f.confidence DESC, f.created_at DESC
            LIMIT $limit
            """
            
            result = await session.run(query, domain=domain, limit=limit)
            facts = []
            
            async for record in result:
                fact_data = dict(record["f"])
                fact = self._neo4j_to_fact(fact_data)
                if fact:
                    facts.append(fact)
            
            return facts
    
    async def search_facts(
        self, 
        query_text: str, 
        fact_type: Optional[str] = None,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 50
    ) -> List[Fact]:
        """Search facts by text content.
        
        Args:
            query_text: Text to search for
            fact_type: Optional fact type filter
            domain: Optional domain filter
            min_confidence: Minimum confidence threshold
            limit: Maximum results to return
            
        Returns:
            List of matching facts
        """
        async with self.driver.session() as session:
            # Build dynamic query
            where_clauses = ["f.confidence >= $min_confidence"]
            params = {
                'query_text': query_text.lower(),
                'min_confidence': min_confidence,
                'limit': limit
            }
            
            if fact_type:
                where_clauses.append("f.fact_type = $fact_type")
                params['fact_type'] = fact_type
            
            if domain:
                where_clauses.append("f.domain = $domain")
                params['domain'] = domain
            
            where_clause = " AND ".join(where_clauses)
            
            query = f"""
            MATCH (f:Fact)
            WHERE {where_clause}
            AND (
                toLower(f.subject) CONTAINS $query_text
                OR toLower(f.object) CONTAINS $query_text
                OR toLower(f.approach) CONTAINS $query_text
                OR toLower(f.solution) CONTAINS $query_text
                OR toLower(f.keywords) CONTAINS $query_text
            )
            RETURN f
            ORDER BY f.confidence DESC, f.created_at DESC
            LIMIT $limit
            """
            
            result = await session.run(query, **params)
            facts = []
            
            async for record in result:
                fact_data = dict(record["f"])
                fact = self._neo4j_to_fact(fact_data)
                if fact:
                    facts.append(fact)
            
            return facts
    
    async def get_related_facts(
        self, 
        fact_id: str, 
        relation_types: Optional[List[str]] = None,
        max_depth: int = 2
    ) -> List[Tuple[Fact, str, float]]:
        """Get facts related to a given fact.
        
        Args:
            fact_id: Source fact ID
            relation_types: Optional list of relation types to follow
            max_depth: Maximum relationship depth
            
        Returns:
            List of tuples (related_fact, relation_type, confidence)
        """
        async with self.driver.session() as session:
            # Build relation type filter
            relation_filter = ""
            params = {'fact_id': fact_id, 'max_depth': max_depth}
            
            if relation_types:
                relation_filter = "AND r.relation_type IN $relation_types"
                params['relation_types'] = relation_types
            
            query = f"""
            MATCH path = (source:Fact {{id: $fact_id}})-[r:FACT_RELATION*1..$max_depth]-(target:Fact)
            WHERE source <> target {relation_filter}
            RETURN DISTINCT target as fact, 
                   [rel in relationships(path) | rel.relation_type] as relation_path,
                   [rel in relationships(path) | rel.confidence] as confidence_path
            ORDER BY length(path), target.confidence DESC
            """
            
            result = await session.run(query, **params)
            related_facts = []
            
            async for record in result:
                fact_data = dict(record["fact"])
                fact = self._neo4j_to_fact(fact_data)
                
                if fact:
                    relation_path = record["relation_path"]
                    confidence_path = record["confidence_path"]
                    
                    # Use first relation type and minimum confidence in path
                    relation_type = relation_path[0] if relation_path else "UNKNOWN"
                    min_confidence = min(confidence_path) if confidence_path else 0.0
                    
                    related_facts.append((fact, relation_type, min_confidence))
            
            return related_facts
    
    async def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact and its relationships.
        
        Args:
            fact_id: Fact ID to delete
            
        Returns:
            True if deleted successfully
        """
        async with self.driver.session() as session:
            query = """
            MATCH (f:Fact {id: $fact_id})
            DETACH DELETE f
            RETURN count(f) as deleted_count
            """
            
            result = await session.run(query, fact_id=fact_id)
            record = await result.single()
            
            if record and record["deleted_count"] > 0:
                self.logger.info("Fact deleted successfully", fact_id=fact_id)
                return True
            
            return False
    
    async def get_fact_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored facts.
        
        Returns:
            Dictionary with fact statistics
        """
        async with self.driver.session() as session:
            query = """
            MATCH (f:Fact)
            OPTIONAL MATCH (f)-[r:FACT_RELATION]-()
            RETURN 
                count(DISTINCT f) as total_facts,
                count(DISTINCT r) as total_relations,
                collect(DISTINCT f.fact_type) as fact_types,
                collect(DISTINCT f.domain) as domains,
                avg(f.confidence) as avg_confidence,
                min(f.confidence) as min_confidence,
                max(f.confidence) as max_confidence
            """
            
            result = await session.run(query)
            record = await result.single()
            
            if record:
                return {
                    'total_facts': record['total_facts'],
                    'total_relations': record['total_relations'],
                    'fact_types': [ft for ft in record['fact_types'] if ft],
                    'domains': [d for d in record['domains'] if d],
                    'avg_confidence': record['avg_confidence'],
                    'min_confidence': record['min_confidence'],
                    'max_confidence': record['max_confidence']
                }
            
            return {}
    
    def _neo4j_to_fact(self, fact_data: Dict[str, Any]) -> Optional[Fact]:
        """Convert Neo4j node data to Fact object.
        
        Args:
            fact_data: Neo4j node properties
            
        Returns:
            Fact object or None if conversion fails
        """
        try:
            # Handle keywords conversion
            keywords_str = fact_data.get('keywords', '')
            keywords = [k.strip() for k in keywords_str.split(',') if k.strip()] if keywords_str else []
            
            # Handle datetime conversion
            from datetime import datetime
            created_at_str = fact_data.get('created_at')
            created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.utcnow()
            
            return Fact(
                id=fact_data['id'],
                subject=fact_data['subject'],
                object=fact_data['object'],
                approach=fact_data.get('approach'),
                solution=fact_data.get('solution'),
                remarks=fact_data.get('remarks'),
                source_chunk_id=fact_data['source_chunk_id'],
                source_document_id=fact_data['source_document_id'],
                extraction_confidence=float(fact_data.get('confidence', 0.0)),
                fact_type=fact_data.get('fact_type', 'definition'),
                domain=fact_data.get('domain'),
                keywords=keywords,
                created_at=created_at,
                language=fact_data.get('language', 'en')
            )
            
        except Exception as e:
            self.logger.warning(
                "Failed to convert Neo4j data to Fact",
                error=str(e),
                fact_data=fact_data
            )
            return None
