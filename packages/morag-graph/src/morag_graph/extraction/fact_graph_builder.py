"""Build knowledge graphs from extracted facts."""

import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import structlog

from morag_reasoning.llm import LLMClient, LLMConfig
from ..models.fact import Fact, FactRelation, FactRelationType
from ..models.graph import Graph
# Removed obsolete fact_prompts - now using agents framework


class FactGraphBuilder:
    """Build knowledge graph from extracted facts."""
    
    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        min_relation_confidence: float = 0.6,
        max_relations_per_fact: int = 5,
        language: str = "en"
    ):
        """Initialize fact graph builder.

        Args:
            model_id: LLM model for relationship extraction
            api_key: API key for LLM service
            min_relation_confidence: Minimum confidence for relationships
            max_relations_per_fact: Maximum relationships per fact
            language: Language for relationship extraction
        """
        self.model_id = model_id
        self.api_key = api_key
        self.min_relation_confidence = min_relation_confidence
        self.max_relations_per_fact = max_relations_per_fact
        self.language = language
        
        self.logger = structlog.get_logger(__name__)
        
        # Initialize LLM client
        llm_config = LLMConfig(
            provider="gemini",
            model=model_id,
            api_key=api_key,
            temperature=0.1,
            max_tokens=2000
        )
        self.llm_client = LLMClient(llm_config)
    
    async def build_fact_graph(self, facts: List[Fact]) -> Graph:
        """Build knowledge graph from extracted facts.
        
        Args:
            facts: List of facts to build graph from
            
        Returns:
            Graph object containing facts and their relationships
        """
        if not facts:
            return Graph(nodes=[], edges=[])
        
        self.logger.info(
            "Starting fact graph building",
            num_facts=len(facts)
        )
        
        try:
            # Create fact relationships
            relationships = await self._create_fact_relationships(facts)
            
            # Build graph structure
            graph = self._build_graph_structure(facts, relationships)
            
            # Index facts for efficient retrieval
            await self._index_facts(facts)
            
            self.logger.info(
                "Fact graph building completed",
                num_facts=len(facts),
                num_relationships=len(relationships)
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(
                "Fact graph building failed",
                error=str(e),
                error_type=type(e).__name__
            )
            # Return empty graph on failure
            return Graph(nodes=[], edges=[])
    
    async def _create_fact_relationships(self, facts: List[Fact]) -> List[FactRelation]:
        """Create semantic relationships between facts.
        
        Args:
            facts: List of facts to analyze for relationships
            
        Returns:
            List of FactRelation objects
        """
        if len(facts) < 2:
            return []
        
        relationships = []
        failed_batches = 0
        max_failed_batches = 3  # Allow up to 3 failed batches before giving up

        # Process facts in batches to avoid overwhelming the LLM
        batch_size = 10
        total_batches = (len(facts) + batch_size - 1) // batch_size

        for i in range(0, len(facts), batch_size):
            batch = facts[i:i + batch_size]
            batch_num = i // batch_size + 1

            self.logger.debug(f"Processing relationship batch {batch_num}/{total_batches}")

            batch_relationships = await self._extract_relationships_for_batch(batch)

            if not batch_relationships:
                failed_batches += 1
                self.logger.warning(
                    f"Batch {batch_num} failed to extract relationships ({failed_batches}/{max_failed_batches} failures)"
                )

                # If too many batches fail, stop processing
                if failed_batches >= max_failed_batches:
                    self.logger.error(
                        f"Too many failed batches ({failed_batches}), stopping relationship extraction",
                        processed_batches=batch_num,
                        total_batches=total_batches
                    )
                    break
            else:
                relationships.extend(batch_relationships)
                # Reset failure counter on success
                failed_batches = 0
        
        # Filter relationships by confidence
        filtered_relationships = [
            rel for rel in relationships 
            if rel.confidence >= self.min_relation_confidence
        ]
        
        self.logger.debug(
            "Fact relationships created",
            total_relationships=len(relationships),
            filtered_relationships=len(filtered_relationships)
        )
        
        return filtered_relationships
    
    async def _extract_relationships_for_batch(self, facts: List[Fact]) -> List[FactRelation]:
        """Extract relationships for a batch of facts with retry logic.

        Args:
            facts: Batch of facts to analyze

        Returns:
            List of relationships found in the batch
        """
        if len(facts) < 2:
            return []

        max_retries = 3
        retry_delay = 1.0  # seconds

        # Initialize variables for exception handling
        response = None
        relationship_data = None

        for attempt in range(max_retries):
            try:
                # Convert facts to dictionaries for LLM prompt
                fact_dicts = [self._fact_to_prompt_dict(fact) for fact in facts]

                self.logger.debug(
                    f"Extracting relationships for {len(facts)} facts (attempt {attempt + 1}/{max_retries})",
                    fact_ids=[f.id for f in facts]
                )

                # Create relationship extraction prompt using agents framework
                prompt = self._create_relationship_extraction_prompt(fact_dicts, self.language)

                self.logger.debug(f"Sending relationship extraction prompt to LLM", prompt_length=len(prompt), prompt_preview=prompt[:300])

                # Get relationships from LLM
                try:
                    response = await self.llm_client.generate(prompt)
                    if not response:
                        self.logger.warning(f"LLM returned empty response for relationship extraction", attempt=attempt + 1)
                        response = ""
                except Exception as llm_error:
                    self.logger.error(f"LLM call failed for relationship extraction", error=str(llm_error), attempt=attempt + 1)
                    response = ""

                self.logger.debug(
                    "LLM relationship response received",
                    response_length=len(response) if response else 0,
                    response_preview=response[:200] if response else "Empty response",
                    attempt=attempt + 1
                )

                # Handle empty response
                if not response or len(response.strip()) == 0:
                    self.logger.warning(f"LLM returned empty or whitespace-only response for relationship extraction", attempt=attempt + 1)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        self.logger.info("No relationships found - LLM consistently returned empty responses")
                        return []

                # Parse response
                relationship_data = self._parse_relationship_response(response)

                # Check if LLM returned empty array (no relationships found)
                if relationship_data is not None and len(relationship_data) == 0:
                    self.logger.info(
                        f"LLM returned empty relationship array on attempt {attempt + 1} - no relationships found between facts",
                        response_preview=response[:300] if response else "Empty response"
                    )
                    return []  # This is a valid response, not an error

                if not relationship_data:
                    self.logger.warning(
                        f"No valid relationship data parsed on attempt {attempt + 1}",
                        response_preview=response[:300] if response else "Empty response",
                        response_length=len(response) if response else 0
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        self.logger.info("No relationships extracted after all retry attempts - this may be normal if facts are unrelated")
                        return []

                self.logger.debug(
                    f"Parsed {len(relationship_data)} relationship candidates",
                    relationship_data=relationship_data,
                    attempt=attempt + 1
                )

                # Convert to FactRelation objects
                relationships = self._create_fact_relation_objects(relationship_data, facts)

                self.logger.info(
                    f"Successfully created {len(relationships)} fact relationships from {len(relationship_data)} candidates on attempt {attempt + 1}"
                )

                return relationships

            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"JSON parsing failed in relationship extraction (attempt {attempt + 1}/{max_retries})",
                    batch_size=len(facts),
                    json_error=str(e),
                    error_position=getattr(e, 'pos', 'unknown')
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    return []

            except KeyError as e:
                # Capture the response and parsed data for debugging
                response_preview = response[:500] if response else "No response available"
                parsed_preview = str(relationship_data)[:200] if relationship_data is not None else "No parsed data"

                # Additional debugging information
                response_length = len(response) if response else 0
                parsed_data_type = type(relationship_data).__name__ if relationship_data is not None else "None"
                parsed_data_length = len(relationship_data) if relationship_data else 0

                self.logger.warning(
                    f"KeyError in relationship extraction (attempt {attempt + 1}/{max_retries})",
                    batch_size=len(facts),
                    key_error=str(e),
                    error_type=type(e).__name__,
                    llm_response_preview=response_preview,
                    llm_response_length=response_length,
                    parsed_data=parsed_preview,
                    parsed_data_type=parsed_data_type,
                    parsed_data_length=parsed_data_length
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    return []

            except Exception as e:
                # Capture more debugging information
                response_info = f"Response length: {len(response) if response else 0}, Type: {type(response).__name__}"
                parsed_info = f"Parsed data: {type(relationship_data).__name__}, Length: {len(relationship_data) if relationship_data else 0}"

                self.logger.warning(
                    f"Relationship extraction failed for batch (attempt {attempt + 1}/{max_retries})",
                    batch_size=len(facts),
                    error=str(e),
                    error_type=type(e).__name__,
                    response_info=response_info,
                    parsed_info=parsed_info
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    self.logger.info("Relationship extraction failed after all attempts - returning empty relationships")
                    return []

        # If we get here, all retries failed
        self.logger.error(
            f"Relationship extraction failed after {max_retries} attempts",
            batch_size=len(facts),
            fact_ids=[f.id for f in facts]
        )
        return []
    
    def _fact_to_prompt_dict(self, fact: Fact) -> Dict[str, Any]:
        """Convert fact to dictionary for LLM prompt.
        
        Args:
            fact: Fact to convert
            
        Returns:
            Dictionary representation for prompt
        """
        return {
            'id': fact.id,
            'fact_text': fact.fact_text,
            'primary_entities': fact.structured_metadata.primary_entities if fact.structured_metadata else [],
            'relationships': fact.structured_metadata.relationships if fact.structured_metadata else [],
            'domain_concepts': fact.structured_metadata.domain_concepts if fact.structured_metadata else [],
            'fact_type': fact.fact_type
        }
    
    def _parse_relationship_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for relationships with robust error handling.

        Args:
            response: Raw LLM response

        Returns:
            List of relationship dictionaries
        """
        if not response or not response.strip():
            return []

        try:
            # Clean the response first
            cleaned_response = response.strip()

            # Pre-process to fix malformed keys with newlines
            cleaned_response = self._preprocess_malformed_keys(cleaned_response)

            # Try to find JSON array in response using multiple patterns
            import re

            # Pattern 1: Standard JSON array
            json_match = re.search(r'\[.*?\]', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    relationships = json.loads(json_str)
                    if isinstance(relationships, list):
                        # Handle empty array case explicitly
                        if len(relationships) == 0:
                            self.logger.debug("LLM returned empty JSON array - no relationships found")
                            return []
                        return self._validate_relationships(relationships)
                except json.JSONDecodeError:
                    pass

            # Pattern 2: Try to extract JSON objects between brackets
            json_objects = re.findall(r'\{[^{}]*\}', cleaned_response, re.DOTALL)
            if json_objects:
                relationships = []
                for obj_str in json_objects:
                    try:
                        obj = json.loads(obj_str)
                        if self._is_valid_relationship_object(obj):
                            relationships.append(obj)
                    except json.JSONDecodeError:
                        continue
                if relationships:
                    return relationships

            # Pattern 3: Try to parse entire response as JSON
            try:
                relationships = json.loads(cleaned_response)
                if isinstance(relationships, list):
                    return self._validate_relationships(relationships)
                elif isinstance(relationships, dict) and self._is_valid_relationship_object(relationships):
                    return [relationships]
            except json.JSONDecodeError:
                pass

            # Pattern 4: Try to fix common JSON issues
            fixed_response = self._fix_common_json_issues(cleaned_response)
            if fixed_response != cleaned_response:
                try:
                    relationships = json.loads(fixed_response)
                    if isinstance(relationships, list):
                        return self._validate_relationships(relationships)
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            self.logger.warning(
                "Unexpected error parsing relationship response",
                error=str(e),
                response_preview=response[:200]
            )

        # Log the failure for debugging
        self.logger.warning(
            "Failed to parse relationship response - no valid JSON found",
            response_preview=response[:300],
            response_length=len(response)
        )

        return []

    def _fix_common_json_issues(self, response: str) -> str:
        """Fix common JSON formatting issues in LLM responses.

        Args:
            response: Raw response string

        Returns:
            Fixed response string
        """
        # Remove common prefixes/suffixes
        response = re.sub(r'^.*?(\[|\{)', r'\1', response, flags=re.DOTALL)
        response = re.sub(r'(\]|\}).*?$', r'\1', response, flags=re.DOTALL)

        # Fix trailing commas
        response = re.sub(r',\s*(\]|\})', r'\1', response)

        # Fix missing quotes around keys
        response = re.sub(r'(\w+):', r'"\1":', response)

        # Fix single quotes to double quotes
        response = response.replace("'", '"')

        # Fix keys with extra whitespace/newlines - more aggressive approach
        response = re.sub(r'"\s*\n?\s*(\w+)\s*\n?\s*":', r'"\1":', response)

        # Handle keys that start with newlines and whitespace
        response = re.sub(r'"\s*\n\s*(\w+)":', r'"\1":', response)

        # Handle keys that end with newlines and whitespace
        response = re.sub(r'"(\w+)\s*\n\s*":', r'"\1":', response)

        return response.strip()

    def _preprocess_malformed_keys(self, response: str) -> str:
        """Preprocess response to fix malformed JSON keys with newlines.

        Args:
            response: Raw response string

        Returns:
            Preprocessed response string
        """
        import re

        # Fix the specific pattern: "[\n    ]key_name" -> "key_name"
        # This handles keys that have newlines and whitespace inside the quotes
        response = re.sub(r'"\s*\n\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*":', r'"\1":', response)

        # Fix keys that have newlines at the end: "key_name[\n    ]" -> "key_name"
        response = re.sub(r'"([a-zA-Z_][a-zA-Z0-9_]*)\s*\n\s*":', r'"\1":', response)

        # Fix keys with excessive whitespace: "   key_name   " -> "key_name"
        response = re.sub(r'"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*":', r'"\1":', response)

        return response

    def _is_valid_relationship_object(self, obj: Dict[str, Any]) -> bool:
        """Check if an object has the required relationship fields.

        Args:
            obj: Dictionary to validate

        Returns:
            True if valid relationship object
        """
        required_fields = ['source_fact_id', 'target_fact_id', 'relation_type']
        return all(field in obj for field in required_fields)

    def _validate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and filter relationship objects.

        Args:
            relationships: List of relationship dictionaries

        Returns:
            List of valid relationship dictionaries
        """
        valid_relationships = []
        for rel in relationships:
            if isinstance(rel, dict):
                # Normalize the relationship object to handle key issues
                normalized_rel = self._normalize_relationship_object(rel)
                if self._is_valid_relationship_object(normalized_rel):
                    # Set default values for missing fields
                    normalized_rel.setdefault('confidence', 0.7)
                    normalized_rel.setdefault('context', '')
                    valid_relationships.append(normalized_rel)
                else:
                    self.logger.debug(f"Skipping invalid relationship object after normalization: {normalized_rel}")
            else:
                self.logger.debug(f"Skipping non-dict relationship object: {rel}")

        return valid_relationships

    def _normalize_relationship_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize relationship object keys to handle whitespace issues.

        Args:
            obj: Raw relationship object

        Returns:
            Normalized relationship object
        """
        normalized = {}

        # Map of expected keys to handle variations
        key_mappings = {
            'source_fact_id': ['source_fact_id', 'sourcefactid', 'source_id'],
            'target_fact_id': ['target_fact_id', 'targetfactid', 'target_id'],
            'relation_type': ['relation_type', 'relationtype', 'type'],
            'confidence': ['confidence', 'conf'],
            'context': ['context', 'description', 'desc']
        }

        # Normalize keys by removing whitespace and converting to lowercase
        for key, value in obj.items():
            # Clean the key
            clean_key = key.strip().lower().replace(' ', '').replace('\n', '').replace('\t', '')

            # Find the correct standard key
            for standard_key, variations in key_mappings.items():
                if clean_key in [v.lower() for v in variations]:
                    normalized[standard_key] = value
                    break
            else:
                # Keep unknown keys as-is but cleaned
                normalized[clean_key] = value

        return normalized

    def _safe_get_key(self, data: Dict[str, Any], key: str, default: str = '') -> str:
        """Safely get a key from a dictionary, handling malformed keys with whitespace.

        Args:
            data: Dictionary to search
            key: Key to find
            default: Default value if key not found

        Returns:
            Value for the key or default
        """
        # First try the exact key
        if key in data:
            return str(data[key])

        # Try to find the key with variations (whitespace, newlines, etc.)
        for actual_key, value in data.items():
            # Clean the actual key by removing whitespace and newlines
            clean_key = actual_key.strip().replace('\n', '').replace('\t', '').replace(' ', '')
            if clean_key == key:
                return str(value)

            # Also try without quotes if the key is quoted
            if clean_key.startswith('"') and clean_key.endswith('"'):
                clean_key = clean_key[1:-1]
                if clean_key == key:
                    return str(value)

        # Key not found
        return default

    def _create_fact_relation_objects(
        self,
        relationship_data: List[Dict[str, Any]],
        facts: List[Fact]
    ) -> List[FactRelation]:
        """Create FactRelation objects from parsed data.

        Args:
            relationship_data: List of relationship dictionaries
            facts: List of facts for ID validation

        Returns:
            List of FactRelation objects
        """
        # Handle empty relationship data
        if not relationship_data:
            self.logger.debug("No relationship data to process - returning empty list")
            return []

        relationships = []
        fact_ids = {fact.id for fact in facts}
        
        for i, rel_data in enumerate(relationship_data):
            try:
                # Validate input data structure
                if not isinstance(rel_data, dict):
                    self.logger.warning(
                        f"Relationship data {i} is not a dictionary",
                        rel_data_type=type(rel_data),
                        rel_data=str(rel_data)[:100]
                    )
                    continue

                # Use safe key extraction to handle malformed keys
                source_id = self._safe_get_key(rel_data, 'source_fact_id')
                target_id = self._safe_get_key(rel_data, 'target_fact_id')
                relation_type = self._safe_get_key(rel_data, 'relation_type')

                # Validate required fields
                if not source_id:
                    self.logger.debug(f"Relationship {i} missing source_fact_id", rel_data=rel_data)
                    continue
                if not target_id:
                    self.logger.debug(f"Relationship {i} missing target_fact_id", rel_data=rel_data)
                    continue
                if not relation_type:
                    self.logger.debug(f"Relationship {i} missing relation_type", rel_data=rel_data)
                    continue

                # Validate fact IDs exist
                if source_id not in fact_ids:
                    self.logger.debug(
                        f"Source fact ID not found: {source_id}",
                        available_ids=list(fact_ids)[:5]  # Show first 5 for debugging
                    )
                    continue
                if target_id not in fact_ids:
                    self.logger.debug(
                        f"Target fact ID not found: {target_id}",
                        available_ids=list(fact_ids)[:5]  # Show first 5 for debugging
                    )
                    continue

                # Validate relation type
                valid_types = FactRelationType.all_types()
                if relation_type not in valid_types:
                    self.logger.debug(
                        f"Invalid relation type: {relation_type}",
                        valid_types=valid_types
                    )
                    continue

                # Avoid self-relationships
                if source_id == target_id:
                    self.logger.debug(f"Skipping self-relationship for fact {source_id}")
                    continue

                # Create relationship object with explicit parameter validation
                try:
                    # Use safe key extraction for confidence and context too
                    confidence_str = self._safe_get_key(rel_data, 'confidence', '0.7')
                    context_str = self._safe_get_key(rel_data, 'context', '')

                    try:
                        confidence = float(confidence_str)
                    except (ValueError, TypeError):
                        confidence = 0.7

                    relationship = FactRelation(
                        source_fact_id=source_id,
                        target_fact_id=target_id,
                        relation_type=relation_type,
                        confidence=confidence,
                        context=context_str,
                        relationship_strength=rel_data.get('relationship_strength'),
                        evidence_quality=rel_data.get('evidence_quality'),
                        source_evidence=rel_data.get('source_evidence')
                    )
                except Exception as create_error:
                    self.logger.warning(
                        f"Failed to create FactRelation object for relationship {i}",
                        source_fact_id=source_id,
                        target_fact_id=target_id,
                        relation_type=relation_type,
                        create_error=str(create_error),
                        create_error_type=type(create_error).__name__
                    )
                    continue

                relationships.append(relationship)
                self.logger.debug(
                    f"Created relationship: {source_id} --[{relation_type}]--> {target_id}"
                )

            except Exception as e:
                self.logger.warning(
                    "Failed to create relationship object",
                    relationship_index=i,
                    rel_data=rel_data,
                    error=str(e),
                    error_type=type(e).__name__
                )
                continue
        
        return relationships
    
    def _build_graph_structure(self, facts: List[Fact], relationships: List[FactRelation]) -> Graph:
        """Build graph structure from facts and relationships.
        
        Args:
            facts: List of facts (nodes)
            relationships: List of relationships (edges)
            
        Returns:
            Graph object
        """
        from ..models.graph import GraphNode, GraphEdge
        
        # Convert facts to graph nodes
        nodes = []
        for fact in facts:
            node = GraphNode(
                id=fact.id,
                label="Fact",
                properties=fact.get_neo4j_properties()
            )
            nodes.append(node)
        
        # Convert relationships to graph edges
        edges = []
        for relationship in relationships:
            edge = GraphEdge(
                source=relationship.source_fact_id,
                target=relationship.target_fact_id,
                type=relationship.relation_type,
                properties=relationship.get_neo4j_properties()
            )
            edges.append(edge)
        
        return Graph(nodes=nodes, edges=edges)
    
    async def _index_facts(self, facts: List[Fact]) -> None:
        """Create keyword and domain indexes for facts.
        
        Args:
            facts: List of facts to index
        """
        # Group facts by domain
        domain_index = {}
        keyword_index = {}
        
        for fact in facts:
            # Domain indexing
            if fact.domain:
                if fact.domain not in domain_index:
                    domain_index[fact.domain] = []
                domain_index[fact.domain].append(fact.id)
            
            # Keyword indexing
            for keyword in fact.keywords:
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                keyword_index[keyword].append(fact.id)
        
        self.logger.debug(
            "Fact indexing completed",
            domains=len(domain_index),
            keywords=len(keyword_index)
        )
    
    def get_related_facts(
        self, 
        fact_id: str, 
        relationships: List[FactRelation],
        max_depth: int = 2
    ) -> List[str]:
        """Get facts related to a given fact through relationships.
        
        Args:
            fact_id: ID of the source fact
            relationships: List of all relationships
            max_depth: Maximum relationship depth to traverse
            
        Returns:
            List of related fact IDs
        """
        related_facts = set()
        current_level = {fact_id}
        
        for depth in range(max_depth):
            next_level = set()
            
            for current_fact in current_level:
                # Find outgoing relationships
                for rel in relationships:
                    if rel.source_fact_id == current_fact:
                        next_level.add(rel.target_fact_id)
                        related_facts.add(rel.target_fact_id)
                    elif rel.target_fact_id == current_fact:
                        next_level.add(rel.source_fact_id)
                        related_facts.add(rel.source_fact_id)
            
            current_level = next_level
            if not current_level:
                break
        
        # Remove the original fact ID
        related_facts.discard(fact_id)
        return list(related_facts)
    
    def analyze_fact_clusters(self, facts: List[Fact], relationships: List[FactRelation]) -> Dict[str, Any]:
        """Analyze clusters of related facts.
        
        Args:
            facts: List of all facts
            relationships: List of all relationships
            
        Returns:
            Dictionary with cluster analysis
        """
        # Build adjacency list
        adjacency = {}
        for fact in facts:
            adjacency[fact.id] = set()
        
        for rel in relationships:
            adjacency[rel.source_fact_id].add(rel.target_fact_id)
            adjacency[rel.target_fact_id].add(rel.source_fact_id)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for fact_id in adjacency:
            if fact_id not in visited:
                cluster = self._dfs_cluster(fact_id, adjacency, visited)
                if len(cluster) > 1:  # Only include clusters with multiple facts
                    clusters.append(cluster)
        
        return {
            'num_clusters': len(clusters),
            'cluster_sizes': [len(cluster) for cluster in clusters],
            'largest_cluster_size': max([len(cluster) for cluster in clusters]) if clusters else 0,
            'isolated_facts': len([cluster for cluster in clusters if len(cluster) == 1])
        }
    
    def _dfs_cluster(self, start_id: str, adjacency: Dict[str, set], visited: set) -> List[str]:
        """Depth-first search to find connected component.
        
        Args:
            start_id: Starting fact ID
            adjacency: Adjacency list representation
            visited: Set of visited fact IDs
            
        Returns:
            List of fact IDs in the cluster
        """
        cluster = []
        stack = [start_id]
        
        while stack:
            fact_id = stack.pop()
            if fact_id not in visited:
                visited.add(fact_id)
                cluster.append(fact_id)
                
                for neighbor in adjacency.get(fact_id, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return cluster

    def _create_relationship_extraction_prompt(self, fact_dicts: List[Dict], language: str = "en") -> str:
        """Create relationship extraction prompt using agents framework pattern.

        Args:
            fact_dicts: List of fact dictionaries
            language: Language for the prompt

        Returns:
            Formatted prompt string
        """
        facts_text = "\n".join([
            f"Fact {i+1}: {fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}"
            for i, fact in enumerate(fact_dicts)
        ])

        prompt = f"""Analyze the following facts and identify relationships between them.

Facts to analyze:
{facts_text}

Please identify semantic relationships between these facts and return them in JSON format.
Each relationship should have:
- source_fact_id: ID of the source fact
- target_fact_id: ID of the target fact
- relationship_type: Type of relationship (e.g., "supports", "contradicts", "elaborates", "temporal")
- confidence: Confidence score (0.0 to 1.0)
- explanation: Brief explanation of the relationship

Return only valid JSON without any additional text."""

        return prompt
