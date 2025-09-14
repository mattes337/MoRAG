"""Graph Tool Controller for Gemini Function Calling.

Implements the function calling schema for graph operations as outlined in:
graph_agent_framework_with_gemini_marktechpost_comparison.md
"""

import asyncio
import structlog
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from .citation_manager import SourceReference
from .graph_fact_extractor import ExtractedFact

logger = structlog.get_logger(__name__)


class ToolCallError(Exception):
    """Raised when a tool call fails or is not allowed."""
    pass


@dataclass
class ToolCall:
    """Represents a function call from Gemini."""
    name: str
    args: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Result of a tool call execution."""
    call_id: Optional[str]
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ActionTrace:
    """Trace of tool call execution for auditability."""
    tool_name: str
    args: Dict[str, Any]
    result: Any
    execution_time: float
    timestamp: str
    error: Optional[str] = None


class GraphToolController:
    """Controller for Gemini function calling with graph operations.
    
    Implements the tool calling schema defined in the graph agent framework
    with enforcement of hop limits, score thresholds, and allowed actions.
    """
    
    # Allowed function names
    ALLOWED_TOOLS = {
        "extract_entities",
        "match_entity", 
        "expand_neighbors",
        "fetch_chunk",
        "extract_facts"
    }
    
    def __init__(
        self,
        max_hops: int = 3,
        score_threshold: float = 0.7,
        max_entities_per_call: int = 10,
        max_neighbors_per_entity: int = 5,
        max_chunks_per_entity: int = 3
    ):
        """Initialize the graph tool controller.
        
        Args:
            max_hops: Maximum depth for neighbor expansion
            score_threshold: Minimum confidence score for results
            max_entities_per_call: Maximum entities to process per call
            max_neighbors_per_entity: Maximum neighbors to return per entity
            max_chunks_per_entity: Maximum chunks to return per entity
        """
        self.max_hops = max_hops
        self.score_threshold = score_threshold
        self.max_entities_per_call = max_entities_per_call
        self.max_neighbors_per_entity = max_neighbors_per_entity
        self.max_chunks_per_entity = max_chunks_per_entity
        
        # Action traces for auditability
        self.action_traces: List[ActionTrace] = []
        
        # Initialize services (will be set by dependency injection)
        self.entity_extractor = None
        self.graph_store = None
        self.chunk_store = None
        self.fact_extractor = None
        
    async def initialize_services(self):
        """Initialize required services."""
        # Import here to avoid circular dependencies
        try:
            from morag_graph.storage import get_neo4j_storage
            from morag_reasoning.graph_fact_extractor import GraphFactExtractor
            from morag_embedding.service import EmbeddingService
            
            self.graph_store = get_neo4j_storage()
            self.fact_extractor = GraphFactExtractor()
            self.embedding_service = EmbeddingService()
            
            logger.info("Graph tool controller services initialized")
        except Exception as e:
            logger.error("Failed to initialize services", error=str(e))
            raise ProcessingError(f"Service initialization failed: {e}")
    
    def get_function_specs(self) -> List[Dict[str, Any]]:
        """Get the JSON schema for Gemini function calling.
        
        Returns:
            List of function specifications for Gemini
        """
        return [
            {
                "name": "extract_entities",
                "description": "Extract entities with broad labels from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to extract entities from"
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "match_entity",
                "description": "Resolve mention to canonical entity in graph",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Entity name to resolve"
                        }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "expand_neighbors",
                "description": "List neighbors up to specified depth",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity ID to expand neighbors for"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Maximum depth to expand (default: 1)",
                            "minimum": 1,
                            "maximum": 3
                        }
                    },
                    "required": ["entity_id"]
                }
            },
            {
                "name": "fetch_chunk",
                "description": "Load document chunks for entity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity ID to fetch chunks for"
                        }
                    },
                    "required": ["entity_id"]
                }
            },
            {
                "name": "extract_facts",
                "description": "Extract actionable facts with machine-readable sources",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to extract facts from"
                        }
                    },
                    "required": ["text"]
                }
            }
        ]
    
    async def handle_tool_call(self, call: ToolCall) -> ToolResult:
        """Handle a tool call from Gemini.
        
        Args:
            call: The tool call to execute
            
        Returns:
            Result of the tool call execution
            
        Raises:
            ToolCallError: If the tool call is not allowed or fails
        """
        start_time = asyncio.get_event_loop().time()
        
        # Validate tool name
        if call.name not in self.ALLOWED_TOOLS:
            error_msg = f"Tool '{call.name}' not allowed. Allowed tools: {self.ALLOWED_TOOLS}"
            logger.warning("Unauthorized tool call attempted", tool=call.name)
            raise ToolCallError(error_msg)
        
        try:
            # Route to appropriate handler
            if call.name == "extract_entities":
                result = await self._extract_entities(call.args)
            elif call.name == "match_entity":
                result = await self._match_entity(call.args)
            elif call.name == "expand_neighbors":
                result = await self._expand_neighbors(call.args)
            elif call.name == "fetch_chunk":
                result = await self._fetch_chunk(call.args)
            elif call.name == "extract_facts":
                result = await self._extract_facts(call.args)
            else:
                raise ToolCallError(f"Handler not implemented for tool: {call.name}")
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Record action trace
            trace = ActionTrace(
                tool_name=call.name,
                args=call.args,
                result=result,
                execution_time=execution_time,
                timestamp=str(asyncio.get_event_loop().time())
            )
            self.action_traces.append(trace)
            
            logger.info(
                "Tool call executed successfully",
                tool=call.name,
                execution_time=execution_time
            )
            
            return ToolResult(
                call_id=call.call_id,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            error_msg = str(e)
            
            # Record failed action trace
            trace = ActionTrace(
                tool_name=call.name,
                args=call.args,
                result=None,
                execution_time=execution_time,
                timestamp=str(asyncio.get_event_loop().time()),
                error=error_msg
            )
            self.action_traces.append(trace)
            
            logger.error(
                "Tool call failed",
                tool=call.name,
                error=error_msg,
                execution_time=execution_time
            )
            
            return ToolResult(
                call_id=call.call_id,
                result=None,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _extract_entities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from text."""
        text = args.get("text", "")
        if not text:
            raise ToolCallError("Text parameter is required")
        
        if not self.fact_extractor:
            await self.initialize_services()
        
        # Use existing fact extractor to get entities
        facts = await self.fact_extractor.extract_facts(text)
        
        # Extract unique entities from facts
        entities = set()
        for fact in facts[:self.max_entities_per_call]:
            if hasattr(fact, 'structured_metadata') and fact.structured_metadata and fact.structured_metadata.primary_entities:
                for entity in fact.structured_metadata.primary_entities:
                    if entity and entity.strip():
                        entities.add(entity.strip())
        
        return {
            "entities": list(entities),
            "count": len(entities)
        }
    
    async def _match_entity(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve entity mention to canonical entity in graph."""
        name = args.get("name", "")
        if not name:
            raise ToolCallError("Name parameter is required")
        
        if not self.graph_store:
            await self.initialize_services()
        
        # Search for entities by name
        try:
            entities = await self.graph_store.search_entities(name, limit=3)
            
            if not entities:
                return {
                    "entity_id": None,
                    "canonical_name": name,
                    "confidence": 0.0,
                    "alternatives": []
                }
            
            # Return best match
            best_match = entities[0]
            alternatives = [{
                "id": e.id,
                "name": e.name,
                "type": getattr(e, 'type', 'unknown')
            } for e in entities[1:]]
            
            return {
                "entity_id": best_match.id,
                "canonical_name": best_match.name,
                "confidence": 1.0,  # Could implement similarity scoring
                "alternatives": alternatives
            }
            
        except Exception as e:
            logger.error("Entity matching failed", name=name, error=str(e))
            return {
                "entity_id": None,
                "canonical_name": name,
                "confidence": 0.0,
                "alternatives": [],
                "error": str(e)
            }
    
    async def _expand_neighbors(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Expand neighbors for an entity."""
        entity_id = args.get("entity_id", "")
        depth = args.get("depth", 1)
        
        if not entity_id:
            raise ToolCallError("Entity ID parameter is required")
        
        # Enforce depth limits
        depth = min(depth, self.max_hops)
        
        if not self.graph_store:
            await self.initialize_services()
        
        try:
            neighbors = await self.graph_store.get_neighbors(
                entity_id, 
                max_depth=depth
            )
            
            # Limit results
            neighbors = neighbors[:self.max_neighbors_per_entity]
            
            neighbor_data = []
            for neighbor in neighbors:
                neighbor_data.append({
                    "id": neighbor.id,
                    "name": neighbor.name,
                    "type": getattr(neighbor, 'type', 'unknown'),
                    "properties": getattr(neighbor, 'properties', {})
                })
            
            return {
                "neighbors": neighbor_data,
                "count": len(neighbor_data),
                "depth_used": depth
            }
            
        except Exception as e:
            logger.error("Neighbor expansion failed", entity_id=entity_id, error=str(e))
            raise ToolCallError(f"Failed to expand neighbors: {e}")
    
    async def _fetch_chunk(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch document chunks for an entity."""
        entity_id = args.get("entity_id", "")
        if not entity_id:
            raise ToolCallError("Entity ID parameter is required")
        
        if not self.graph_store:
            await self.initialize_services()
        
        try:
            # Get entity first
            entity = await self.graph_store.get_entity(entity_id)
            if not entity:
                return {
                    "chunks": [],
                    "count": 0,
                    "error": "Entity not found"
                }
            
            # For now, return mock chunks - this would need integration with chunk storage
            chunks = []
            
            # TODO: Implement actual chunk retrieval
            # chunks = await self.chunk_store.get_chunks_by_entity(entity_id)
            
            chunk_data = []
            for i, chunk in enumerate(chunks[:self.max_chunks_per_entity]):
                # Extract filename from document_id or document_title
                filename = "unknown"
                if hasattr(chunk, 'document_title') and chunk.document_title:
                    filename = chunk.document_title
                elif hasattr(chunk, 'document_id') and chunk.document_id:
                    filename = chunk.document_id
                
                # Format structured citation
                source_tag = f"[document:{filename}:{i}:entity_id={entity_id}]"
                
                chunk_data.append({
                    "text": getattr(chunk, 'text', ''),
                    "source": source_tag,
                    "chunk_id": getattr(chunk, 'id', f"chunk_{i}"),
                    "metadata": getattr(chunk, 'metadata', {})
                })
            
            return {
                "chunks": chunk_data,
                "count": len(chunk_data),
                "entity_name": entity.name
            }
            
        except Exception as e:
            logger.error("Chunk fetching failed", entity_id=entity_id, error=str(e))
            raise ToolCallError(f"Failed to fetch chunks: {e}")
    
    async def _extract_facts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Extract facts with machine-readable sources."""
        text = args.get("text", "")
        if not text:
            raise ToolCallError("Text parameter is required")
        
        if not self.fact_extractor:
            await self.initialize_services()
        
        try:
            facts = await self.fact_extractor.extract_facts(text)
            
            fact_data = []
            for i, fact in enumerate(facts):
                # Ensure structured citation format
                source_tag = f"[document:extracted_text:{i}:fact_id={fact.fact_id}]"
                
                fact_data.append({
                    "fact_id": fact.fact_id,
                    "subject": getattr(fact, 'subject', ''),
                    "predicate": getattr(fact, 'predicate', ''),
                    "object": getattr(fact, 'object', ''),
                    "confidence": getattr(fact, 'confidence', 0.0),
                    "source": source_tag,
                    "metadata": getattr(fact, 'metadata', {})
                })
            
            # Filter by confidence threshold
            filtered_facts = [
                f for f in fact_data 
                if f['confidence'] >= self.score_threshold
            ]
            
            return {
                "facts": filtered_facts,
                "total_extracted": len(fact_data),
                "filtered_count": len(filtered_facts),
                "score_threshold": self.score_threshold
            }
            
        except Exception as e:
            logger.error("Fact extraction failed", error=str(e))
            raise ToolCallError(f"Failed to extract facts: {e}")
    
    def get_action_traces(self) -> List[ActionTrace]:
        """Get all action traces for auditability."""
        return self.action_traces.copy()
    
    def clear_action_traces(self):
        """Clear action traces."""
        self.action_traces.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        tool_counts = {}
        total_time = 0.0
        error_count = 0
        
        for trace in self.action_traces:
            tool_counts[trace.tool_name] = tool_counts.get(trace.tool_name, 0) + 1
            total_time += trace.execution_time
            if trace.error:
                error_count += 1
        
        return {
            "total_calls": len(self.action_traces),
            "tool_counts": tool_counts,
            "total_execution_time": total_time,
            "error_count": error_count,
            "success_rate": (len(self.action_traces) - error_count) / max(len(self.action_traces), 1)
        }