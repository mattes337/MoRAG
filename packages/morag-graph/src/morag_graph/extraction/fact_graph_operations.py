"""Core fact graph operations including LLM interactions."""

import json
import re
import asyncio
from typing import List, Dict, Any, Optional
import structlog

from morag_reasoning.llm import LLMClient
from ..models.fact import Fact, FactRelation, FactRelationType


class FactGraphOperations:
    """Handle core fact graph operations and LLM interactions."""

    def __init__(self, llm_client: LLMClient, logger: Optional[structlog.BoundLogger] = None):
        """Initialize operations handler."""
        self.llm_client = llm_client
        self.logger = logger or structlog.get_logger(__name__)

    async def create_fact_relationships(
        self,
        facts: List[Fact],
        min_confidence: float,
        max_relations_per_fact: int
    ) -> List[FactRelation]:
        """Create semantic relationships between facts.

        Args:
            facts: List of facts to analyze for relationships
            min_confidence: Minimum confidence threshold
            max_relations_per_fact: Maximum relationships per fact

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

            batch_relationships = await self._extract_relationships_for_batch(
                batch, min_confidence, max_relations_per_fact
            )

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

            # Brief pause between batches to avoid rate limits
            await asyncio.sleep(0.5)

        self.logger.info(
            f"Relationship extraction completed",
            total_relationships=len(relationships),
            failed_batches=failed_batches
        )

        return relationships

    async def _extract_relationships_for_batch(
        self,
        facts: List[Fact],
        min_confidence: float,
        max_relations_per_fact: int
    ) -> List[FactRelation]:
        """Extract relationships from a batch of facts using LLM."""
        try:
            # Prepare facts for the LLM prompt
            fact_data = []
            for i, fact in enumerate(facts):
                fact_dict = self._fact_to_prompt_dict(fact)
                fact_dict["id"] = i
                fact_data.append(fact_dict)

            # Create the prompt for relationship extraction
            prompt = self._create_relationship_prompt(fact_data, max_relations_per_fact)

            # Query the LLM
            response = await self.llm_client.complete(prompt)

            if not response or not response.strip():
                self.logger.warning("Empty response from LLM for relationship extraction")
                return []

            # Parse the response
            parsed_relationships = self._parse_relationship_response(response)

            # Create FactRelation objects
            fact_relations = self._create_fact_relation_objects(
                parsed_relationships, facts, min_confidence
            )

            self.logger.debug(f"Extracted {len(fact_relations)} relationships from batch of {len(facts)} facts")

            return fact_relations

        except Exception as e:
            self.logger.error(
                f"Error extracting relationships for batch",
                error=str(e),
                error_type=type(e).__name__,
                batch_size=len(facts)
            )
            return []

    def _fact_to_prompt_dict(self, fact: Fact) -> Dict[str, Any]:
        """Convert a Fact object to dictionary for LLM prompt."""
        return {
            "subject": fact.subject,
            "object": fact.object,
            "approach": fact.approach,
            "solution": fact.solution,
            "confidence": fact.confidence,
            "context": fact.context
        }

    def _create_relationship_prompt(self, facts: List[Dict[str, Any]], max_relations: int) -> str:
        """Create prompt for relationship extraction."""
        facts_text = json.dumps(facts, indent=2)

        return f"""Analyze the following facts and identify semantic relationships between them.

FACTS:
{facts_text}

INSTRUCTIONS:
1. Find meaningful relationships between facts (causal, temporal, hierarchical, similarity, etc.)
2. Maximum {max_relations} relationships per fact
3. Focus on high-confidence relationships (0.7+)
4. Return JSON array with this structure:

[
  {{
    "source_fact_id": 0,
    "target_fact_id": 1,
    "relation_type": "causes|enables|requires|similar_to|contradicts|temporal_before|part_of",
    "confidence": 0.85,
    "explanation": "Brief explanation of the relationship"
  }}
]

RESPONSE (JSON only, no other text):"""

    def _parse_relationship_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response containing relationship data."""
        try:
            # Clean and fix common JSON issues
            cleaned_response = self._fix_common_json_issues(response)

            # Parse JSON
            relationships = json.loads(cleaned_response)

            # Ensure we have a list
            if not isinstance(relationships, list):
                self.logger.warning("Response is not a list, wrapping in array")
                relationships = [relationships] if relationships else []

            # Validate and clean relationships
            validated_relationships = self._validate_relationships(relationships)

            return validated_relationships

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse relationship JSON: {e}")
            self.logger.debug(f"Raw response: {response[:500]}...")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error parsing relationships: {e}")
            return []

    def _fix_common_json_issues(self, response: str) -> str:
        """Fix common JSON formatting issues in LLM responses."""
        # Remove markdown code blocks
        response = re.sub(r'```(?:json)?\s*', '', response)
        response = re.sub(r'\s*```', '', response)

        # Remove any leading/trailing non-JSON content
        response = response.strip()

        # Find the first '[' and last ']' to extract JSON array
        start = response.find('[')
        end = response.rfind(']')

        if start >= 0 and end > start:
            response = response[start:end+1]

        # Fix common JSON issues
        response = self._preprocess_malformed_keys(response)

        return response

    def _preprocess_malformed_keys(self, response: str) -> str:
        """Fix malformed JSON keys."""
        # Fix unquoted keys (basic cases)
        response = re.sub(r'(\w+):', r'"\1":', response)

        # Fix single quotes to double quotes
        response = response.replace("'", '"')

        # Remove trailing commas
        response = re.sub(r',\s*}', '}', response)
        response = re.sub(r',\s*]', ']', response)

        return response

    def _validate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean relationship objects."""
        validated = []

        for rel in relationships:
            if not isinstance(rel, dict):
                continue

            if self._is_valid_relationship_object(rel):
                normalized = self._normalize_relationship_object(rel)
                validated.append(normalized)
            else:
                self.logger.debug(f"Invalid relationship object: {rel}")

        return validated

    def _is_valid_relationship_object(self, obj: Dict[str, Any]) -> bool:
        """Check if relationship object has required fields."""
        required_fields = ["source_fact_id", "target_fact_id", "relation_type", "confidence"]
        return all(field in obj for field in required_fields)

    def _normalize_relationship_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize relationship object fields."""
        return {
            "source_fact_id": int(obj["source_fact_id"]),
            "target_fact_id": int(obj["target_fact_id"]),
            "relation_type": str(obj["relation_type"]).lower(),
            "confidence": float(obj["confidence"]),
            "explanation": self._safe_get_key(obj, "explanation", "")
        }

    def _safe_get_key(self, data: Dict[str, Any], key: str, default: str = '') -> str:
        """Safely get a key from dictionary with fallback."""
        value = data.get(key, default)
        if value is None:
            return default
        return str(value)

    def _create_fact_relation_objects(
        self,
        relationships: List[Dict[str, Any]],
        facts: List[Fact],
        min_confidence: float
    ) -> List[FactRelation]:
        """Create FactRelation objects from parsed relationship data."""
        fact_relations = []

        for rel_data in relationships:
            try:
                source_id = rel_data["source_fact_id"]
                target_id = rel_data["target_fact_id"]
                confidence = rel_data["confidence"]

                # Check confidence threshold
                if confidence < min_confidence:
                    continue

                # Validate fact IDs
                if source_id >= len(facts) or target_id >= len(facts):
                    self.logger.warning(f"Invalid fact IDs: source={source_id}, target={target_id}, max={len(facts)-1}")
                    continue

                if source_id == target_id:
                    continue  # Skip self-references

                # Map relation type string to enum
                relation_type_str = rel_data["relation_type"]
                relation_type = self._map_relation_type(relation_type_str)

                if relation_type is None:
                    self.logger.warning(f"Unknown relation type: {relation_type_str}")
                    continue

                # Create FactRelation object
                fact_relation = FactRelation(
                    source_fact=facts[source_id],
                    target_fact=facts[target_id],
                    relation_type=relation_type,
                    confidence=confidence,
                    explanation=rel_data.get("explanation", ""),
                    context={}
                )

                fact_relations.append(fact_relation)

            except (KeyError, ValueError, TypeError) as e:
                self.logger.warning(f"Error creating fact relation: {e}, data: {rel_data}")
                continue

        return fact_relations

    def _map_relation_type(self, relation_type_str: str) -> Optional[FactRelationType]:
        """Map string to FactRelationType enum."""
        type_mapping = {
            "causes": FactRelationType.CAUSAL,
            "enables": FactRelationType.CAUSAL,
            "requires": FactRelationType.DEPENDENCY,
            "similar_to": FactRelationType.SIMILARITY,
            "contradicts": FactRelationType.CONTRADICTION,
            "temporal_before": FactRelationType.TEMPORAL,
            "part_of": FactRelationType.HIERARCHICAL
        }

        return type_mapping.get(relation_type_str.lower())
