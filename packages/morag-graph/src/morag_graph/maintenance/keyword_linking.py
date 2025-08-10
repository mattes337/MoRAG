"""Keyword Linking Maintenance Job

Create Entity->Entity links between keywords based on co-occurrence of shared facts.
Optionally uses an LLM to infer a specific, normalized relationship type for each pair
(e.g., NARROWS_TO, CAUSES, ASSOCIATED_WITH) to improve agent traversal and retrieval.

Defaults to dry-run for safety. Uses small batched writes and a job_tag for idempotency.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from morag_graph.storage import Neo4jStorage, Neo4jConfig

logger = structlog.get_logger(__name__)


@dataclass
class KeywordLinkingConfig:
    cooccurrence_min_share: float = 0.18  # min share of parent's facts co-occurring with proposal
    limit_parents: int = 10               # how many parent keywords to consider per run
    max_links_per_parent: int = 6         # cap links per parent to avoid explosions
    batch_size: int = 200
    dry_run: bool = True
    job_tag: str = ""
    fallback_link_type: str = "RELATES_TO"  # used only if inference fails

    def ensure_defaults(self) -> None:
        if not self.job_tag:
            self.job_tag = f"kw_link_{int(time.time())}"


class KeywordLinkingService:
    """Create Entity->Entity links between keywords using co-occurrence and LLM typing."""

    def __init__(self, storage: Neo4jStorage, config: Optional[KeywordLinkingConfig] = None):
        self.storage = storage
        self.config = config or KeywordLinkingConfig()
        self.config.ensure_defaults()
        self._llm_client = None

    async def _get_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client
        try:
            # Always prefer Gemini (via LLMClient env defaults)
            from morag_reasoning.llm import LLMClient  # type: ignore
            # Pass None to use environment variables for configuration
            self._llm_client = LLMClient(None)
            return self._llm_client
        except Exception as e:
            logger.warning("LLM client unavailable, will fallback to ASSOCIATED_WITH", error=str(e))
            return None

    async def run(self) -> Dict[str, Any]:
        parents = await self._find_parent_candidates(limit=self.config.limit_parents)
        summary: Dict[str, Any] = {"job_tag": self.config.job_tag, "processed": []}
        for row in parents:
            try:
                parent_id, parent_name, total_facts = row["k_id"], row["k_name"], row["fact_count"]
                proposals = await self._propose_links(parent_id, total_facts)
                proposals = proposals[: self.config.max_links_per_parent]
                if not proposals:
                    summary["processed"].append({
                        "parent": parent_name,
                        "links": [],
                        "total_facts": total_facts,
                    })
                    continue
                results = await self._link_parent(parent_id, parent_name, proposals)
                summary["processed"].append(results)
            except Exception as e:
                logger.error("Keyword linking failed", error=str(e))
        return summary

    async def _find_parent_candidates(self, limit: int) -> List[Dict[str, Any]]:
        q = """
        MATCH (k:Entity)
        WITH k
        MATCH (f:Fact)-[r]->(k)
        WITH k, count(DISTINCT f) AS fact_count
        WHERE fact_count >= 10
        RETURN k.id AS k_id, k.name AS k_name, fact_count
        ORDER BY fact_count DESC
        LIMIT $limit
        """
        return await self.storage._connection_ops._execute_query(q, {"limit": limit})

    async def _propose_links(self, k_id: str, total_facts: int) -> List[Dict[str, Any]]:
        q = """
        MATCH (k:Entity {id: $k_id})
        MATCH (f:Fact)-[r1]->(k)
        MATCH (f)-[r2]->(p:Entity)
        WHERE p <> k
        WITH p, count(DISTINCT f) AS cofacts
        WITH p, cofacts, toFloat(cofacts) / $total_facts AS share
        WHERE share >= $min_share
        RETURN p.id AS p_id, p.name AS p_name, cofacts, share
        ORDER BY cofacts DESC
        LIMIT 50
        """
        params = {
            "k_id": k_id,
            "total_facts": max(1, total_facts),
            "min_share": self.config.cooccurrence_min_share,
        }
        rows = await self.storage._connection_ops._execute_query(q, params)
        logger.info(
            "Keyword linking proposals evaluated",
            keyword_id=k_id,
            total_facts=total_facts,
            min_share=self.config.cooccurrence_min_share,
            found=len(rows),
        )
        return rows

    async def _infer_type(self, parent_name: str, child_name: str, samples: List[str]) -> str:
        client = await self._get_llm_client()
        if not client:
            return f"{self.config.fallback_link_type}|A_TO_B"

        sample_text = "\n\n".join(samples[:3]) if samples else ""
        prompt = (
            f"SYSTEM: You are a JSON-only API. You MUST respond with ONLY valid JSON. Any other text will cause system failure.\n\n"
            f"CONSTRAINT: Your response must be EXACTLY one JSON object. No explanations, no markdown, no additional text.\n\n"
            f"TASK: Determine the SPECIFIC relationship type (avoid generic terms):\n"
            f"A = {parent_name}\n"
            f"B = {child_name}\n"
            f"Evidence = {sample_text[:150] if sample_text else 'none'}\n\n"
            f"REQUIRED FORMAT (copy exactly, replace TYPE and DIRECTION):\n"
            f'{{"relation_type": "TYPE", "direction": "DIRECTION"}}\n\n'
            f"PREFERRED TYPES (choose most specific, avoid ASSOCIATED_WITH):\n"
            f"Medical: CAUSES, PREVENTS, TREATS, CURES, IMPROVES, WORSENS\n"
            f"Functional: REQUIRES, ENABLES, SUPPORTS, INHIBITS, ACTIVATES, REGULATES\n"
            f"Structural: PART_OF, CONTAINS, SPECIALIZES, GENERALIZES\n"
            f"Process: LEADS_TO, RESULTS_FROM, PRODUCES, CONSUMES, USES\n"
            f"Change: INCREASES, DECREASES, REDUCES, MANAGES, INFLUENCES\n"
            f"DIRECTION: A_TO_B (A affects B), B_TO_A (B affects A)\n\n"
            f"JSON:"
        )

        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                text = await client.generate(prompt)
                logger.debug(f"LLM response (attempt {attempt + 1})",
                           response_preview=text[:200],
                           response_length=len(text))

                # Parse JSON response directly (should be JSON-only now)
                import json, re

                fallback_data = {
                    "relation_type": "INFLUENCES",  # More specific than RELATES_TO
                    "direction": "A_TO_B"
                }

                # Clean the response text
                cleaned_text = text.strip()

                # Remove any potential markdown formatting
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()

                # Try direct JSON parsing first
                try:
                    data = json.loads(cleaned_text)
                    logger.debug(f"Direct JSON parsing successful")
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON
                    json_match = re.search(r'\{[^{}]*"relation_type"[^{}]*"direction"[^{}]*\}', text)
                    if json_match:
                        try:
                            data = json.loads(json_match.group(0))
                            logger.debug(f"Extracted JSON from response")
                        except json.JSONDecodeError:
                            logger.warning(f"JSON extraction failed (attempt {attempt + 1})",
                                         response_text=text[:300])
                            data = fallback_data
                    else:
                        logger.warning(f"No JSON found in response (attempt {attempt + 1})",
                                     response_text=text[:300])
                        data = fallback_data

                rtype = str(data.get("relation_type", "")).strip().upper().replace(" ", "_")
                direction = str(data.get("direction", "A_TO_B")).strip().upper()

                if not rtype:
                    rtype = self.config.fallback_link_type
                if direction not in {"A_TO_B", "B_TO_A"}:
                    direction = "A_TO_B"
                # Convert generic relationship types to more specific ones
                if rtype in {"RELATED_TO", "RELATES_TO", "IS_RELATED_TO"}:
                    rtype = "INFLUENCES"  # Use a more specific fallback
                elif rtype == "ASSOCIATED_WITH":
                    # Try to infer a more specific relationship based on context
                    if any(word in sample_text.lower() for word in ["treat", "cure", "heal", "therapy"]):
                        rtype = "TREATS"
                    elif any(word in sample_text.lower() for word in ["cause", "lead", "result"]):
                        rtype = "CAUSES"
                    elif any(word in sample_text.lower() for word in ["prevent", "avoid", "protect"]):
                        rtype = "PREVENTS"
                    elif any(word in sample_text.lower() for word in ["improve", "enhance", "boost"]):
                        rtype = "IMPROVES"
                    elif any(word in sample_text.lower() for word in ["support", "help", "assist"]):
                        rtype = "SUPPORTS"
                    else:
                        rtype = "INFLUENCES"  # Final fallback

                logger.debug(f"Parsed relationship", relation_type=rtype, direction=direction)
                return f"{rtype}|{direction}"

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"LLM inference failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s",
                                 error=str(e), parent=parent_name, child=child_name)
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning("LLM inference failed after all retries, using fallback type",
                                 error=str(e), parent=parent_name, child=child_name)
                    return f"INFLUENCES|A_TO_B"

    async def _sample_cofacts(self, parent_id: str, child_id: str) -> List[str]:
        q = """
        MATCH (f:Fact)-[]->(k:Entity {id: $parent_id})
        MATCH (f)-[]->(p:Entity {id: $child_id})
        RETURN f.subject + ' ' + COALESCE(f.approach, '') + ' ' +
               COALESCE(f.object, '') + ' ' + COALESCE(f.solution, '') + ' ' +
               COALESCE(f.remarks, '') AS txt
        LIMIT 3
        """
        rows = await self.storage._connection_ops._execute_query(q, {"parent_id": parent_id, "child_id": child_id})
        return [r.get("txt", "").strip() for r in rows if r.get("txt", "").strip()]

    async def _link_parent(self, parent_id: str, parent_name: str, proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        created = 0
        planned: List[Dict[str, Any]] = []
        for p in proposals:
            child_id, child_name = p["p_id"], p["p_name"]
            samples = await self._sample_cofacts(parent_id, child_id)
            type_and_dir = await self._infer_type(parent_name, child_name, samples)
            rtype, direction = type_and_dir.split("|", 1)

            if self.config.dry_run:
                planned.append({"parent": parent_name, "child": child_name, "type": rtype, "direction": direction})
                continue

            # Apply link
            if direction == "A_TO_B":
                q = f"""
                MATCH (a:Entity {{id: $a}}), (b:Entity {{id: $b}})
                MERGE (a)-[h:{rtype}]->(b)
                ON CREATE SET h.created_at = datetime()
                SET h.job_tag = $job_tag, h.created_from = 'keyword_linking'
                RETURN count(h) AS linked
                """
                params = {"a": parent_id, "b": child_id, "job_tag": self.config.job_tag}
            else:
                q = f"""
                MATCH (a:Entity {{id: $a}}), (b:Entity {{id: $b}})
                MERGE (b)-[h:{rtype}]->(a)
                ON CREATE SET h.created_at = datetime()
                SET h.job_tag = $job_tag, h.created_from = 'keyword_linking'
                RETURN count(h) AS linked
                """
                params = {"a": parent_id, "b": child_id, "job_tag": self.config.job_tag}
            res = await self.storage._connection_ops._execute_query(q, params)
            created += (res[0]["linked"] if res else 0)
            planned.append({"parent": parent_name, "child": child_name, "type": rtype, "direction": direction})

        logger.info("Keyword links processed", parent=parent_name, count=len(planned), created=created)
        return {"parent": parent_name, "links": planned, "created": created}


async def run_keyword_linking(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Entrypoint to run the keyword linking job."""
    import os
    neo_config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true",
        trust_all_certificates=os.getenv("NEO4J_TRUST_ALL_CERTIFICATES", "false").lower() == "true",
    )
    storage = Neo4jStorage(neo_config)
    await storage.connect()
    try:
        cfg = KeywordLinkingConfig()
        if config_overrides:
            for k, v in config_overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        cfg.ensure_defaults()
        svc = KeywordLinkingService(storage, cfg)
        return await svc.run()
    finally:
        await storage.disconnect()


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description="Run Keyword Linking maintenance job")
    parser.add_argument("--share", type=float, default=0.18, help="Min co-occurrence share")
    parser.add_argument("--limit-parents", type=int, default=10)
    parser.add_argument("--max-per-parent", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--fallback-type", type=str, default="ASSOCIATED_WITH")
    parser.add_argument("--no-llm", action="store_true", default=False)
    parser.add_argument("--apply", action="store_true", help="Actually apply changes (disable dry-run)")
    parser.add_argument("--job-tag", type=str, default="")
    args = parser.parse_args()

    overrides = {
        "cooccurrence_min_share": args.share,
        "limit_parents": args.limit_parents,
        "max_links_per_parent": args.max_per_parent,
        "batch_size": args.batch_size,
        "fallback_link_type": args.fallback_type,
        "use_llm": not args.no_llm,
        "dry_run": not args.apply,
        "job_tag": args.job_tag,
    }

    result = asyncio.run(run_keyword_linking(overrides))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

