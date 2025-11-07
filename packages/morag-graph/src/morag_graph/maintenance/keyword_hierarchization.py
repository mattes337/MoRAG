"""Keyword Hierarchization Maintenance Job

This job identifies over-broad keywords (Entity nodes) with too many facts attached
and partitions their facts across a small set of more specific/general keywords.

Design goals:
- Safe for live systems: reads/writes in small batches, short transactions
- Idempotent-ish via job_tag on created relationships
- No heavy locks: avoid DETACH DELETE on large sets, delete per-batch only
- Reuse existing Neo4jStorage connection ops for queries
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import structlog
from morag_graph.storage import Neo4jConfig, Neo4jStorage

from .base import MaintenanceJobBase, validate_float_range, validate_positive_int
from .query_optimizer import QueryOptimizer

logger = structlog.get_logger(__name__)


@dataclass
class HierarchizationConfig:
    threshold_min_facts: int = 50
    min_new_keywords: int = 3
    max_new_keywords: int = 6
    min_per_new_keyword: int = 5
    max_move_ratio: float = 0.8  # keep some facts on original
    cooccurrence_min_share: float = (
        0.18  # proposal must cover >= 18% of keyword's facts
    )
    batch_size: int = 200
    dry_run: bool = True
    detach_moved: bool = True
    # Link parent keyword to proposals by default
    link_entities: bool = True
    job_tag: str = ""
    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

    def ensure_defaults(self) -> None:
        if not self.job_tag:
            self.job_tag = f"kw_hier_{int(time.time())}"


class KeywordHierarchizationService(MaintenanceJobBase):
    """Implements the keyword hierarchization algorithm using Neo4j.

    Facts connect to Entity via various relation types. We treat any (f:Fact)-[r]->(e:Entity)
    as a fact attachment. We propose co-occurring Entities as candidates.
    """

    def __init__(
        self, storage: Neo4jStorage, config: Optional[HierarchizationConfig] = None
    ):
        self.storage = storage
        self.hier_config = config or HierarchizationConfig()
        self.hier_config.ensure_defaults()

        # Initialize base class with config dict
        config_dict = {
            "job_tag": self.hier_config.job_tag,
            "dry_run": self.hier_config.dry_run,
            "batch_size": self.hier_config.batch_size,
            "circuit_breaker_threshold": self.hier_config.circuit_breaker_threshold,
            "circuit_breaker_timeout": self.hier_config.circuit_breaker_timeout,
        }
        super().__init__(config_dict)

        self.query_optimizer = QueryOptimizer(storage)
        self._llm_client = None

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate integer parameters
        errors.extend(
            validate_positive_int(
                self.hier_config.threshold_min_facts,
                "threshold_min_facts",
                min_value=1,
                max_value=10000,
            )
        )
        errors.extend(
            validate_positive_int(
                self.hier_config.min_new_keywords,
                "min_new_keywords",
                min_value=1,
                max_value=20,
            )
        )
        errors.extend(
            validate_positive_int(
                self.hier_config.max_new_keywords,
                "max_new_keywords",
                min_value=1,
                max_value=50,
            )
        )
        errors.extend(
            validate_positive_int(
                self.hier_config.min_per_new_keyword,
                "min_per_new_keyword",
                min_value=1,
                max_value=1000,
            )
        )
        errors.extend(
            validate_positive_int(
                self.hier_config.batch_size, "batch_size", min_value=1, max_value=10000
            )
        )

        # Validate float parameters
        errors.extend(
            validate_float_range(
                self.hier_config.max_move_ratio,
                "max_move_ratio",
                min_value=0.0,
                max_value=1.0,
            )
        )
        errors.extend(
            validate_float_range(
                self.hier_config.cooccurrence_min_share,
                "cooccurrence_min_share",
                min_value=0.0,
                max_value=1.0,
            )
        )

        # Validate logical constraints
        if self.hier_config.min_new_keywords > self.hier_config.max_new_keywords:
            errors.append("min_new_keywords must be <= max_new_keywords")

        return errors

    async def _get_llm_client(self):
        # Always try Gemini; fallback if not available
        if self._llm_client is not None:
            return self._llm_client
        try:
            from morag_reasoning.llm import LLMClient  # type: ignore

            # Pass None to use environment variables for configuration
            self._llm_client = LLMClient(None)
            return self._llm_client
        except Exception as e:
            logger.warning(
                "LLM client unavailable for entity link typing; will use fallback type",
                error=str(e),
            )
            return None

    async def run(self, limit_keywords: int = 5) -> Dict[str, Any]:
        """Run hierarchization for up to limit_keywords candidates.

        Returns a summary dict for logging/reporting.
        """
        self.log_job_start()

        # Validate configuration
        config_errors = self.validate_config()
        if config_errors:
            raise ValueError(f"Configuration errors: {'; '.join(config_errors)}")

        candidates = await self._find_candidates(limit_keywords)
        summary: Dict[str, Any] = {
            "job_tag": self.hier_config.job_tag,
            "processed": [],
            "total_candidates": len(candidates),
            "errors": [],
        }

        # Process candidates with error handling
        async def process_candidate(k):
            k_id, k_name, fact_count = k["k_id"], k["k_name"], k["fact_count"]
            return await self._process_keyword(k_id, k_name, fact_count)

        batch_result = await self.safe_execute_batch(candidates, process_candidate)
        summary["processed"] = batch_result.successful
        summary["errors"] = [str(e) for _, e in batch_result.failed]

        self.log_job_complete(summary)
        return summary

    async def _find_candidates(self, limit_keywords: int) -> List[Dict[str, Any]]:
        """Find entity candidates with high fact counts using optimized query."""
        return await self.query_optimizer.find_entities_by_fact_count(
            min_facts=self.hier_config.threshold_min_facts, limit=limit_keywords
        )

    async def _process_keyword(
        self, k_id: str, k_name: str, total_facts: int
    ) -> Dict[str, Any]:
        # Propose related entities via co-occurrence on shared facts
        proposals = await self._propose_keywords(k_id, total_facts)
        if not proposals:
            return {
                "keyword": k_name,
                "total_facts": total_facts,
                "proposals": [],
                "moved": 0,
                "kept": total_facts,
            }

        # Determine, per fact, which proposals it explicitly mentions
        fact_map = await self._map_facts_to_proposals(
            k_id, [p["e_id"] for p in proposals]
        )
        # Obtain the relationship type from (f)-[r]->(k) so we can mirror it to proposal
        reltypes = await self._get_reltypes_to_keyword(k_id)

        # Build assignments with simple affinity: direct mention of proposal entity wins
        assignments: Dict[str, List[str]] = {p["e_id"]: [] for p in proposals}
        kept = 0
        for fid, info in fact_map.items():
            targets = info.get("proposal_ids", [])
            if not targets:
                kept += 1
                continue
            # Choose first target for determinism; later we’ll balance
            assignments[targets[0]].append(fid)

        # Enforce balancing and limits
        assignments = self._enforce_balance(assignments, total_facts)

        moved_count = sum(len(v) for v in assignments.values())
        if self.hier_config.dry_run:
            logger.info(
                "[DRY-RUN] Hierarchization plan",
                keyword=k_name,
                proposals=[p["e_name"] for p in proposals],
                moved=moved_count,
                kept=total_facts - moved_count,
            )
            return {
                "keyword": k_name,
                "total_facts": total_facts,
                "proposals": [p["e_name"] for p in proposals],
                "moved": moved_count,
                "kept": total_facts - moved_count,
                "job_tag": self.hier_config.job_tag,
            }

        # Apply changes in small batches
        await self._apply_rewiring(
            k_id, assignments, reltypes, detach=self.hier_config.detach_moved
        )
        # Optionally create entity-to-entity links from parent to proposals that received assignments
        if self.hier_config.link_entities:
            await self._link_parent_to_proposals(k_id, proposals, assignments)

        return {
            "keyword": k_name,
            "total_facts": total_facts,
            "proposals": [p["e_name"] for p in proposals],
            "moved": moved_count,
            "kept": total_facts - moved_count,
            "job_tag": self.hier_config.job_tag,
        }

    async def _propose_keywords(
        self, k_id: str, total_facts: int
    ) -> List[Dict[str, Any]]:
        q = """
        MATCH (k:Entity {id: $k_id})
        MATCH (f:Fact)-[r1]->(k)
        MATCH (f)-[r2]->(e:Entity)
        WHERE e <> k
        WITH e, count(DISTINCT f) AS cofacts
        WITH e, cofacts, toFloat(cofacts) / $total_facts AS share
        WHERE share >= $min_share
        RETURN e.id AS e_id, e.name AS e_name, cofacts, share
        ORDER BY cofacts DESC
        LIMIT 50
        """
        params = {
            "k_id": k_id,
            "total_facts": max(1, total_facts),
            "min_share": self.hier_config.cooccurrence_min_share,
        }
        rows = await self.storage._connection_ops._execute_query(q, params)
        # Log diagnostics for visibility when no proposals are produced
        logger.info(
            "Co-occurrence proposals evaluated",
            keyword_id=k_id,
            total_facts=total_facts,
            min_share=self.hier_config.cooccurrence_min_share,
            found=len(rows),
        )
        # Trim and diversify simply by taking top N for now
        trimmed = rows[: self.hier_config.max_new_keywords]
        if len(trimmed) < self.hier_config.min_new_keywords:
            logger.info(
                "Insufficient proposals after trimming",
                required=self.hier_config.min_new_keywords,
                found=len(trimmed),
                max_new=self.hier_config.max_new_keywords,
                keyword_id=k_id,
            )
            return []
        return trimmed

    async def _map_facts_to_proposals(
        self, k_id: str, proposal_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        if not proposal_ids:
            return {}
        q = """
        MATCH (f:Fact)-[r]->(k:Entity {id: $k_id})
        OPTIONAL MATCH (f)-[r2]->(p:Entity)
        WHERE p.id IN $proposal_ids
        WITH f, collect(DISTINCT p.id) AS proposal_ids
        RETURN f.id AS fact_id, proposal_ids
        """
        rows = await self.storage._connection_ops._execute_query(
            q, {"k_id": k_id, "proposal_ids": proposal_ids}
        )
        return {
            row["fact_id"]: {
                "proposal_ids": [pid for pid in row["proposal_ids"] if pid]
            }
            for row in rows
        }

    async def _get_reltypes_to_keyword(self, k_id: str) -> Dict[str, str]:
        q = """
        MATCH (f:Fact)-[r]->(k:Entity {id: $k_id})
        RETURN f.id AS fact_id, type(r) AS reltype
        """
        rows = await self.storage._connection_ops._execute_query(q, {"k_id": k_id})
        return {row["fact_id"]: row["reltype"] for row in rows}

    def _enforce_balance(
        self, assignments: Dict[str, List[str]], total_facts: int
    ) -> Dict[str, List[str]]:
        # Remove proposals that do not meet minimum assigned facts
        filtered = {
            pid: facts
            for pid, facts in assignments.items()
            if len(facts) >= self.hier_config.min_per_new_keyword
        }
        # Enforce max move ratio
        max_moves = int(self.hier_config.max_move_ratio * total_facts)
        moved_so_far = 0
        balanced: Dict[str, List[str]] = {pid: [] for pid in filtered.keys()}
        # Round-robin through proposals to keep distribution even
        # Flatten as list of (pid, fid)
        pairs: List[Tuple[str, str]] = []
        for pid, fids in filtered.items():
            for fid in fids:
                pairs.append((pid, fid))
        # Simple round-robin by cycling pids
        order = list(filtered.keys())
        idx = 0
        for pid, fid in pairs:
            if moved_so_far >= max_moves:
                break
            # assign in order cycling proposals
            target_pid = order[idx % len(order)]
            balanced[target_pid].append(fid)
            moved_so_far += 1
            idx += 1
        # Remove empties
        return {pid: fids for pid, fids in balanced.items() if fids}

    async def _apply_rewiring(
        self,
        k_id: str,
        assignments: Dict[str, List[str]],
        reltypes: Dict[str, str],
        detach: bool,
    ) -> None:
        # Create new edges in batches, using same reltype as (f)->(k)
        for pid, fids in assignments.items():
            for i in range(0, len(fids), self.hier_config.batch_size):
                batch = fids[i : i + self.hier_config.batch_size]
                await self._batch_attach(k_id, pid, batch, reltypes)
        # Optionally detach from parent for moved facts
        if detach:
            for pid, fids in assignments.items():
                for i in range(0, len(fids), self.hier_config.batch_size):
                    batch = fids[i : i + self.hier_config.batch_size]
                    await self._batch_detach_from_keyword(k_id, batch)

    async def _link_parent_to_proposals(
        self,
        k_id: str,
        proposals: List[Dict[str, Any]],
        assignments: Dict[str, List[str]],
    ) -> None:
        # Only link proposals that actually received assignments (to avoid noise)
        active_pids = [pid for pid, fids in assignments.items() if fids]
        if not active_pids:
            return
        # Group proposals by pid for quick lookup
        pid_to_name: Dict[str, str] = {p["e_id"]: p["e_name"] for p in proposals}

    async def _infer_entity_link_type(
        self, parent_name: str, child_name: str, samples: List[str]
    ) -> str:
        client = await self._get_llm_client()
        fallback = "ASSOCIATED_WITH"
        # If no LLM available, fallback
        if not client:
            return f"{fallback}|A_TO_B"

        async def llm_inference():
            sample_text = "\n\n".join(samples[:3]) if samples else ""
            prompt = (
                "You are normalizing relationship types between generic keywords in a knowledge graph.\n"
                "Return STRICT JSON with keys relation_type and direction.\n"
                "- relation_type: UPPERCASE, SINGULAR, descriptive but concise (e.g., NARROWS_TO, CAUSES, TREATS, ASSOCIATED_WITH).\n"
                "- direction: 'A_TO_B' if from the first term to the second, else 'B_TO_A'.\n"
                f"Terms: A='{parent_name}', B='{child_name}'.\n"
                f"Evidence (optional):\n{sample_text}\n"
                "JSON only."
            )
            text = await client.generate(prompt)
            import json
            import re

            m = re.search(r"\{.*\}", text, re.S)
            data = json.loads(m.group(0)) if m else json.loads(text)
            rtype = str(data.get("relation_type", "")).strip().upper().replace(" ", "_")
            direction = str(data.get("direction", "A_TO_B")).strip().upper()
            if not rtype:
                rtype = fallback
            if direction not in {"A_TO_B", "B_TO_A"}:
                direction = "A_TO_B"
            if rtype in {"RELATED_TO", "RELATES_TO", "IS_RELATED_TO"}:
                rtype = fallback
            return f"{rtype}|{direction}"

        try:
            return await self.safe_llm_call(llm_inference)
        except Exception as e:
            logger.warning(
                "LLM inference failed for entity link type; using fallback",
                error=str(e),
            )
            return f"{fallback}|A_TO_B"

    async def _sample_cofacts(self, parent_id: str, child_id: str) -> List[str]:
        q = """
        MATCH (f:Fact)-[]->(k:Entity {id: $parent_id})
        MATCH (f)-[]->(p:Entity {id: $child_id})
        RETURN f.subject + ' ' + COALESCE(f.approach, '') + ' ' +
               COALESCE(f.object, '') + ' ' + COALESCE(f.solution, '') + ' ' +
               COALESCE(f.remarks, '') AS txt
        LIMIT 3
        """
        rows = await self.storage._connection_ops._execute_query(
            q, {"parent_id": parent_id, "child_id": child_id}
        )
        return [r.get("txt", "").strip() for r in rows if r.get("txt", "").strip()]
        # For each active proposal, infer relation type + direction and create link
        parent_name = (
            await self.storage._connection_ops._execute_query(
                "MATCH (k:Entity {id: $k_id}) RETURN k.name AS name", {"k_id": k_id}
            )
        )[0].get("name", k_id)

        total_linked = 0
        for pid in active_pids:
            child_name = pid_to_name.get(pid, pid)
            samples = await self._sample_cofacts(k_id, pid)
            type_and_dir = await self._infer_entity_link_type(
                parent_name, child_name, samples
            )
            rtype, direction = type_and_dir.split("|", 1)
            if direction == "A_TO_B":
                q = f"""
                MATCH (a:Entity {{id: $parent}}), (b:Entity {{id: $child}})
                MERGE (a)-[h:{rtype}]->(b)
                ON CREATE SET h.created_at = datetime()
                SET h.job_tag = $job_tag, h.created_from = 'keyword_hierarchization'
                RETURN count(h) AS linked
                """
            else:
                q = f"""
                MATCH (a:Entity {{id: $parent}}), (b:Entity {{id: $child}})
                MERGE (b)-[h:{rtype}]->(a)
                ON CREATE SET h.created_at = datetime()
                SET h.job_tag = $job_tag, h.created_from = 'keyword_hierarchization'
                RETURN count(h) AS linked
                """
            res = await self.storage._connection_ops._execute_query(
                q, {"parent": k_id, "child": pid, "job_tag": self.hier_config.job_tag}
            )
            total_linked += res[0]["linked"] if res else 0

        logger.info(
            "Entity links created",
            parent_id=k_id,
            proposals=[pid_to_name.get(pid, pid) for pid in active_pids],
            link_type="LLM_INFERRED",
            total_linked=total_linked,
        )

    async def _batch_attach(
        self, k_id: str, pid: str, fid_batch: List[str], reltypes: Dict[str, str]
    ) -> None:
        # We must apply the original reltype per fact
        # Build rows as {fid, p_id, reltype}
        rows = [
            {"fid": fid, "pid": pid, "reltype": reltypes.get(fid, "RELATES_TO")}
            for fid in fid_batch
        ]
        # Execute write: MERGE relationship of dynamic type
        # We need to UNWIND and use apoc.create.relationship alternative if APOC available; else run per-reltype groups
        # Group by reltype to construct static-type queries
        by_type: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            by_type.setdefault(r["reltype"], []).append(r)
        for rtype, items in by_type.items():
            q = f"""
            UNWIND $items AS row
            MATCH (f:Fact {{id: row.fid}})
            MATCH (p:Entity {{id: row.pid}})
            MERGE (f)-[r:{rtype}]->(p)
            ON CREATE SET r.created_at = datetime()
            SET r.job_tag = $job_tag, r.created_from = 'keyword_hierarchization'
            RETURN count(r) AS created
            """
            await self.storage._connection_ops._execute_query(
                q, {"items": items, "job_tag": self.hier_config.job_tag}
            )

    async def _batch_detach_from_keyword(self, k_id: str, fid_batch: List[str]) -> None:
        q = """
        MATCH (f:Fact)-[r]->(k:Entity {id: $k_id})
        WHERE f.id IN $fids
        WITH r LIMIT 500
        DELETE r
        RETURN count(*) AS deleted
        """
        # LIMIT to keep transactions small (server can still stream)
        await self.storage._connection_ops._execute_query(
            q, {"k_id": k_id, "fids": fid_batch}
        )


async def run_keyword_hierarchization(
    config_overrides: Optional[Dict[str, Any]] = None, limit_keywords: int = 5
) -> Dict[str, Any]:
    """Convenience entrypoint for running inside workers/CLI."""
    # Load Neo4j from environment
    import os

    neo_config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true",
        trust_all_certificates=os.getenv(
            "NEO4J_TRUST_ALL_CERTIFICATES", "false"
        ).lower()
        == "true",
    )
    storage = Neo4jStorage(neo_config)
    await storage.connect()
    try:
        cfg = HierarchizationConfig()
        if config_overrides:
            for k, v in config_overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        cfg.ensure_defaults()
        svc = KeywordHierarchizationService(storage, cfg)
        return await svc.run(limit_keywords=limit_keywords)
    finally:
        await storage.disconnect()


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run Keyword Hierarchization maintenance job"
    )
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--min-new", type=int, default=3)
    parser.add_argument("--max-new", type=int, default=6)
    parser.add_argument("--min-per", type=int, default=5)
    parser.add_argument("--max-move-ratio", type=float, default=0.8)
    parser.add_argument(
        "--share", type=float, default=0.18, help="Min co-occurrence share"
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--limit-keywords", type=int, default=5)
    parser.add_argument("--detach-moved", action="store_true", default=False)
    # Linking is enabled by default and uses LLM to determine type; no flags needed
    parser.add_argument(
        "--apply", action="store_true", help="Actually apply changes (disable dry-run)"
    )
    parser.add_argument("--job-tag", type=str, default="")
    args = parser.parse_args()

    overrides = {
        "threshold_min_facts": args.threshold,
        "min_new_keywords": args.min_new,
        "max_new_keywords": args.max_new,
        "min_per_new_keyword": args.min_per,
        "max_move_ratio": args.max_move_ratio,
        "cooccurrence_min_share": args.share,
        "batch_size": args.batch_size,
        "dry_run": not args.apply,
        "detach_moved": args.detach_moved,
        # link_entities defaults to True
        "link_entities": True,
        "job_tag": args.job_tag,
    }

    result = asyncio.run(
        run_keyword_hierarchization(overrides, limit_keywords=args.limit_keywords)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
