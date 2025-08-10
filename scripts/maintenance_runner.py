#!/usr/bin/env python3
"""Generic maintenance runner for one-shot maintenance container.

Select jobs to run via MORAG_MAINT_JOBS (comma-separated).
Currently supported jobs:
- keyword_deduplication
- keyword_hierarchization
- keyword_linking
- relationship_merger
- relationship_cleanup

Exit code is non-zero if any selected job fails (respecting fail-fast behavior).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
# Ensure local packages are importable without installing
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
for rel in [
    "packages/morag-graph/src",
    "packages/morag-core/src",
    "packages/morag-reasoning/src",
    "packages/morag-services/src",
]:
    p = project_root / rel
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from typing import Any, Dict, List, Callable

import structlog

# Load .env at startup so Neo4j and LLM credentials are available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # Silent fallback if python-dotenv is not installed
    pass

logger = structlog.get_logger(__name__)


def _parse_bool(val: str, default: bool = False) -> bool:
    if val is None:
        return default
    s = val.strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def parse_kwh_overrides() -> Dict[str, Any]:
    """Parse keyword hierarchization environment variables."""
    mapping = {
        "MORAG_KWH_THRESHOLD": ("threshold_min_facts", int),
        "MORAG_KWH_MIN_NEW": ("min_new_keywords", int),
        "MORAG_KWH_MAX_NEW": ("max_new_keywords", int),
        "MORAG_KWH_MIN_PER": ("min_per_new_keyword", int),
        "MORAG_KWH_MAX_MOVE_RATIO": ("max_move_ratio", float),
        "MORAG_KWH_SHARE": ("cooccurrence_min_share", float),
        "MORAG_KWH_BATCH_SIZE": ("batch_size", int),
        "MORAG_KWH_DETACH_MOVED": ("detach_moved", lambda v: v.strip().lower() == "true"),
        # Linking defaults to True and uses LLM to infer types; no envs for toggling/type
        "MORAG_KWH_DRY_RUN": ("dry_run", lambda v: v.strip().lower() == "true"),
        "MORAG_KWH_JOB_TAG": ("job_tag", str),
        "MORAG_KWH_LIMIT_KEYWORDS": ("limit_keywords", int),
    }

    overrides: Dict[str, Any] = {}
    for env_key, (param_key, caster) in mapping.items():
        val = os.getenv(env_key)
        if val is not None and val != "":
            try:
                overrides[param_key] = caster(val)
            except Exception:
                logger.warning("Invalid env value for keyword hierarchization override", key=env_key, value=val)

    # Ensure link_entities defaults to True if not provided
    if "link_entities" not in overrides:
        overrides["link_entities"] = True

    # Ensure dry_run defaults to False (apply changes by default)
    if "dry_run" not in overrides:
        overrides["dry_run"] = False

    return overrides



def parse_kwd_overrides() -> Dict[str, Any]:
    """Parse keyword deduplication environment variables."""
    mapping = {
        "MORAG_KWD_SIMILARITY_THRESHOLD": ("similarity_threshold", float),
        "MORAG_KWD_MAX_CLUSTER_SIZE": ("max_cluster_size", int),
        # "MORAG_KWD_MIN_FACTS": ("min_fact_threshold", int),  # REMOVED - filter was counterproductive
        "MORAG_KWD_PRESERVE_CONFIDENCE": ("preserve_high_confidence", float),
        "MORAG_KWD_SEMANTIC_WEIGHT": ("semantic_similarity_weight", float),
        "MORAG_KWD_BATCH_SIZE": ("batch_size", int),
        "MORAG_KWD_LIMIT_ENTITIES": ("limit_entities", int),
        "MORAG_KWD_APPLY": ("dry_run", lambda v: not (v.strip().lower() == "true")),
        "MORAG_KWD_JOB_TAG": ("job_tag", str),
        "MORAG_KWD_ENABLE_ROTATION": ("enable_rotation", lambda v: v.strip().lower() == "true"),
        "MORAG_KWD_PROCESS_ALL_SMALL": ("process_all_if_small", lambda v: v.strip().lower() == "true"),
    }


def parse_rel_merger_overrides() -> Dict[str, Any]:
    """Parse relationship merger environment variables."""
    mapping = {
        "MORAG_REL_SIMILARITY_THRESHOLD": ("similarity_threshold", float),
        "MORAG_REL_BATCH_SIZE": ("batch_size", int),
        "MORAG_REL_LIMIT_RELATIONS": ("limit_relations", int),
        "MORAG_REL_DRY_RUN": ("dry_run", lambda v: v.strip().lower() == "true"),
        "MORAG_REL_JOB_TAG": ("job_tag", str),
        "MORAG_REL_ENABLE_ROTATION": ("enable_rotation", lambda v: v.strip().lower() == "true"),
        "MORAG_REL_MERGE_BIDIRECTIONAL": ("merge_bidirectional", lambda v: v.strip().lower() == "true"),
        "MORAG_REL_MERGE_TRANSITIVE": ("merge_transitive", lambda v: v.strip().lower() == "true"),
        "MORAG_REL_MIN_CONFIDENCE": ("min_confidence", float),
    }

    overrides: Dict[str, Any] = {}
    for env_key, (param_key, caster) in mapping.items():
        val = os.getenv(env_key)
        if val is not None and val != "":
            try:
                overrides[param_key] = caster(val)
            except Exception:
                logger.warning("Invalid env value for relationship merger override", key=env_key, value=val)

    # Ensure dry_run defaults to False (apply changes by default)
    if "dry_run" not in overrides:
        overrides["dry_run"] = False

    # Ensure rotation defaults to True (prevent starvation)
    if "enable_rotation" not in overrides:
        overrides["enable_rotation"] = True

    # Ensure bidirectional merge defaults to True
    if "merge_bidirectional" not in overrides:
        overrides["merge_bidirectional"] = True

    # Ensure transitive merge defaults to False (more conservative)
    if "merge_transitive" not in overrides:
        overrides["merge_transitive"] = False

    # Log job_tag for rotation tracking
    if "job_tag" in overrides:
        logger.info("Using explicit job_tag for rotation", job_tag=overrides["job_tag"])
    else:
        logger.info("Using default date-based job_tag for rotation (set MORAG_REL_JOB_TAG for custom rotation)")

    # Log batch size clarification if set
    if "limit_relations" in overrides:
        logger.info("Rotation batch size configured", limit_relations=overrides["limit_relations"])
    if "batch_size" in overrides:
        logger.info("Merge batch size configured (internal processing)", batch_size=overrides["batch_size"])

    return overrides


def parse_cleanup_overrides() -> Dict[str, Any]:
    """Parse relationship cleanup environment variables."""
    mapping = {
        "MORAG_REL_CLEANUP_DRY_RUN": ("dry_run", lambda v: v.strip().lower() == "true"),
        "MORAG_REL_CLEANUP_BATCH_SIZE": ("batch_size", int),
        "MORAG_REL_CLEANUP_LIMIT_RELATIONS": ("limit_relations", int),
        "MORAG_REL_CLEANUP_MIN_CONFIDENCE": ("min_confidence", float),
        "MORAG_REL_CLEANUP_REMOVE_UNRELATED": ("remove_unrelated", lambda v: v.strip().lower() == "true"),
        "MORAG_REL_CLEANUP_REMOVE_GENERIC": ("remove_generic", lambda v: v.strip().lower() == "true"),
        "MORAG_REL_CLEANUP_CONSOLIDATE_SIMILAR": ("consolidate_similar", lambda v: v.strip().lower() == "true"),
        "MORAG_REL_CLEANUP_SIMILARITY_THRESHOLD": ("similarity_threshold", float),
        "MORAG_REL_CLEANUP_JOB_TAG": ("job_tag", str),
        "MORAG_REL_CLEANUP_ENABLE_ROTATION": ("enable_rotation", lambda v: v.strip().lower() == "true"),
    }

    overrides: Dict[str, Any] = {}
    for env_key, (param_key, caster) in mapping.items():
        val = os.getenv(env_key)
        if val is not None and val != "":
            try:
                overrides[param_key] = caster(val)
            except Exception:
                logger.warning("Invalid env value for relationship cleanup override", key=env_key, value=val)

    # Ensure dry_run defaults to False (apply changes by default)
    if "dry_run" not in overrides:
        overrides["dry_run"] = False

    return overrides


def parse_kwd_overrides() -> Dict[str, Any]:
    """Parse keyword deduplication environment variables."""
    mapping = {
        "MORAG_KWD_SIMILARITY_THRESHOLD": ("similarity_threshold", float),
        "MORAG_KWD_MAX_CLUSTER_SIZE": ("max_cluster_size", int),
        # "MORAG_KWD_MIN_FACTS": ("min_fact_threshold", int),  # REMOVED - filter was counterproductive
        "MORAG_KWD_PRESERVE_CONFIDENCE": ("preserve_high_confidence", float),
        "MORAG_KWD_SEMANTIC_WEIGHT": ("semantic_similarity_weight", float),
        "MORAG_KWD_BATCH_SIZE": ("batch_size", int),
        "MORAG_KWD_LIMIT_ENTITIES": ("limit_entities", int),
        "MORAG_KWD_DRY_RUN": ("dry_run", lambda v: v.strip().lower() == "true"),
        "MORAG_KWD_JOB_TAG": ("job_tag", str),
        "MORAG_KWD_ENABLE_ROTATION": ("enable_rotation", lambda v: v.strip().lower() == "true"),
        "MORAG_KWD_PROCESS_ALL_SMALL": ("process_all_if_small", lambda v: v.strip().lower() == "true"),
    }

    overrides: Dict[str, Any] = {}
    for env_key, (param_key, caster) in mapping.items():
        val = os.getenv(env_key)
        if val is not None and val != "":
            try:
                overrides[param_key] = caster(val)
            except Exception:
                logger.warning("Invalid env value for keyword deduplication override", key=env_key, value=val)

    # Ensure dry_run defaults to False (safe default for deduplication)
    if "dry_run" not in overrides:
        overrides["dry_run"] = False

    # Ensure rotation defaults to True (prevent starvation)
    if "enable_rotation" not in overrides:
        overrides["enable_rotation"] = True

    # Ensure process_all_if_small defaults to True (efficiency for small datasets)
    if "process_all_if_small" not in overrides:
        overrides["process_all_if_small"] = True

    # Log job_tag for rotation tracking
    if "job_tag" in overrides:
        logger.info("Using explicit job_tag for rotation", job_tag=overrides["job_tag"])
    else:
        logger.info("Using default date-based job_tag for rotation (set MORAG_KWD_JOB_TAG for custom rotation)")

    # Log batch size clarification if set
    if "limit_entities" in overrides:
        logger.info("Rotation batch size configured", limit_entities=overrides["limit_entities"])
    if "batch_size" in overrides:
        logger.info("Merge batch size configured (internal processing)", batch_size=overrides["batch_size"])

    return overrides


async def run_keyword_hierarchization_job() -> Dict[str, Any]:
    from morag_graph.maintenance.keyword_hierarchization import run_keyword_hierarchization

    overrides = parse_kwh_overrides()
    limit_keywords = int(overrides.pop("limit_keywords", 5))
    logger.info("Running job: keyword_hierarchization", overrides=overrides, limit_keywords=limit_keywords)
    result = await run_keyword_hierarchization(overrides, limit_keywords=limit_keywords)
    return {"job": "keyword_hierarchization", "result": result}


async def run_keyword_linking_job() -> Dict[str, Any]:
    from morag_graph.maintenance.keyword_linking import run_keyword_linking

    # Reuse KWH envs where applicable; add linking-specific ones
    overrides = parse_kwh_overrides()
    # Additional envs for linking could be parsed here if added later
    logger.info("Running job: keyword_linking", overrides=overrides)
    result = await run_keyword_linking(overrides)
    return {"job": "keyword_linking", "result": result}


async def run_keyword_deduplication_job() -> Dict[str, Any]:
    from morag_graph.maintenance.keyword_deduplication import run_keyword_deduplication

    overrides = parse_kwd_overrides()
    logger.info("Running job: keyword_deduplication", overrides=overrides)
    result = await run_keyword_deduplication(overrides)
    return {"job": "keyword_deduplication", "result": result}


async def run_relationship_merger_job() -> Dict[str, Any]:
    from morag_graph.maintenance.relationship_merger import run_relationship_merger

    overrides = parse_rel_merger_overrides()
    logger.info("Running job: relationship_merger", overrides=overrides)
    result = await run_relationship_merger(overrides)
    return {"job": "relationship_merger", "result": result}


async def run_relationship_cleanup_job() -> Dict[str, Any]:
    from morag_graph.maintenance.relationship_cleanup import run_relationship_cleanup

    overrides = parse_cleanup_overrides()
    logger.info("Running job: relationship_cleanup", overrides=overrides)
    result = await run_relationship_cleanup(overrides)
    return {"job": "relationship_cleanup", "result": result}


async def main_async() -> int:
    # Jobs to run; if not set, run all (deduplication first for optimal order)
    jobs_env = os.getenv("MORAG_MAINT_JOBS")
    if not jobs_env:
        jobs = ["keyword_deduplication", "keyword_hierarchization", "keyword_linking", "relationship_merger", "relationship_cleanup"]
    else:
        jobs = [j.strip() for j in jobs_env.split(",") if j.strip()]

    fail_fast = _parse_bool(os.getenv("MORAG_MAINT_FAIL_FAST", "true"), default=True)

    # Job dispatch table
    job_handlers: Dict[str, Callable[[], Any]] = {
        "keyword_deduplication": run_keyword_deduplication_job,
        "keyword_hierarchization": run_keyword_hierarchization_job,
        "keyword_linking": run_keyword_linking_job,
        "relationship_merger": run_relationship_merger_job,
        "relationship_cleanup": run_relationship_cleanup_job,
    }

    summaries: List[Dict[str, Any]] = []
    errors: List[str] = []

    for job in jobs:
        handler = job_handlers.get(job)
        if not handler:
            msg = f"Unknown maintenance job: {job}"
            logger.error(msg)
            errors.append(msg)
            if fail_fast:
                break
            else:
                continue
        try:
            summary = await handler()
            summaries.append(summary)
        except Exception as e:
            logger.error("Maintenance job failed", job=job, error=str(e))
            errors.append(f"{job}: {e}")
            if fail_fast:
                break

    output = {
        "jobs": jobs,
        "summaries": summaries,
        "errors": errors,
        "success": len(errors) == 0,
    }
    print(json.dumps(output, indent=2))
    return 0 if len(errors) == 0 else 1


def main() -> int:
    try:
        return asyncio.run(main_async())
    except Exception as e:
        logger.error("Maintenance runner failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())

