#!/usr/bin/env python3
"""Generic maintenance runner for one-shot maintenance container.

Select jobs to run via MORAG_MAINT_JOBS (comma-separated).
Currently supported jobs:
- keyword_hierarchization

Exit code is non-zero if any selected job fails (respecting fail-fast behavior).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Callable

import structlog

logger = structlog.get_logger(__name__)


def _parse_bool(val: str, default: bool = False) -> bool:
    if val is None:
        return default
    s = val.strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def parse_kwh_overrides() -> Dict[str, Any]:
    mapping = {
        "MORAG_KWH_THRESHOLD": ("threshold_min_facts", int),
        "MORAG_KWH_MIN_NEW": ("min_new_keywords", int),
        "MORAG_KWH_MAX_NEW": ("max_new_keywords", int),
        "MORAG_KWH_MIN_PER": ("min_per_new_keyword", int),
        "MORAG_KWH_MAX_MOVE_RATIO": ("max_move_ratio", float),
        "MORAG_KWH_SHARE": ("cooccurrence_min_share", float),
        "MORAG_KWH_BATCH_SIZE": ("batch_size", int),
        "MORAG_KWH_DETACH_MOVED": ("detach_moved", lambda v: v.strip().lower() == "true"),
        "MORAG_KWH_APPLY": ("dry_run", lambda v: not (v.strip().lower() == "true")),
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
                logger.warning("Invalid env value for maintenance override", key=env_key, value=val)
    return overrides


async def run_keyword_hierarchization_job() -> Dict[str, Any]:
    from morag_graph.maintenance.keyword_hierarchization import run_keyword_hierarchization

    overrides = parse_kwh_overrides()
    limit_keywords = int(overrides.pop("limit_keywords", 5))
    logger.info("Running job: keyword_hierarchization", overrides=overrides, limit_keywords=limit_keywords)
    result = await run_keyword_hierarchization(overrides, limit_keywords=limit_keywords)
    return {"job": "keyword_hierarchization", "result": result}


async def main_async() -> int:
    # Jobs to run, default to keyword_hierarchization
    jobs_env = os.getenv("MORAG_MAINT_JOBS", "keyword_hierarchization")
    jobs: List[str] = [j.strip() for j in jobs_env.split(",") if j.strip()]

    fail_fast = _parse_bool(os.getenv("MORAG_MAINT_FAIL_FAST", "true"), default=True)

    # Job dispatch table
    job_handlers: Dict[str, Callable[[], Any]] = {
        "keyword_hierarchization": run_keyword_hierarchization_job,
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

