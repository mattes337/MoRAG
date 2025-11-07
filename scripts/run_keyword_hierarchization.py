#!/usr/bin/env python3
"""Standalone runner for Keyword Hierarchization maintenance job.

Intended to be used in a short-lived maintenance container that runs to completion.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict

import structlog

logger = structlog.get_logger(__name__)


def parse_env_overrides() -> Dict[str, Any]:
    # Map environment variables with MORAG_ prefix to overrides
    mapping = {
        "MORAG_KWH_THRESHOLD": ("threshold_min_facts", int),
        "MORAG_KWH_MIN_NEW": ("min_new_keywords", int),
        "MORAG_KWH_MAX_NEW": ("max_new_keywords", int),
        "MORAG_KWH_MIN_PER": ("min_per_new_keyword", int),
        "MORAG_KWH_MAX_MOVE_RATIO": ("max_move_ratio", float),
        "MORAG_KWH_SHARE": ("cooccurrence_min_share", float),
        "MORAG_KWH_BATCH_SIZE": ("batch_size", int),
        "MORAG_KWH_DETACH_MOVED": ("detach_moved", lambda v: v.lower() == "true"),
        "MORAG_KWH_APPLY": (
            "dry_run",
            lambda v: not (v.lower() == "true"),
        ),  # APPLY true -> dry_run False
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
                logger.warning(
                    "Invalid env value for maintenance override", key=env_key, value=val
                )
    return overrides


async def main_async() -> int:
    from morag_graph.maintenance.keyword_hierarchization import (
        run_keyword_hierarchization,
    )

    overrides = parse_env_overrides()
    limit_keywords = int(overrides.pop("limit_keywords", 5))

    logger.info(
        "Starting Keyword Hierarchization",
        overrides=overrides,
        limit_keywords=limit_keywords,
    )
    result = await run_keyword_hierarchization(overrides, limit_keywords=limit_keywords)
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except Exception as e:
        logger.error("Keyword Hierarchization failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
