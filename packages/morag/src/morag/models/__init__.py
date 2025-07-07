"""API models for MoRAG."""

from .remote_job_api import (
    CreateRemoteJobRequest,
    CreateRemoteJobResponse,
    PollJobsRequest,
    PollJobsResponse,
    SubmitResultRequest,
    SubmitResultResponse,
    JobStatusResponse,
)

from .enhanced_query import (
    QueryType,
    ExpansionStrategy,
    FusionStrategy,
    EnhancedQueryRequest,
    EntityInfo,
    RelationInfo,
    GraphContext,
    EnhancedResult,
    EnhancedQueryResponse,
    EntityQueryRequest,
    GraphTraversalRequest,
    GraphPath,
    GraphTraversalResponse,
    GraphAnalyticsRequest,
)

# Legacy models temporarily removed - will be recreated
# from .legacy import (
#     LegacyQueryRequest,
#     LegacyQueryResponse,
#     LegacyResult,
# )

__all__ = [
    "CreateRemoteJobRequest",
    "CreateRemoteJobResponse",
    "PollJobsRequest",
    "PollJobsResponse",
    "SubmitResultRequest",
    "SubmitResultResponse",
    "JobStatusResponse",
    "QueryType",
    "ExpansionStrategy",
    "FusionStrategy",
    "EnhancedQueryRequest",
    "EntityInfo",
    "RelationInfo",
    "GraphContext",
    "EnhancedResult",
    "EnhancedQueryResponse",
    "EntityQueryRequest",
    "GraphTraversalRequest",
    "GraphPath",
    "GraphTraversalResponse",
    "GraphAnalyticsRequest",
    # Legacy models temporarily removed
    # "LegacyQueryRequest",
    # "LegacyQueryResponse",
    # "LegacyResult",
]
