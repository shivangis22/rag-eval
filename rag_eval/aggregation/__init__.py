"""Aggregation module for statistical analysis and grouping."""

from rag_eval.aggregation.stats import (
    compute_statistics,
    compute_confidence_intervals,
    detect_outliers
)
from rag_eval.aggregation.grouping import (
    group_by_metadata,
    cluster_failures,
    identify_failure_patterns
)

__all__ = [
    "compute_statistics",
    "compute_confidence_intervals",
    "detect_outliers",
    "group_by_metadata",
    "cluster_failures",
    "identify_failure_patterns",
]

