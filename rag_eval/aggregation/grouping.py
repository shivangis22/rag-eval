"""Grouping and failure analysis utilities."""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

from rag_eval.dataset.schema import SampleResult

logger = logging.getLogger(__name__)


def group_by_metadata(
    results: List[SampleResult],
    key: str
) -> Dict[str, List[SampleResult]]:
    """Group results by metadata field.
    
    Args:
        results: List of sample results
        key: Metadata key to group by
        
    Returns:
        Dictionary mapping metadata values to result lists
        
    Example:
        >>> grouped = group_by_metadata(results, "category")
        >>> for category, category_results in grouped.items():
        ...     print(f"{category}: {len(category_results)} samples")
    """
    groups = defaultdict(list)
    
    for result in results:
        if key in result.sample.metadata:
            value = result.sample.metadata[key]
            groups[str(value)].append(result)
        else:
            groups["unknown"].append(result)
    
    return dict(groups)


def cluster_failures(
    results: List[SampleResult],
    metric_name: str,
    threshold: float,
    n_clusters: int = 5
) -> List[Dict[str, Any]]:
    """Cluster failed samples to identify patterns.
    
    Args:
        results: List of sample results
        metric_name: Metric to analyze
        threshold: Threshold for failure
        n_clusters: Number of clusters
        
    Returns:
        List of cluster information dictionaries
        
    Example:
        >>> clusters = cluster_failures(results, "faithfulness", 0.8)
        >>> for cluster in clusters:
        ...     print(f"Cluster {cluster['id']}: {cluster['size']} samples")
    """
    # Filter failed samples
    failed_samples = []
    for result in results:
        if metric_name in result.metrics:
            metric_result = result.metrics[metric_name]
            if metric_result.score is not None and metric_result.score < threshold:
                failed_samples.append(result)
    
    if not failed_samples:
        logger.info("No failed samples found")
        return []
    
    logger.info(f"Found {len(failed_samples)} failed samples")
    
    # Simple clustering based on metadata
    clusters = []
    
    # Group by category if available
    if any("category" in s.sample.metadata for s in failed_samples):
        category_groups = group_by_metadata(failed_samples, "category")
        
        for idx, (category, samples) in enumerate(category_groups.items()):
            if len(samples) >= 2:  # Minimum cluster size
                clusters.append({
                    "id": idx,
                    "type": "category",
                    "category": category,
                    "size": len(samples),
                    "samples": samples,
                    "common_issues": _identify_common_issues(samples, metric_name)
                })
    
    # Group by score range
    score_ranges = [
        (0.0, 0.2, "very_low"),
        (0.2, 0.4, "low"),
        (0.4, 0.6, "medium"),
        (0.6, threshold, "below_threshold")
    ]
    
    for idx, (min_score, max_score, label) in enumerate(score_ranges):
        range_samples = [
            s for s in failed_samples
            if metric_name in s.metrics
            and s.metrics[metric_name].score is not None
            and min_score <= s.metrics[metric_name].score < max_score
        ]
        
        if range_samples:
            clusters.append({
                "id": len(clusters),
                "type": "score_range",
                "range": label,
                "min_score": min_score,
                "max_score": max_score,
                "size": len(range_samples),
                "samples": range_samples,
                "common_issues": _identify_common_issues(range_samples, metric_name)
            })
    
    return clusters[:n_clusters]  # Limit to n_clusters


def identify_failure_patterns(
    results: List[SampleResult],
    min_support: int = 3
) -> List[Dict[str, Any]]:
    """Identify common patterns in failures.
    
    Args:
        results: List of sample results
        min_support: Minimum number of samples for a pattern
        
    Returns:
        List of identified patterns
        
    Example:
        >>> patterns = identify_failure_patterns(results)
        >>> for pattern in patterns:
        ...     print(f"Pattern: {pattern['description']}")
        ...     print(f"Frequency: {pattern['frequency']}")
    """
    patterns = []
    
    # Pattern 1: Metrics that frequently fail together
    metric_failures = defaultdict(list)
    for result in results:
        failed_metrics = [
            name for name, metric_result in result.metrics.items()
            if metric_result.score is not None and metric_result.score < 0.7
        ]
        if len(failed_metrics) > 1:
            metric_failures[tuple(sorted(failed_metrics))].append(result)
    
    for metrics_combo, samples in metric_failures.items():
        if len(samples) >= min_support:
            patterns.append({
                "type": "co_failure",
                "metrics": list(metrics_combo),
                "frequency": len(samples),
                "samples": samples,
                "description": f"Metrics {', '.join(metrics_combo)} often fail together"
            })
    
    # Pattern 2: Specific metadata values associated with failures
    metadata_failures = defaultdict(lambda: defaultdict(list))
    for result in results:
        has_failure = any(
            m.score is not None and m.score < 0.7
            for m in result.metrics.values()
        )
        if has_failure:
            for key, value in result.sample.metadata.items():
                metadata_failures[key][str(value)].append(result)
    
    for key, value_dict in metadata_failures.items():
        for value, samples in value_dict.items():
            if len(samples) >= min_support:
                patterns.append({
                    "type": "metadata_pattern",
                    "metadata_key": key,
                    "metadata_value": value,
                    "frequency": len(samples),
                    "samples": samples,
                    "description": f"Samples with {key}={value} frequently fail"
                })
    
    # Pattern 3: Query length patterns
    short_query_failures = []
    long_query_failures = []
    
    for result in results:
        has_failure = any(
            m.score is not None and m.score < 0.7
            for m in result.metrics.values()
        )
        if has_failure:
            query_len = len(result.sample.query.split())
            if query_len < 5:
                short_query_failures.append(result)
            elif query_len > 20:
                long_query_failures.append(result)
    
    if len(short_query_failures) >= min_support:
        patterns.append({
            "type": "query_length",
            "length_category": "short",
            "frequency": len(short_query_failures),
            "samples": short_query_failures,
            "description": "Short queries (< 5 words) frequently fail"
        })
    
    if len(long_query_failures) >= min_support:
        patterns.append({
            "type": "query_length",
            "length_category": "long",
            "frequency": len(long_query_failures),
            "samples": long_query_failures,
            "description": "Long queries (> 20 words) frequently fail"
        })
    
    # Sort by frequency
    patterns.sort(key=lambda x: x["frequency"], reverse=True)
    
    return patterns


def _identify_common_issues(
    samples: List[SampleResult],
    metric_name: str
) -> List[str]:
    """Identify common issues in a group of samples.
    
    Args:
        samples: List of sample results
        metric_name: Metric to analyze
        
    Returns:
        List of common issue descriptions
    """
    issues = []
    
    # Check for common metadata values
    metadata_counts = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        for key, value in sample.sample.metadata.items():
            metadata_counts[key][str(value)] += 1
    
    for key, value_counts in metadata_counts.items():
        for value, count in value_counts.items():
            if count / len(samples) > 0.5:  # More than 50% have this value
                issues.append(f"Common {key}: {value}")
    
    # Check average scores
    scores = [
        s.metrics[metric_name].score
        for s in samples
        if metric_name in s.metrics and s.metrics[metric_name].score is not None
    ]
    
    if scores:
        avg_score = sum(scores) / len(scores)
        issues.append(f"Average {metric_name}: {avg_score:.3f}")
    
    # Check for empty contexts
    empty_contexts = sum(
        1 for s in samples
        if not s.sample.retrieved_docs
    )
    if empty_contexts > 0:
        issues.append(f"{empty_contexts} samples with no contexts")
    
    # Check for short answers
    short_answers = sum(
        1 for s in samples
        if len(s.sample.generated_answer.split()) < 10
    )
    if short_answers / len(samples) > 0.5:
        issues.append("Many short answers (< 10 words)")
    
    return issues


def analyze_metric_correlations(
    results: List[SampleResult]
) -> Dict[str, Dict[str, float]]:
    """Analyze correlations between different metrics.
    
    Args:
        results: List of sample results
        
    Returns:
        Dictionary mapping metric pairs to correlation coefficients
        
    Example:
        >>> correlations = analyze_metric_correlations(results)
        >>> print(correlations["faithfulness"]["relevance"])
    """
    try:
        import numpy as np
        from scipy.stats import pearsonr
    except ImportError:
        logger.error("numpy and scipy required for correlation analysis")
        return {}
    
    # Collect all metric names
    metric_names = set()
    for result in results:
        metric_names.update(result.metrics.keys())
    
    metric_names = sorted(list(metric_names))
    
    # Build score matrix
    score_matrix = {}
    for metric_name in metric_names:
        scores = []
        for result in results:
            if metric_name in result.metrics:
                score = result.metrics[metric_name].score
                if score is not None:
                    scores.append(score)
                else:
                    scores.append(np.nan)
            else:
                scores.append(np.nan)
        score_matrix[metric_name] = np.array(scores)
    
    # Compute correlations
    correlations = {}
    for metric1 in metric_names:
        correlations[metric1] = {}
        for metric2 in metric_names:
            if metric1 == metric2:
                correlations[metric1][metric2] = 1.0
            else:
                # Remove NaN values
                mask = ~(np.isnan(score_matrix[metric1]) | np.isnan(score_matrix[metric2]))
                if mask.sum() > 2:  # Need at least 3 points
                    corr, _ = pearsonr(
                        score_matrix[metric1][mask],
                        score_matrix[metric2][mask]
                    )
                    correlations[metric1][metric2] = float(corr)
                else:
                    correlations[metric1][metric2] = 0.0
    
    return correlations


