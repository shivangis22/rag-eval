"""Statistical aggregation utilities."""

from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats as scipy_stats
import logging

logger = logging.getLogger(__name__)


def compute_statistics(scores: List[float]) -> Dict[str, float]:
    """Compute comprehensive statistics for a list of scores.
    
    Args:
        scores: List of numeric scores
        
    Returns:
        Dictionary with statistical measures
        
    Example:
        >>> scores = [0.8, 0.85, 0.9, 0.75, 0.88]
        >>> stats = compute_statistics(scores)
        >>> print(stats['mean'], stats['std'])
    """
    if not scores:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0
        }
    
    scores_array = np.array(scores)
    
    return {
        "count": len(scores),
        "mean": float(np.mean(scores_array)),
        "std": float(np.std(scores_array)),
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array)),
        "median": float(np.median(scores_array)),
        "p25": float(np.percentile(scores_array, 25)),
        "p75": float(np.percentile(scores_array, 75)),
        "p95": float(np.percentile(scores_array, 95)),
        "p99": float(np.percentile(scores_array, 99)),
        "iqr": float(np.percentile(scores_array, 75) - np.percentile(scores_array, 25)),
        "variance": float(np.var(scores_array)),
        "skewness": float(scipy_stats.skew(scores_array)),
        "kurtosis": float(scipy_stats.kurtosis(scores_array))
    }


def compute_confidence_intervals(
    scores: List[float],
    confidence: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """Compute confidence intervals for scores.
    
    Args:
        scores: List of numeric scores
        confidence: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        Dictionary with confidence intervals
        
    Example:
        >>> scores = [0.8, 0.85, 0.9, 0.75, 0.88]
        >>> ci = compute_confidence_intervals(scores)
        >>> print(ci['mean'])  # (lower_bound, upper_bound)
    """
    if not scores or len(scores) < 2:
        return {
            "mean": (0.0, 0.0),
            "median": (0.0, 0.0)
        }
    
    scores_array = np.array(scores)
    
    # Confidence interval for mean
    mean = np.mean(scores_array)
    sem = scipy_stats.sem(scores_array)
    ci_mean = scipy_stats.t.interval(
        confidence,
        len(scores_array) - 1,
        loc=mean,
        scale=sem
    )
    
    # Bootstrap confidence interval for median
    n_bootstrap = 1000
    bootstrap_medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores_array, size=len(scores_array), replace=True)
        bootstrap_medians.append(np.median(sample))
    
    ci_median = np.percentile(
        bootstrap_medians,
        [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100]
    )
    
    return {
        "mean": (float(ci_mean[0]), float(ci_mean[1])),
        "median": (float(ci_median[0]), float(ci_median[1]))
    }


def detect_outliers(
    scores: List[float],
    method: str = "iqr",
    threshold: float = 1.5
) -> Dict[str, any]:
    """Detect outliers in scores.
    
    Args:
        scores: List of numeric scores
        method: Detection method ("iqr", "zscore", "modified_zscore")
        threshold: Threshold for outlier detection
        
    Returns:
        Dictionary with outlier information
        
    Example:
        >>> scores = [0.8, 0.85, 0.9, 0.75, 0.88, 0.1, 0.95]
        >>> outliers = detect_outliers(scores)
        >>> print(outliers['outlier_indices'])
    """
    if not scores:
        return {
            "outlier_indices": [],
            "outlier_values": [],
            "num_outliers": 0,
            "outlier_ratio": 0.0
        }
    
    scores_array = np.array(scores)
    
    if method == "iqr":
        outlier_mask = _detect_outliers_iqr(scores_array, threshold)
    elif method == "zscore":
        outlier_mask = _detect_outliers_zscore(scores_array, threshold)
    elif method == "modified_zscore":
        outlier_mask = _detect_outliers_modified_zscore(scores_array, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    outlier_indices = np.where(outlier_mask)[0].tolist()
    outlier_values = scores_array[outlier_mask].tolist()
    
    return {
        "outlier_indices": outlier_indices,
        "outlier_values": outlier_values,
        "num_outliers": len(outlier_indices),
        "outlier_ratio": len(outlier_indices) / len(scores),
        "method": method,
        "threshold": threshold
    }


def _detect_outliers_iqr(scores: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """Detect outliers using IQR method.
    
    Args:
        scores: Array of scores
        threshold: IQR multiplier (default: 1.5)
        
    Returns:
        Boolean mask of outliers
    """
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (scores < lower_bound) | (scores > upper_bound)


def _detect_outliers_zscore(scores: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using Z-score method.
    
    Args:
        scores: Array of scores
        threshold: Z-score threshold (default: 3.0)
        
    Returns:
        Boolean mask of outliers
    """
    mean = np.mean(scores)
    std = np.std(scores)
    
    if std == 0:
        return np.zeros(len(scores), dtype=bool)
    
    z_scores = np.abs((scores - mean) / std)
    return z_scores > threshold


def _detect_outliers_modified_zscore(
    scores: np.ndarray,
    threshold: float = 3.5
) -> np.ndarray:
    """Detect outliers using modified Z-score method.
    
    More robust to outliers than standard Z-score.
    
    Args:
        scores: Array of scores
        threshold: Modified Z-score threshold (default: 3.5)
        
    Returns:
        Boolean mask of outliers
    """
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    
    if mad == 0:
        return np.zeros(len(scores), dtype=bool)
    
    modified_z_scores = 0.6745 * (scores - median) / mad
    return np.abs(modified_z_scores) > threshold


def compare_distributions(
    scores1: List[float],
    scores2: List[float],
    test: str = "ttest"
) -> Dict[str, any]:
    """Compare two score distributions statistically.
    
    Args:
        scores1: First list of scores
        scores2: Second list of scores
        test: Statistical test ("ttest", "mannwhitney", "ks")
        
    Returns:
        Dictionary with test results
        
    Example:
        >>> baseline_scores = [0.8, 0.85, 0.9]
        >>> new_scores = [0.82, 0.88, 0.92]
        >>> result = compare_distributions(baseline_scores, new_scores)
        >>> print(result['p_value'], result['significant'])
    """
    if not scores1 or not scores2:
        return {
            "test": test,
            "statistic": None,
            "p_value": None,
            "significant": False,
            "error": "Empty score lists"
        }
    
    scores1_array = np.array(scores1)
    scores2_array = np.array(scores2)
    
    try:
        if test == "ttest":
            statistic, p_value = scipy_stats.ttest_ind(scores1_array, scores2_array)
        elif test == "mannwhitney":
            statistic, p_value = scipy_stats.mannwhitneyu(scores1_array, scores2_array)
        elif test == "ks":
            statistic, p_value = scipy_stats.ks_2samp(scores1_array, scores2_array)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        return {
            "test": test,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "mean_diff": float(np.mean(scores2_array) - np.mean(scores1_array)),
            "effect_size": _compute_cohens_d(scores1_array, scores2_array)
        }
    except Exception as e:
        logger.error(f"Statistical test failed: {e}")
        return {
            "test": test,
            "statistic": None,
            "p_value": None,
            "significant": False,
            "error": str(e)
        }


def _compute_cohens_d(scores1: np.ndarray, scores2: np.ndarray) -> float:
    """Compute Cohen's d effect size.
    
    Args:
        scores1: First array of scores
        scores2: Second array of scores
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(scores1), len(scores2)
    var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((np.mean(scores2) - np.mean(scores1)) / pooled_std)
