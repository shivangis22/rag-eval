"""Base metric interface and configuration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

from rag_eval.dataset.schema import EvaluationSample, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Configuration for metrics.
    
    Attributes:
        name: Metric name
        threshold: Pass/fail threshold
        weight: Weight for aggregation
        enabled: Whether metric is enabled
        config: Additional metric-specific configuration
    """
    name: str
    threshold: Optional[float] = None
    weight: float = 1.0
    enabled: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Initialize config dict if None."""
        if self.config is None:
            self.config = {}


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics.
    
    All metrics must inherit from this class and implement the required methods.
    
    Attributes:
        name: Metric name
        threshold: Optional threshold for pass/fail
        requires_ground_truth: Whether metric needs ground truth
        requires_contexts: Whether metric needs retrieved contexts
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs: Any
    ):
        """Initialize metric.
        
        Args:
            name: Metric name (defaults to class name)
            threshold: Pass/fail threshold
            **kwargs: Additional metric-specific parameters
        """
        self.name = name or self.__class__.__name__
        self.threshold = threshold
        self.config = kwargs
        
        # Metadata about metric requirements
        self.requires_ground_truth = False
        self.requires_contexts = True
        self.requires_answer = True
        
        logger.debug(f"Initialized metric: {self.name}")
    
    @abstractmethod
    def compute(self, sample: EvaluationSample) -> MetricResult:
        """Compute metric for a single sample.
        
        Args:
            sample: Evaluation sample
            
        Returns:
            MetricResult with score and details
            
        Raises:
            ValueError: If sample is missing required fields
        """
        pass
    
    def compute_batch(self, samples: List[EvaluationSample]) -> List[MetricResult]:
        """Compute metric for multiple samples.
        
        Default implementation calls compute() for each sample.
        Override for batch-optimized computation.
        
        Args:
            samples: List of evaluation samples
            
        Returns:
            List of metric results
        """
        results = []
        for sample in samples:
            try:
                result = self.compute(sample)
                results.append(result)
            except Exception as e:
                logger.error(f"Error computing {self.name} for sample: {e}")
                results.append(MetricResult(
                    metric_name=self.name,
                    score=None,
                    error=str(e)
                ))
        return results
    
    def aggregate(self, results: List[MetricResult]) -> Dict[str, float]:
        """Aggregate results across multiple samples.
        
        Default implementation computes mean, std, min, max, median.
        Override for custom aggregation logic.
        
        Args:
            results: List of metric results
            
        Returns:
            Dictionary with aggregated statistics
        """
        import numpy as np
        
        # Extract valid scores
        scores = [r.score for r in results if r.score is not None]
        
        if not scores:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "count": 0
            }
        
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "p25": float(np.percentile(scores, 25)),
            "p75": float(np.percentile(scores, 75)),
            "p95": float(np.percentile(scores, 95)),
            "count": len(scores)
        }
    
    def validate_sample(self, sample: EvaluationSample) -> None:
        """Validate that sample has required fields.
        
        Args:
            sample: Sample to validate
            
        Raises:
            ValueError: If sample is missing required fields
        """
        if self.requires_ground_truth and not sample.ground_truth:
            raise ValueError(f"{self.name} requires ground_truth")
        
        if self.requires_contexts and not sample.retrieved_docs:
            raise ValueError(f"{self.name} requires retrieved_docs")
        
        if self.requires_answer and not sample.generated_answer:
            raise ValueError(f"{self.name} requires generated_answer")
    
    def _create_result(
        self,
        score: Optional[float],
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> MetricResult:
        """Create a MetricResult object.
        
        Args:
            score: Metric score
            details: Additional details
            error: Error message if computation failed
            
        Returns:
            MetricResult object
        """
        passed = None
        if score is not None and self.threshold is not None:
            passed = score >= self.threshold
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            details=details or {},
            error=error
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', threshold={self.threshold})"


class CompositeMetric(BaseMetric):
    """Composite metric that combines multiple metrics.
    
    Useful for creating custom metrics that aggregate multiple base metrics.
    """
    
    def __init__(
        self,
        metrics: List[BaseMetric],
        aggregation: str = "mean",
        name: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize composite metric.
        
        Args:
            metrics: List of base metrics to combine
            aggregation: Aggregation method ('mean', 'min', 'max', 'weighted')
            name: Metric name
            **kwargs: Additional parameters
        """
        super().__init__(name=name or "CompositeMetric", **kwargs)
        self.metrics = metrics
        self.aggregation = aggregation
        
        # Inherit requirements from child metrics
        self.requires_ground_truth = any(m.requires_ground_truth for m in metrics)
        self.requires_contexts = any(m.requires_contexts for m in metrics)
        self.requires_answer = any(m.requires_answer for m in metrics)
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        """Compute all child metrics and aggregate.
        
        Args:
            sample: Evaluation sample
            
        Returns:
            Aggregated metric result
        """
        results = []
        details = {}
        
        for metric in self.metrics:
            try:
                result = metric.compute(sample)
                if result.score is not None:
                    results.append(result.score)
                    details[metric.name] = result.score
            except Exception as e:
                logger.warning(f"Error in {metric.name}: {e}")
        
        if not results:
            return self._create_result(None, error="All child metrics failed")
        
        # Aggregate scores
        if self.aggregation == "mean":
            import numpy as np
            score = float(np.mean(results))
        elif self.aggregation == "min":
            score = min(results)
        elif self.aggregation == "max":
            score = max(results)
        elif self.aggregation == "weighted":
            # Use metric weights if available
            weights = [getattr(m, 'weight', 1.0) for m in self.metrics]
            score = sum(s * w for s, w in zip(results, weights)) / sum(weights)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return self._create_result(score, details=details)


class CachedMetric(BaseMetric):
    """Wrapper that adds caching to any metric.
    
    Useful for expensive metrics (e.g., LLM-based) to avoid recomputation.
    """
    
    def __init__(self, metric: BaseMetric, cache_size: int = 1000):
        """Initialize cached metric.
        
        Args:
            metric: Base metric to wrap
            cache_size: Maximum cache size
        """
        super().__init__(name=f"Cached{metric.name}")
        self.metric = metric
        self.cache: Dict[str, MetricResult] = {}
        self.cache_size = cache_size
        
        # Inherit properties
        self.requires_ground_truth = metric.requires_ground_truth
        self.requires_contexts = metric.requires_contexts
        self.requires_answer = metric.requires_answer
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        """Compute metric with caching.
        
        Args:
            sample: Evaluation sample
            
        Returns:
            Metric result (from cache if available)
        """
        # Create cache key from sample
        cache_key = self._create_cache_key(sample)
        
        # Check cache
        if cache_key in self.cache:
            logger.debug(f"Cache hit for {self.name}")
            return self.cache[cache_key]
        
        # Compute and cache
        result = self.metric.compute(sample)
        
        # Manage cache size
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[cache_key] = result
        return result
    
    def _create_cache_key(self, sample: EvaluationSample) -> str:
        """Create cache key from sample.
        
        Args:
            sample: Evaluation sample
            
        Returns:
            Cache key string
        """
        import hashlib
        import json
        
        # Create deterministic key from relevant fields
        key_data = {
            "query": sample.query,
            "contexts": sample.retrieved_contexts,
            "answer": sample.generated_answer,
            "ground_truth": sample.ground_truth
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

