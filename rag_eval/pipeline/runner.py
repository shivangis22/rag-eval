"""Main evaluation pipeline runner."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from rag_eval.metrics.base import BaseMetric
from rag_eval.dataset.schema import (
    RAGDataset,
    RAGSample,
    SampleResult,
    EvaluationResults,
    MetricResult
)

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Main evaluator for RAG systems.
    
    Orchestrates evaluation across multiple metrics and samples.
    
    Args:
        metrics: List of metrics to evaluate
        config: Optional configuration dictionary
        verbose: Enable verbose logging
    
    Example:
        >>> from rag_eval.metrics.retrieval import RecallAtK
        >>> from rag_eval.metrics.llm_metrics import FaithfulnessMetric
        >>> 
        >>> evaluator = RAGEvaluator(
        ...     metrics=[RecallAtK(k=5), FaithfulnessMetric()]
        ... )
        >>> results = evaluator.evaluate(dataset)
    """
    
    def __init__(
        self,
        metrics: List[BaseMetric],
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ):
        """Initialize evaluator.
        
        Args:
            metrics: List of metrics to evaluate
            config: Optional configuration
            verbose: Enable verbose logging
        """
        self.metrics = metrics
        self.config = config or {}
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        logger.info(f"Initialized RAGEvaluator with {len(metrics)} metrics")
        for metric in metrics:
            logger.debug(f"  - {metric.name}")
    
    def evaluate(self, dataset: RAGDataset) -> EvaluationResults:
        """Evaluate a dataset.
        
        Args:
            dataset: Dataset to evaluate
            
        Returns:
            EvaluationResults with all metrics and aggregations
        """
        logger.info(f"Starting evaluation of dataset: {dataset.name or 'unnamed'}")
        logger.info(f"Dataset size: {len(dataset)} samples")
        
        results = []
        
        # Evaluate each sample
        for idx, sample in enumerate(dataset.samples):
            if self.verbose and (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(dataset)} samples")
            
            sample_result = self.evaluate_sample(sample)
            results.append(sample_result)
        
        # Aggregate results
        aggregated_metrics = self._aggregate_results(results)
        
        # Create evaluation results
        eval_results = EvaluationResults(
            dataset_name=dataset.name,
            results=results,
            aggregated_metrics=aggregated_metrics,
            metadata={
                "num_samples": len(dataset),
                "num_metrics": len(self.metrics),
                "metric_names": [m.name for m in self.metrics],
                "config": self.config
            },
            timestamp=datetime.utcnow()
        )
        
        logger.info("Evaluation complete")
        logger.info(f"Results: {len(results)} samples evaluated")
        
        return eval_results
    
    def evaluate_sample(self, sample: RAGSample) -> SampleResult:
        """Evaluate a single sample.
        
        Args:
            sample: Sample to evaluate
            
        Returns:
            SampleResult with metric results
        """
        metric_results = {}
        
        for metric in self.metrics:
            try:
                result = metric.compute(sample)
                metric_results[metric.name] = result
                
                if self.verbose:
                    score_str = f"{result.score:.3f}" if result.score is not None else "N/A"
                    logger.debug(f"  {metric.name}: {score_str}")
                    
            except Exception as e:
                logger.error(f"Error computing {metric.name}: {e}")
                metric_results[metric.name] = MetricResult(
                    metric_name=metric.name,
                    score=None,
                    error=str(e)
                )
        
        # Calculate overall score (average of all metrics)
        scores = [r.score for r in metric_results.values() if r.score is not None]
        overall_score = sum(scores) / len(scores) if scores else None
        
        return SampleResult(
            sample=sample,
            metrics=metric_results,
            overall_score=overall_score,
            timestamp=datetime.utcnow()
        )
    
    def _aggregate_results(
        self,
        results: List[SampleResult]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate results across all samples.
        
        Args:
            results: List of sample results
            
        Returns:
            Dictionary mapping metric names to aggregated statistics
        """
        aggregated = {}
        
        for metric in self.metrics:
            # Collect all results for this metric
            metric_results = [
                r.metrics[metric.name]
                for r in results
                if metric.name in r.metrics
            ]
            
            # Use metric's aggregation method
            aggregated[metric.name] = metric.aggregate(metric_results)
        
        return aggregated
    
    def evaluate_with_threshold(
        self,
        dataset: RAGDataset,
        thresholds: Dict[str, float]
    ) -> tuple[EvaluationResults, bool]:
        """Evaluate dataset and check against thresholds.
        
        Args:
            dataset: Dataset to evaluate
            thresholds: Dictionary mapping metric names to threshold values
            
        Returns:
            Tuple of (results, passed) where passed indicates if all thresholds met
        """
        results = self.evaluate(dataset)
        
        passed = True
        for metric_name, threshold in thresholds.items():
            if metric_name in results.aggregated_metrics:
                mean_score = results.aggregated_metrics[metric_name].get("mean", 0.0)
                if mean_score < threshold:
                    logger.warning(
                        f"Threshold not met for {metric_name}: "
                        f"{mean_score:.3f} < {threshold:.3f}"
                    )
                    passed = False
                else:
                    logger.info(
                        f"Threshold met for {metric_name}: "
                        f"{mean_score:.3f} >= {threshold:.3f}"
                    )
        
        return results, passed
    
    def get_failed_samples(
        self,
        results: EvaluationResults,
        metric_name: str,
        threshold: float
    ) -> List[SampleResult]:
        """Get samples that failed a specific metric threshold.
        
        Args:
            results: Evaluation results
            metric_name: Name of metric to check
            threshold: Threshold value
            
        Returns:
            List of failed sample results
        """
        failed = []
        
        for sample_result in results.results:
            if metric_name in sample_result.metrics:
                metric_result = sample_result.metrics[metric_name]
                if metric_result.score is not None and metric_result.score < threshold:
                    failed.append(sample_result)
        
        return failed
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RAGEvaluator":
        """Create evaluator from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured RAGEvaluator instance
        """
        # Import metrics dynamically based on config
        from rag_eval.metrics.retrieval import RecallAtK, PrecisionAtK, NDCG, MRR
        from rag_eval.metrics.llm_metrics import (
            FaithfulnessMetric,
            AnswerRelevanceMetric,
            CoherenceMetric
        )
        
        metric_map = {
            "recall@k": RecallAtK,
            "precision@k": PrecisionAtK,
            "ndcg": NDCG,
            "mrr": MRR,
            "faithfulness": FaithfulnessMetric,
            "relevance": AnswerRelevanceMetric,
            "coherence": CoherenceMetric,
        }
        
        metrics = []
        metric_configs = config.get("metrics", {})
        
        for metric_name, metric_config in metric_configs.items():
            if not metric_config.get("enabled", True):
                continue
            
            metric_class = metric_map.get(metric_name)
            if metric_class:
                # Extract metric-specific config
                kwargs = {k: v for k, v in metric_config.items() if k != "enabled"}
                metrics.append(metric_class(**kwargs))
            else:
                logger.warning(f"Unknown metric: {metric_name}")
        
        return cls(
            metrics=metrics,
            config=config,
            verbose=config.get("evaluator", {}).get("verbose", False)
        )

