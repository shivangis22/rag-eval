"""Async evaluation pipeline for faster processing."""

import asyncio
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from rag_eval.metrics.base import BaseMetric
from rag_eval.dataset.schema import (
    RAGDataset,
    RAGSample,
    SampleResult,
    EvaluationResults,
    MetricResult
)

logger = logging.getLogger(__name__)


class AsyncRAGEvaluator:
    """Async evaluator for faster RAG evaluation.
    
    Processes samples concurrently for improved performance,
    especially useful for LLM-based metrics.
    
    Args:
        metrics: List of metrics to evaluate
        max_concurrent: Maximum concurrent evaluations
        config: Optional configuration
        verbose: Enable verbose logging
    
    Example:
        >>> evaluator = AsyncRAGEvaluator(
        ...     metrics=[RecallAtK(k=5), FaithfulnessMetric()],
        ...     max_concurrent=10
        ... )
        >>> results = await evaluator.evaluate_async(dataset)
    """
    
    def __init__(
        self,
        metrics: List[BaseMetric],
        max_concurrent: int = 10,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ):
        """Initialize async evaluator.
        
        Args:
            metrics: List of metrics to evaluate
            max_concurrent: Maximum concurrent tasks
            config: Optional configuration
            verbose: Enable verbose logging
        """
        self.metrics = metrics
        self.max_concurrent = max_concurrent
        self.config = config or {}
        self.verbose = verbose
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        logger.info(f"Initialized AsyncRAGEvaluator with {len(metrics)} metrics")
        logger.info(f"Max concurrent tasks: {max_concurrent}")
    
    async def evaluate_async(self, dataset: RAGDataset) -> EvaluationResults:
        """Evaluate dataset asynchronously.
        
        Args:
            dataset: Dataset to evaluate
            
        Returns:
            EvaluationResults with all metrics and aggregations
        """
        logger.info(f"Starting async evaluation of dataset: {dataset.name or 'unnamed'}")
        logger.info(f"Dataset size: {len(dataset)} samples")
        
        # Create tasks for all samples
        tasks = [
            self._evaluate_sample_async(sample, idx)
            for idx, sample in enumerate(dataset.samples)
        ]
        
        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Sample {idx} failed: {result}")
                # Create error result
                valid_results.append(SampleResult(
                    sample=dataset.samples[idx],
                    metrics={},
                    overall_score=None,
                    timestamp=datetime.utcnow()
                ))
            else:
                valid_results.append(result)
        
        # Aggregate results
        aggregated_metrics = self._aggregate_results(valid_results)
        
        # Create evaluation results
        eval_results = EvaluationResults(
            dataset_name=dataset.name,
            results=valid_results,
            aggregated_metrics=aggregated_metrics,
            metadata={
                "num_samples": len(dataset),
                "num_metrics": len(self.metrics),
                "metric_names": [m.name for m in self.metrics],
                "max_concurrent": self.max_concurrent,
                "config": self.config
            },
            timestamp=datetime.utcnow()
        )
        
        logger.info("Async evaluation complete")
        return eval_results
    
    async def _evaluate_sample_async(
        self,
        sample: RAGSample,
        idx: int
    ) -> SampleResult:
        """Evaluate a single sample asynchronously.
        
        Args:
            sample: Sample to evaluate
            idx: Sample index
            
        Returns:
            SampleResult with metric results
        """
        async with self.semaphore:
            if self.verbose and (idx + 1) % 10 == 0:
                logger.info(f"Processing sample {idx + 1}")
            
            # Run metrics concurrently
            metric_tasks = [
                self._compute_metric_async(metric, sample)
                for metric in self.metrics
            ]
            
            metric_results_list = await asyncio.gather(*metric_tasks, return_exceptions=True)
            
            # Build metric results dict
            metric_results = {}
            for metric, result in zip(self.metrics, metric_results_list):
                if isinstance(result, Exception):
                    logger.error(f"Metric {metric.name} failed: {result}")
                    metric_results[metric.name] = MetricResult(
                        metric_name=metric.name,
                        score=None,
                        error=str(result)
                    )
                else:
                    metric_results[metric.name] = result
            
            # Calculate overall score
            scores = [r.score for r in metric_results.values() if r.score is not None]
            overall_score = sum(scores) / len(scores) if scores else None
            
            return SampleResult(
                sample=sample,
                metrics=metric_results,
                overall_score=overall_score,
                timestamp=datetime.utcnow()
            )
    
    async def _compute_metric_async(
        self,
        metric: BaseMetric,
        sample: RAGSample
    ) -> MetricResult:
        """Compute metric asynchronously.
        
        Args:
            metric: Metric to compute
            sample: Sample to evaluate
            
        Returns:
            MetricResult
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, metric.compute, sample)
    
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
    
    async def evaluate_batch_async(
        self,
        samples: List[RAGSample],
        batch_size: int = 32
    ) -> List[SampleResult]:
        """Evaluate samples in batches asynchronously.
        
        Args:
            samples: List of samples to evaluate
            batch_size: Batch size for processing
            
        Returns:
            List of sample results
        """
        results = []
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(samples)-1)//batch_size + 1}")
            
            batch_tasks = [
                self._evaluate_sample_async(sample, i + idx)
                for idx, sample in enumerate(batch)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch evaluation failed: {result}")
                else:
                    results.append(result)
        
        return results


def run_async_evaluation(
    evaluator: AsyncRAGEvaluator,
    dataset: RAGDataset
) -> EvaluationResults:
    """Helper function to run async evaluation in sync context.
    
    Args:
        evaluator: Async evaluator instance
        dataset: Dataset to evaluate
        
    Returns:
        EvaluationResults
    """
    return asyncio.run(evaluator.evaluate_async(dataset))

