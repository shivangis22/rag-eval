"""Retrieval-based metrics for RAG evaluation."""

from typing import List, Set, Optional
import numpy as np
import logging

from rag_eval.metrics.base import BaseMetric
from rag_eval.dataset.schema import RAGSample, MetricResult

logger = logging.getLogger(__name__)


class RecallAtK(BaseMetric):
    """Recall@K metric for retrieval evaluation.
    
    Measures what fraction of relevant documents are retrieved in top-k.
    
    Args:
        k: Number of top documents to consider
        threshold: Optional threshold for pass/fail
    """
    
    def __init__(self, k: int = 5, threshold: Optional[float] = None, **kwargs):
        super().__init__(name=f"Recall@{k}", threshold=threshold, **kwargs)
        self.k = k
        self.requires_contexts = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute Recall@K for a sample.
        
        Args:
            sample: RAG sample with retrieved_docs and relevant_docs
            
        Returns:
            MetricResult with recall score
        """
        self.validate_sample(sample)
        
        if not sample.relevant_docs:
            return self._create_result(None, error="No relevant documents provided")
        
        retrieved = set(sample.retrieved_docs[:self.k])
        relevant = set(sample.relevant_docs)
        
        if not relevant:
            return self._create_result(None, error="Empty relevant document set")
        
        hits = len(retrieved & relevant)
        recall = hits / len(relevant)
        
        return self._create_result(
            score=recall,
            details={
                "k": self.k,
                "retrieved_count": len(retrieved),
                "relevant_count": len(relevant),
                "hits": hits
            }
        )


class PrecisionAtK(BaseMetric):
    """Precision@K metric for retrieval evaluation.
    
    Measures what fraction of retrieved documents are relevant.
    
    Args:
        k: Number of top documents to consider
        threshold: Optional threshold for pass/fail
    """
    
    def __init__(self, k: int = 5, threshold: Optional[float] = None, **kwargs):
        super().__init__(name=f"Precision@{k}", threshold=threshold, **kwargs)
        self.k = k
        self.requires_contexts = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute Precision@K for a sample.
        
        Args:
            sample: RAG sample with retrieved_docs and relevant_docs
            
        Returns:
            MetricResult with precision score
        """
        self.validate_sample(sample)
        
        if not sample.relevant_docs:
            return self._create_result(None, error="No relevant documents provided")
        
        retrieved = sample.retrieved_docs[:self.k]
        relevant = set(sample.relevant_docs)
        
        if not retrieved:
            return self._create_result(0.0, details={"k": self.k, "retrieved_count": 0})
        
        hits = sum(1 for doc in retrieved if doc in relevant)
        precision = hits / len(retrieved)
        
        return self._create_result(
            score=precision,
            details={
                "k": self.k,
                "hits": hits,
                "retrieved_count": len(retrieved)
            }
        )


class NDCG(BaseMetric):
    """Normalized Discounted Cumulative Gain.
    
    Measures ranking quality with position-based discounting.
    
    Args:
        k: Number of top documents to consider
        threshold: Optional threshold for pass/fail
    """
    
    def __init__(self, k: int = 10, threshold: Optional[float] = None, **kwargs):
        super().__init__(name=f"NDCG@{k}", threshold=threshold, **kwargs)
        self.k = k
        self.requires_contexts = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute NDCG@K for a sample.
        
        Args:
            sample: RAG sample with retrieved_docs and relevant_docs
            
        Returns:
            MetricResult with NDCG score
        """
        self.validate_sample(sample)
        
        if not sample.relevant_docs:
            return self._create_result(None, error="No relevant documents provided")
        
        retrieved = sample.retrieved_docs[:self.k]
        relevant = set(sample.relevant_docs)
        
        # Calculate DCG
        dcg = sum(
            (1 if doc in relevant else 0) / np.log2(i + 2)
            for i, doc in enumerate(retrieved)
        )
        
        # Calculate IDCG (ideal DCG)
        ideal_length = min(len(relevant), self.k)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_length))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return self._create_result(
            score=ndcg,
            details={
                "dcg": float(dcg),
                "idcg": float(idcg),
                "k": self.k
            }
        )


class MRR(BaseMetric):
    """Mean Reciprocal Rank.
    
    Measures the rank of the first relevant document.
    
    Args:
        threshold: Optional threshold for pass/fail
    """
    
    def __init__(self, threshold: Optional[float] = None, **kwargs):
        super().__init__(name="MRR", threshold=threshold, **kwargs)
        self.requires_contexts = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute MRR for a sample.
        
        Args:
            sample: RAG sample with retrieved_docs and relevant_docs
            
        Returns:
            MetricResult with reciprocal rank
        """
        self.validate_sample(sample)
        
        if not sample.relevant_docs:
            return self._create_result(None, error="No relevant documents provided")
        
        relevant = set(sample.relevant_docs)
        
        for i, doc in enumerate(sample.retrieved_docs, 1):
            if doc in relevant:
                rr = 1.0 / i
                return self._create_result(
                    score=rr,
                    details={"rank": i, "reciprocal_rank": rr}
                )
        
        # No relevant document found
        return self._create_result(
            score=0.0,
            details={"rank": None, "reciprocal_rank": 0.0}
        )


class F1Score(BaseMetric):
    """F1 Score combining precision and recall.
    
    Harmonic mean of precision and recall at k.
    
    Args:
        k: Number of top documents to consider
        threshold: Optional threshold for pass/fail
    """
    
    def __init__(self, k: int = 5, threshold: Optional[float] = None, **kwargs):
        super().__init__(name=f"F1@{k}", threshold=threshold, **kwargs)
        self.k = k
        self.requires_contexts = True
        
        # Initialize sub-metrics
        self.precision_metric = PrecisionAtK(k=k)
        self.recall_metric = RecallAtK(k=k)
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute F1 score for a sample.
        
        Args:
            sample: RAG sample with retrieved_docs and relevant_docs
            
        Returns:
            MetricResult with F1 score
        """
        self.validate_sample(sample)
        
        # Compute precision and recall
        precision_result = self.precision_metric.compute(sample)
        recall_result = self.recall_metric.compute(sample)
        
        if precision_result.score is None or recall_result.score is None:
            return self._create_result(None, error="Could not compute precision or recall")
        
        precision = precision_result.score
        recall = recall_result.score
        
        # Calculate F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return self._create_result(
            score=f1,
            details={
                "k": self.k,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        )


class ContextPrecision(BaseMetric):
    """Context Precision - measures if retrieved contexts are relevant.
    
    Similar to Precision@K but specifically for RAG context evaluation.
    """
    
    def __init__(self, threshold: Optional[float] = None, **kwargs):
        super().__init__(name="ContextPrecision", threshold=threshold, **kwargs)
        self.requires_ground_truth = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute context precision.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with context precision score
        """
        self.validate_sample(sample)
        
        if not sample.relevant_docs:
            return self._create_result(None, error="No relevant documents provided")
        
        if not sample.retrieved_docs:
            return self._create_result(0.0, details={"retrieved_count": 0})
        
        relevant = set(sample.relevant_docs)
        relevant_retrieved = sum(1 for doc in sample.retrieved_docs if doc in relevant)
        
        precision = relevant_retrieved / len(sample.retrieved_docs)
        
        return self._create_result(
            score=precision,
            details={
                "relevant_retrieved": relevant_retrieved,
                "total_retrieved": len(sample.retrieved_docs)
            }
        )


class ContextRecall(BaseMetric):
    """Context Recall - measures if all relevant contexts are retrieved.
    
    Similar to Recall but specifically for RAG context evaluation.
    """
    
    def __init__(self, threshold: Optional[float] = None, **kwargs):
        super().__init__(name="ContextRecall", threshold=threshold, **kwargs)
        self.requires_ground_truth = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute context recall.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with context recall score
        """
        self.validate_sample(sample)
        
        if not sample.relevant_docs:
            return self._create_result(None, error="No relevant documents provided")
        
        relevant = set(sample.relevant_docs)
        retrieved = set(sample.retrieved_docs)
        
        relevant_retrieved = len(relevant & retrieved)
        recall = relevant_retrieved / len(relevant)
        
        return self._create_result(
            score=recall,
            details={
                "relevant_retrieved": relevant_retrieved,
                "total_relevant": len(relevant)
            }
        )
