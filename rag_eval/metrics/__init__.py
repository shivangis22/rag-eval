"""Metrics module for RAG evaluation."""

from rag_eval.metrics.base import BaseMetric, MetricConfig
from rag_eval.metrics.retrieval import RecallAtK, PrecisionAtK, NDCG, MRR
from rag_eval.metrics.semantic import SemanticSimilarity, BERTScore
from rag_eval.metrics.llm_metrics import Faithfulness, Relevance, Coherence
from rag_eval.metrics.hallucination import HallucinationDetector

__all__ = [
    "BaseMetric",
    "MetricConfig",
    "RecallAtK",
    "PrecisionAtK",
    "NDCG",
    "MRR",
    "SemanticSimilarity",
    "BERTScore",
    "Faithfulness",
    "Relevance",
    "Coherence",
    "HallucinationDetector",
]

