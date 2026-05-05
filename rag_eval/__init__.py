"""RAG Eval Pro - Production-grade RAG evaluation library."""

from rag_eval.dataset.loader import load_dataset
from rag_eval.dataset.schema import RAGSample, RAGDataset, EvaluationSample, EvaluationDataset
from rag_eval.pipeline.runner import RAGEvaluator
from rag_eval.metrics.base import BaseMetric

__version__ = "0.1.0"
__author__ = "Shivangi Shukla"
__license__ = "MIT"

__all__ = [
    "load_dataset",
    "RAGSample",
    "RAGDataset",
    "EvaluationSample",  # Backward compatibility
    "EvaluationDataset",  # Backward compatibility
    "RAGEvaluator",
    "BaseMetric",
]

