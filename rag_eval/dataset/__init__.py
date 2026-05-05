"""Dataset module for loading and validating evaluation datasets."""

from rag_eval.dataset.schema import (
    RAGSample,
    RAGDataset,
    MetricResult,
    EvaluationSample,  # Backward compatibility
    EvaluationDataset,  # Backward compatibility
)
from rag_eval.dataset.loader import load_dataset, save_dataset
from rag_eval.dataset.validator import validate_dataset

__all__ = [
    "RAGSample",
    "RAGDataset",
    "MetricResult",
    "EvaluationSample",
    "EvaluationDataset",
    "load_dataset",
    "save_dataset",
    "validate_dataset",
]

