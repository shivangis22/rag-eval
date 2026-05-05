"""Dataset validation utilities."""

from typing import List, Dict, Any, Optional
import logging

from rag_eval.dataset.schema import EvaluationDataset, EvaluationSample

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DatasetValidator:
    """Validator for evaluation datasets.
    
    Performs comprehensive validation checks on datasets including:
    - Schema validation
    - Data quality checks
    - Consistency checks
    - Completeness checks
    """
    
    def __init__(
        self,
        require_ground_truth: bool = False,
        require_contexts: bool = True,
        min_context_length: int = 1,
        min_answer_length: int = 1
    ):
        """Initialize validator.
        
        Args:
            require_ground_truth: Whether ground truth is required
            require_contexts: Whether retrieved contexts are required
            min_context_length: Minimum length for context strings
            min_answer_length: Minimum length for answer strings
        """
        self.require_ground_truth = require_ground_truth
        self.require_contexts = require_contexts
        self.min_context_length = min_context_length
        self.min_answer_length = min_answer_length
    
    def validate(
        self,
        dataset: EvaluationDataset,
        strict: bool = False
    ) -> Dict[str, Any]:
        """Validate dataset and return validation report.
        
        Args:
            dataset: Dataset to validate
            strict: If True, raise exception on any error
            
        Returns:
            Validation report with errors and warnings
            
        Raises:
            ValidationError: If strict=True and validation fails
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_samples": len(dataset.samples),
                "samples_with_ground_truth": 0,
                "samples_with_contexts": 0,
                "samples_with_answers": 0,
            }
        }
        
        # Validate each sample
        for idx, sample in enumerate(dataset.samples):
            sample_errors = self._validate_sample(sample, idx)
            report["errors"].extend(sample_errors)
            
            # Update stats
            if sample.ground_truth:
                report["stats"]["samples_with_ground_truth"] += 1
            if sample.retrieved_docs:
                report["stats"]["samples_with_contexts"] += 1
            if sample.generated_answer:
                report["stats"]["samples_with_answers"] += 1
        
        # Add warnings
        self._add_warnings(report)
        
        # Determine if valid
        report["valid"] = len(report["errors"]) == 0
        
        if not report["valid"] and strict:
            error_msg = "\n".join(report["errors"])
            raise ValidationError(f"Dataset validation failed:\n{error_msg}")
        
        return report
    
    def _validate_sample(
        self,
        sample: EvaluationSample,
        idx: int
    ) -> List[str]:
        """Validate a single sample.
        
        Args:
            sample: Sample to validate
            idx: Sample index
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Check query
        if not sample.query or len(sample.query.strip()) == 0:
            errors.append(f"Sample {idx}: Empty query")
        
        # Check contexts
        if self.require_contexts:
            if not sample.retrieved_docs:
                errors.append(f"Sample {idx}: No retrieved docs")
            else:
                for ctx_idx, ctx in enumerate(sample.retrieved_docs):
                    if len(ctx.strip()) < self.min_context_length:
                        errors.append(
                            f"Sample {idx}, Context {ctx_idx}: "
                            f"Context too short (min: {self.min_context_length})"
                        )
        
        # Check answer
        if sample.generated_answer:
            if len(sample.generated_answer.strip()) < self.min_answer_length:
                errors.append(
                    f"Sample {idx}: Answer too short (min: {self.min_answer_length})"
                )
        
        # Check ground truth
        if self.require_ground_truth and not sample.ground_truth:
            errors.append(f"Sample {idx}: Missing ground truth")
        
        return errors
    
    def _add_warnings(self, report: Dict[str, Any]) -> None:
        """Add warnings based on dataset statistics.
        
        Args:
            report: Validation report to update
        """
        stats = report["stats"]
        total = stats["total_samples"]
        
        # Warn if few samples have ground truth
        gt_ratio = stats["samples_with_ground_truth"] / total
        if gt_ratio < 0.5:
            report["warnings"].append(
                f"Only {gt_ratio:.1%} of samples have ground truth"
            )
        
        # Warn if few samples have contexts
        ctx_ratio = stats["samples_with_contexts"] / total
        if ctx_ratio < 0.9:
            report["warnings"].append(
                f"Only {ctx_ratio:.1%} of samples have retrieved contexts"
            )
        
        # Warn if few samples have answers
        ans_ratio = stats["samples_with_answers"] / total
        if ans_ratio < 0.9:
            report["warnings"].append(
                f"Only {ans_ratio:.1%} of samples have generated answers"
            )


def validate_dataset(
    dataset: EvaluationDataset,
    require_ground_truth: bool = False,
    require_contexts: bool = True,
    strict: bool = False
) -> Dict[str, Any]:
    """Validate evaluation dataset.
    
    Convenience function for dataset validation.
    
    Args:
        dataset: Dataset to validate
        require_ground_truth: Whether ground truth is required
        require_contexts: Whether retrieved contexts are required
        strict: If True, raise exception on validation failure
        
    Returns:
        Validation report
        
    Raises:
        ValidationError: If strict=True and validation fails
        
    Example:
        >>> report = validate_dataset(dataset, strict=True)
        >>> if report["valid"]:
        ...     print("Dataset is valid!")
    """
    validator = DatasetValidator(
        require_ground_truth=require_ground_truth,
        require_contexts=require_contexts
    )
    return validator.validate(dataset, strict=strict)


def check_dataset_compatibility(
    dataset: EvaluationDataset,
    metric_requirements: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Check if dataset is compatible with metric requirements.
    
    Args:
        dataset: Dataset to check
        metric_requirements: Dict mapping metric names to required fields
        
    Returns:
        Compatibility report
        
    Example:
        >>> requirements = {
        ...     "faithfulness": ["retrieved_contexts", "generated_answer"],
        ...     "recall": ["ground_truth_contexts", "retrieved_contexts"]
        ... }
        >>> report = check_dataset_compatibility(dataset, requirements)
    """
    report = {
        "compatible": True,
        "incompatible_metrics": [],
        "missing_fields": {}
    }
    
    for metric_name, required_fields in metric_requirements.items():
        missing = []
        
        for sample in dataset.samples:
            for field in required_fields:
                value = getattr(sample, field, None)
                if value is None or (isinstance(value, (list, str)) and not value):
                    missing.append(field)
                    break
        
        if missing:
            report["compatible"] = False
            report["incompatible_metrics"].append(metric_name)
            report["missing_fields"][metric_name] = list(set(missing))
    
    return report


def get_dataset_statistics(dataset: EvaluationDataset) -> Dict[str, Any]:
    """Get comprehensive statistics about the dataset.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary with dataset statistics
        
    Example:
        >>> stats = get_dataset_statistics(dataset)
        >>> print(f"Average query length: {stats['avg_query_length']}")
    """
    stats = {
        "total_samples": len(dataset.samples),
        "samples_with_ground_truth": 0,
        "samples_with_contexts": 0,
        "samples_with_answers": 0,
        "avg_query_length": 0,
        "avg_answer_length": 0,
        "avg_contexts_per_sample": 0,
        "avg_context_length": 0,
        "metadata_keys": set(),
        "categories": {}
    }
    
    total_query_len = 0
    total_answer_len = 0
    total_contexts = 0
    total_context_len = 0
    
    for sample in dataset.samples:
        # Count presence
        if sample.ground_truth:
            stats["samples_with_ground_truth"] += 1
        if sample.retrieved_docs:
            stats["samples_with_contexts"] += 1
        if sample.generated_answer:
            stats["samples_with_answers"] += 1
        
        # Accumulate lengths
        total_query_len += len(sample.query)
        if sample.generated_answer:
            total_answer_len += len(sample.generated_answer)
        
        # Context stats
        if sample.retrieved_docs:
            total_contexts += len(sample.retrieved_docs)
            for ctx in sample.retrieved_docs:
                total_context_len += len(ctx)
        
        # Metadata
        stats["metadata_keys"].update(sample.metadata.keys())
        
        # Categories
        if "category" in sample.metadata:
            category = sample.metadata["category"]
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
    
    # Calculate averages
    total = len(dataset.samples)
    stats["avg_query_length"] = total_query_len / total if total > 0 else 0
    stats["avg_answer_length"] = (
        total_answer_len / stats["samples_with_answers"]
        if stats["samples_with_answers"] > 0 else 0
    )
    stats["avg_contexts_per_sample"] = total_contexts / total if total > 0 else 0
    stats["avg_context_length"] = (
        total_context_len / total_contexts if total_contexts > 0 else 0
    )
    
    # Convert set to list for JSON serialization
    stats["metadata_keys"] = sorted(list(stats["metadata_keys"]))
    
    return stats
