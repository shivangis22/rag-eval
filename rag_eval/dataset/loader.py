"""Dataset loading and saving utilities."""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Union
import logging

from rag_eval.dataset.schema import EvaluationDataset, EvaluationSample

logger = logging.getLogger(__name__)


def load_dataset(
    path: Union[str, Path],
    format: str = "auto"
) -> EvaluationDataset:
    """Load evaluation dataset from file.
    
    Supports JSON and CSV formats. Format is auto-detected from file extension
    if not specified.
    
    Args:
        path: Path to dataset file
        format: File format ('json', 'csv', or 'auto')
        
    Returns:
        EvaluationDataset object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is unsupported or file is invalid
        
    Example:
        >>> dataset = load_dataset("data/eval_set.json")
        >>> print(f"Loaded {len(dataset)} samples")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    # Auto-detect format
    if format == "auto":
        format = path.suffix.lower().lstrip(".")
        if format not in ["json", "csv"]:
            raise ValueError(f"Cannot auto-detect format for extension: {path.suffix}")
    
    logger.info(f"Loading dataset from {path} (format: {format})")
    
    if format == "json":
        return _load_json(path)
    elif format == "csv":
        return _load_csv(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_json(path: Path) -> EvaluationDataset:
    """Load dataset from JSON file.
    
    Expected format:
    {
        "samples": [
            {
                "query": "...",
                "retrieved_contexts": [...],
                "generated_answer": "...",
                ...
            }
        ],
        "metadata": {...},
        "name": "...",
        "version": "..."
    }
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both wrapped and unwrapped formats
        if isinstance(data, dict) and "samples" in data:
            return EvaluationDataset(**data)
        elif isinstance(data, list):
            # List of samples directly
            return EvaluationDataset(samples=data)
        else:
            raise ValueError("Invalid JSON format: expected dict with 'samples' or list")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading JSON dataset: {e}")


def _load_csv(path: Path) -> EvaluationDataset:
    """Load dataset from CSV file.
    
    Expected columns:
    - query (required)
    - retrieved_contexts (JSON array or pipe-separated)
    - generated_answer
    - ground_truth
    - ground_truth_contexts (JSON array or pipe-separated)
    - Additional columns become metadata
    """
    try:
        samples = []
        
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is 1)
                try:
                    sample_dict = _parse_csv_row(row)
                    samples.append(EvaluationSample(**sample_dict))
                except Exception as e:
                    logger.warning(f"Skipping row {row_num}: {e}")
        
        if not samples:
            raise ValueError("No valid samples found in CSV")
        
        return EvaluationDataset(
            samples=samples,
            metadata={"source": str(path), "format": "csv"}
        )
        
    except Exception as e:
        raise ValueError(f"Error loading CSV dataset: {e}")


def _parse_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Parse a CSV row into sample dictionary."""
    sample_dict: Dict[str, Any] = {
        "query": row.get("query", "").strip(),
        "generated_answer": row.get("generated_answer", "").strip(),
        "ground_truth": row.get("ground_truth", "").strip() or None,
        "metadata": {}
    }
    
    # Parse retrieved_contexts
    contexts_str = row.get("retrieved_contexts", "")
    if contexts_str:
        sample_dict["retrieved_contexts"] = _parse_list_field(contexts_str)
    
    # Parse ground_truth_contexts
    gt_contexts_str = row.get("ground_truth_contexts", "")
    if gt_contexts_str:
        sample_dict["ground_truth_contexts"] = _parse_list_field(gt_contexts_str)
    
    # Add remaining columns as metadata
    standard_fields = {
        "query", "retrieved_contexts", "generated_answer",
        "ground_truth", "ground_truth_contexts"
    }
    for key, value in row.items():
        if key not in standard_fields and value:
            sample_dict["metadata"][key] = value
    
    return sample_dict


def _parse_list_field(value: str) -> List[str]:
    """Parse a list field from CSV (JSON array or pipe-separated)."""
    value = value.strip()
    
    # Try JSON array first
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    
    # Fall back to pipe-separated
    return [item.strip() for item in value.split("|") if item.strip()]


def save_dataset(
    dataset: EvaluationDataset,
    path: Union[str, Path],
    format: str = "auto",
    indent: int = 2
) -> None:
    """Save evaluation dataset to file.
    
    Args:
        dataset: Dataset to save
        path: Output file path
        format: File format ('json', 'csv', or 'auto')
        indent: JSON indentation (for JSON format)
        
    Raises:
        ValueError: If format is unsupported
        
    Example:
        >>> save_dataset(dataset, "output/results.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect format
    if format == "auto":
        format = path.suffix.lower().lstrip(".")
        if format not in ["json", "csv"]:
            format = "json"  # Default to JSON
    
    logger.info(f"Saving dataset to {path} (format: {format})")
    
    if format == "json":
        _save_json(dataset, path, indent)
    elif format == "csv":
        _save_csv(dataset, path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _save_json(dataset: EvaluationDataset, path: Path, indent: int) -> None:
    """Save dataset to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            dataset.model_dump(),
            f,
            indent=indent,
            ensure_ascii=False,
            default=str
        )


def _save_csv(dataset: EvaluationDataset, path: Path) -> None:
    """Save dataset to CSV file."""
    if not dataset.samples:
        raise ValueError("Cannot save empty dataset to CSV")
    
    # Determine all possible metadata keys
    metadata_keys = set()
    for sample in dataset.samples:
        metadata_keys.update(sample.metadata.keys())
    
    fieldnames = [
        "query",
        "retrieved_contexts",
        "generated_answer",
        "ground_truth",
        "ground_truth_contexts"
    ] + sorted(metadata_keys)
    
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for sample in dataset.samples:
            row = {
                "query": sample.query,
                "retrieved_contexts": json.dumps(sample.retrieved_contexts),
                "generated_answer": sample.generated_answer,
                "ground_truth": sample.ground_truth or "",
                "ground_truth_contexts": json.dumps(sample.ground_truth_contexts or []),
            }
            # Add metadata
            row.update(sample.metadata)
            writer.writerow(row)


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load evaluation results from JSON file.
    
    Args:
        path: Path to results file
        
    Returns:
        Results dictionary
        
    Example:
        >>> results = load_results("output/results.json")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(
    results: Dict[str, Any],
    path: Union[str, Path],
    indent: int = 2
) -> None:
    """Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        path: Output file path
        indent: JSON indentation
        
    Example:
        >>> save_results(results, "output/results.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=indent, ensure_ascii=False, default=str)

