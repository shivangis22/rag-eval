"""Tests for dataset module."""

import pytest
import json
from pathlib import Path
from rag_eval.dataset import (
    EvaluationSample,
    EvaluationDataset,
    load_dataset,
    save_dataset,
    validate_dataset
)


class TestEvaluationSample:
    """Test EvaluationSample schema."""
    
    def test_valid_sample(self):
        """Test creating a valid sample."""
        sample = EvaluationSample(
            query="What is Python?",
            retrieved_contexts=["Python is a programming language."],
            generated_answer="Python is a high-level programming language.",
            ground_truth="Python is a programming language."
        )
        assert sample.query == "What is Python?"
        assert len(sample.retrieved_contexts) == 1
    
    def test_empty_query_raises_error(self):
        """Test that empty query raises validation error."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            EvaluationSample(
                query="",
                retrieved_contexts=["context"],
                generated_answer="answer"
            )
    
    def test_whitespace_query_raises_error(self):
        """Test that whitespace-only query raises error."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            EvaluationSample(
                query="   ",
                retrieved_contexts=["context"],
                generated_answer="answer"
            )
    
    def test_contexts_cleaned(self):
        """Test that contexts are cleaned of whitespace."""
        sample = EvaluationSample(
            query="test",
            retrieved_contexts=["  context1  ", "", "context2"],
            generated_answer="answer"
        )
        assert len(sample.retrieved_contexts) == 2
        assert sample.retrieved_contexts[0] == "context1"
    
    def test_metadata_optional(self):
        """Test that metadata is optional."""
        sample = EvaluationSample(
            query="test",
            retrieved_contexts=["context"],
            generated_answer="answer"
        )
        assert sample.metadata == {}
    
    def test_metadata_preserved(self):
        """Test that metadata is preserved."""
        metadata = {"category": "test", "difficulty": "easy"}
        sample = EvaluationSample(
            query="test",
            retrieved_contexts=["context"],
            generated_answer="answer",
            metadata=metadata
        )
        assert sample.metadata == metadata


class TestEvaluationDataset:
    """Test EvaluationDataset schema."""
    
    def test_valid_dataset(self):
        """Test creating a valid dataset."""
        samples = [
            EvaluationSample(
                query="test1",
                retrieved_contexts=["context1"],
                generated_answer="answer1"
            ),
            EvaluationSample(
                query="test2",
                retrieved_contexts=["context2"],
                generated_answer="answer2"
            )
        ]
        dataset = EvaluationDataset(samples=samples)
        assert len(dataset) == 2
    
    def test_empty_dataset_raises_error(self):
        """Test that empty dataset raises error."""
        with pytest.raises(ValueError, match="at least one sample"):
            EvaluationDataset(samples=[])
    
    def test_dataset_iteration(self):
        """Test iterating over dataset."""
        samples = [
            EvaluationSample(
                query=f"test{i}",
                retrieved_contexts=[f"context{i}"],
                generated_answer=f"answer{i}"
            )
            for i in range(3)
        ]
        dataset = EvaluationDataset(samples=samples)
        
        count = 0
        for sample in dataset:
            assert isinstance(sample, EvaluationSample)
            count += 1
        assert count == 3
    
    def test_dataset_indexing(self):
        """Test indexing dataset."""
        samples = [
            EvaluationSample(
                query=f"test{i}",
                retrieved_contexts=[f"context{i}"],
                generated_answer=f"answer{i}"
            )
            for i in range(3)
        ]
        dataset = EvaluationDataset(samples=samples)
        
        assert dataset[0].query == "test0"
        assert dataset[1].query == "test1"
        assert dataset[-1].query == "test2"


class TestDatasetLoader:
    """Test dataset loading and saving."""
    
    def test_load_json_dataset(self, tmp_path):
        """Test loading JSON dataset."""
        # Create test dataset
        data = {
            "samples": [
                {
                    "query": "test",
                    "retrieved_contexts": ["context"],
                    "generated_answer": "answer",
                    "ground_truth": "truth"
                }
            ]
        }
        
        # Save to file
        file_path = tmp_path / "test_dataset.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        # Load dataset
        dataset = load_dataset(file_path)
        assert len(dataset) == 1
        assert dataset[0].query == "test"
    
    def test_load_json_list_format(self, tmp_path):
        """Test loading JSON with list format."""
        data = [
            {
                "query": "test1",
                "retrieved_contexts": ["context1"],
                "generated_answer": "answer1"
            },
            {
                "query": "test2",
                "retrieved_contexts": ["context2"],
                "generated_answer": "answer2"
            }
        ]
        
        file_path = tmp_path / "test_dataset.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        dataset = load_dataset(file_path)
        assert len(dataset) == 2
    
    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading JSON dataset."""
        # Create dataset
        samples = [
            EvaluationSample(
                query="test",
                retrieved_contexts=["context"],
                generated_answer="answer",
                ground_truth="truth"
            )
        ]
        dataset = EvaluationDataset(samples=samples)
        
        # Save
        file_path = tmp_path / "output.json"
        save_dataset(dataset, file_path)
        
        # Load
        loaded = load_dataset(file_path)
        assert len(loaded) == len(dataset)
        assert loaded[0].query == dataset[0].query
    
    def test_file_not_found_raises_error(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent.json")


class TestDatasetValidator:
    """Test dataset validation."""
    
    def test_validate_valid_dataset(self):
        """Test validating a valid dataset."""
        samples = [
            EvaluationSample(
                query="test",
                retrieved_contexts=["context"],
                generated_answer="answer",
                ground_truth="truth"
            )
        ]
        dataset = EvaluationDataset(samples=samples)
        
        report = validate_dataset(dataset)
        assert report["valid"] is True
        assert len(report["errors"]) == 0
    
    def test_validate_missing_contexts(self):
        """Test validation with missing contexts."""
        samples = [
            EvaluationSample(
                query="test",
                retrieved_contexts=[],
                generated_answer="answer"
            )
        ]
        dataset = EvaluationDataset(samples=samples)
        
        report = validate_dataset(dataset, require_contexts=True)
        assert report["valid"] is False
        assert len(report["errors"]) > 0
    
    def test_validate_missing_ground_truth(self):
        """Test validation with missing ground truth."""
        samples = [
            EvaluationSample(
                query="test",
                retrieved_contexts=["context"],
                generated_answer="answer"
            )
        ]
        dataset = EvaluationDataset(samples=samples)
        
        report = validate_dataset(dataset, require_ground_truth=True)
        assert report["valid"] is False
    
    def test_validate_strict_mode(self):
        """Test strict validation mode."""
        samples = [
            EvaluationSample(
                query="test",
                retrieved_contexts=[],
                generated_answer="answer"
            )
        ]
        dataset = EvaluationDataset(samples=samples)
        
        with pytest.raises(Exception):
            validate_dataset(dataset, require_contexts=True, strict=True)


@pytest.fixture
def sample_dataset():
    """Fixture providing a sample dataset."""
    samples = [
        EvaluationSample(
            query=f"Query {i}",
            retrieved_contexts=[f"Context {i}"],
            generated_answer=f"Answer {i}",
            ground_truth=f"Truth {i}",
            metadata={"category": "test", "index": i}
        )
        for i in range(5)
    ]
    return EvaluationDataset(samples=samples)


def test_dataset_statistics(sample_dataset):
    """Test computing dataset statistics."""
    from rag_eval.dataset.validator import get_dataset_statistics
    
    stats = get_dataset_statistics(sample_dataset)
    assert stats["total_samples"] == 5
    assert stats["samples_with_ground_truth"] == 5
    assert stats["samples_with_contexts"] == 5
    assert stats["avg_query_length"] > 0

