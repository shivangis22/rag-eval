"""Comprehensive end-to-end tests for RAG Eval Pro.

Tests all major features and workflows of the library.
"""

import pytest
import json
import asyncio
from pathlib import Path
from typing import List

from rag_eval.dataset.schema import RAGSample, RAGDataset, EvaluationResults
from rag_eval.dataset.loader import load_dataset, save_dataset
from rag_eval.dataset.validator import validate_dataset, get_dataset_statistics
from rag_eval.metrics.retrieval import RecallAtK, PrecisionAtK, NDCG, MRR, F1Score
from rag_eval.metrics.semantic import SemanticSimilarity
from rag_eval.metrics.llm_metrics import FaithfulnessMetric, AnswerRelevanceMetric
from rag_eval.metrics.hallucination import HallucinationDetector, GroundingScore
from rag_eval.pipeline.runner import RAGEvaluator
from rag_eval.pipeline.async_runner import AsyncRAGEvaluator
from rag_eval.pipeline.caching import LLMCache, CachedLLMMetric
from rag_eval.aggregation.stats import (
    compute_statistics,
    compute_confidence_intervals,
    detect_outliers,
    compare_distributions
)
from rag_eval.aggregation.grouping import (
    group_by_metadata,
    cluster_failures,
    identify_failure_patterns
)


class TestDatasetOperations:
    """Test dataset creation, loading, and validation."""
    
    def test_create_rag_sample(self):
        """Test creating a RAG sample."""
        sample = RAGSample(
            query="What is Python?",
            retrieved_docs=["Python is a programming language."],
            generated_answer="Python is a high-level programming language.",
            ground_truth="Python is a programming language.",
            relevant_docs=["Python is a programming language."],
            metadata={"category": "programming", "difficulty": "easy"}
        )
        
        assert sample.query == "What is Python?"
        assert len(sample.retrieved_docs) == 1
        assert sample.metadata["category"] == "programming"
    
    def test_create_rag_dataset(self):
        """Test creating a RAG dataset."""
        samples = [
            RAGSample(
                query=f"Query {i}",
                retrieved_docs=[f"Doc {i}"],
                generated_answer=f"Answer {i}",
                ground_truth=f"Truth {i}",
                relevant_docs=[f"Doc {i}"]
            )
            for i in range(5)
        ]
        
        dataset = RAGDataset(
            name="test_dataset",
            version="1.0",
            samples=samples
        )
        
        assert len(dataset) == 5
        assert dataset.name == "test_dataset"
        assert dataset[0].query == "Query 0"
    
    def test_dataset_validation(self):
        """Test dataset validation."""
        samples = [
            RAGSample(
                query="Test query",
                retrieved_docs=["context"],
                generated_answer="answer",
                ground_truth="truth",
                relevant_docs=["context"]
            )
        ]
        dataset = RAGDataset(samples=samples)
        
        report = validate_dataset(dataset, require_ground_truth=True)
        assert report["valid"] is True
        assert len(report["errors"]) == 0
    
    def test_dataset_statistics(self):
        """Test dataset statistics computation."""
        samples = [
            RAGSample(
                query=f"Query {i}" * 5,
                retrieved_docs=[f"Doc {i}"] * 3,
                generated_answer=f"Answer {i}" * 10,
                ground_truth=f"Truth {i}",
                relevant_docs=[f"Doc {i}"],
                metadata={"category": "test"}
            )
            for i in range(10)
        ]
        dataset = RAGDataset(samples=samples)
        
        stats = get_dataset_statistics(dataset)
        assert stats["total_samples"] == 10
        assert stats["samples_with_ground_truth"] == 10
        assert stats["avg_contexts_per_sample"] == 3.0
    
    def test_save_and_load_dataset(self, tmp_path):
        """Test saving and loading dataset."""
        samples = [
            RAGSample(
                query="Test",
                retrieved_docs=["doc"],
                generated_answer="answer"
            )
        ]
        dataset = RAGDataset(samples=samples, name="test")
        
        # Save
        file_path = tmp_path / "test_dataset.json"
        save_dataset(dataset, file_path)
        
        # Load
        loaded = load_dataset(file_path)
        assert len(loaded) == len(dataset)
        assert loaded.name == dataset.name


class TestRetrievalMetrics:
    """Test retrieval-based metrics."""
    
    @pytest.fixture
    def sample_with_retrieval(self):
        """Sample with retrieval data."""
        return RAGSample(
            query="Test query",
            retrieved_docs=["doc1", "doc2", "doc3", "doc4", "doc5"],
            generated_answer="Test answer",
            ground_truth="Test truth",
            relevant_docs=["doc1", "doc3", "doc6"]
        )
    
    def test_recall_at_k(self, sample_with_retrieval):
        """Test Recall@K metric."""
        metric = RecallAtK(k=5)
        result = metric.compute(sample_with_retrieval)
        
        assert result.score is not None
        assert 0 <= result.score <= 1
        assert result.details["k"] == 5
        assert result.details["hits"] == 2  # doc1 and doc3
    
    def test_precision_at_k(self, sample_with_retrieval):
        """Test Precision@K metric."""
        metric = PrecisionAtK(k=5)
        result = metric.compute(sample_with_retrieval)
        
        assert result.score is not None
        assert result.score == 0.4  # 2 relevant out of 5 retrieved
    
    def test_ndcg(self, sample_with_retrieval):
        """Test NDCG metric."""
        metric = NDCG(k=5)
        result = metric.compute(sample_with_retrieval)
        
        assert result.score is not None
        assert 0 <= result.score <= 1
    
    def test_mrr(self, sample_with_retrieval):
        """Test MRR metric."""
        metric = MRR()
        result = metric.compute(sample_with_retrieval)
        
        assert result.score is not None
        assert result.score == 1.0  # First doc is relevant
    
    def test_f1_score(self, sample_with_retrieval):
        """Test F1 Score metric."""
        metric = F1Score(k=5)
        result = metric.compute(sample_with_retrieval)
        
        assert result.score is not None
        assert 0 <= result.score <= 1


class TestSemanticMetrics:
    """Test semantic similarity metrics."""
    
    @pytest.fixture
    def sample_with_semantics(self):
        """Sample for semantic testing."""
        return RAGSample(
            query="What is machine learning?",
            retrieved_docs=["Machine learning is a subset of AI."],
            generated_answer="Machine learning is a type of artificial intelligence.",
            ground_truth="Machine learning is a subset of artificial intelligence."
        )
    
    @pytest.mark.slow
    def test_semantic_similarity(self, sample_with_semantics):
        """Test semantic similarity metric."""
        metric = SemanticSimilarity(
            model="all-MiniLM-L6-v2",
            compare_to="ground_truth"
        )
        result = metric.compute(sample_with_semantics)
        
        assert result.score is not None
        assert 0 <= result.score <= 1
        assert result.score > 0.7  # Should be high similarity


class TestHallucinationDetection:
    """Test hallucination detection metrics."""
    
    @pytest.fixture
    def sample_with_hallucination(self):
        """Sample with potential hallucination."""
        return RAGSample(
            query="What is the capital of France?",
            retrieved_docs=["Paris is the capital of France."],
            generated_answer="The capital of France is Paris, and it has a population of 10 million people.",
            ground_truth="Paris"
        )
    
    def test_keyword_hallucination_detection(self, sample_with_hallucination):
        """Test keyword-based hallucination detection."""
        detector = HallucinationDetector(method="keyword")
        result = detector.compute(sample_with_hallucination)
        
        assert result.score is not None
        assert 0 <= result.score <= 1
        # Should detect some hallucination (population claim)
    
    def test_grounding_score(self, sample_with_hallucination):
        """Test grounding score metric."""
        metric = GroundingScore()
        result = metric.compute(sample_with_hallucination)
        
        assert result.score is not None
        assert 0 <= result.score <= 1


class TestEvaluationPipeline:
    """Test evaluation pipeline."""
    
    @pytest.fixture
    def test_dataset(self):
        """Create test dataset."""
        samples = [
            RAGSample(
                query=f"Query {i}",
                retrieved_docs=[f"Doc {i}-1", f"Doc {i}-2"],
                generated_answer=f"Answer {i}",
                ground_truth=f"Truth {i}",
                relevant_docs=[f"Doc {i}-1"],
                metadata={"category": "test", "index": i}
            )
            for i in range(10)
        ]
        return RAGDataset(name="test", samples=samples)
    
    def test_sync_evaluation(self, test_dataset):
        """Test synchronous evaluation."""
        metrics = [
            RecallAtK(k=2),
            PrecisionAtK(k=2),
        ]
        
        evaluator = RAGEvaluator(metrics=metrics)
        results = evaluator.evaluate(test_dataset)
        
        assert isinstance(results, EvaluationResults)
        assert len(results.results) == 10
        assert len(results.aggregated_metrics) == 2
        assert "Recall@2" in results.aggregated_metrics
        assert "Precision@2" in results.aggregated_metrics
    
    @pytest.mark.asyncio
    async def test_async_evaluation(self, test_dataset):
        """Test asynchronous evaluation."""
        metrics = [
            RecallAtK(k=2),
            PrecisionAtK(k=2),
        ]
        
        evaluator = AsyncRAGEvaluator(
            metrics=metrics,
            max_concurrent=5
        )
        results = await evaluator.evaluate_async(test_dataset)
        
        assert isinstance(results, EvaluationResults)
        assert len(results.results) == 10
    
    def test_evaluation_with_threshold(self, test_dataset):
        """Test evaluation with threshold checking."""
        metrics = [RecallAtK(k=2, threshold=0.5)]
        evaluator = RAGEvaluator(metrics=metrics)
        
        thresholds = {"Recall@2": 0.3}
        results, passed = evaluator.evaluate_with_threshold(
            test_dataset,
            thresholds
        )
        
        assert isinstance(results, EvaluationResults)
        assert isinstance(passed, bool)
    
    def test_failed_samples_identification(self, test_dataset):
        """Test identifying failed samples."""
        metrics = [RecallAtK(k=2)]
        evaluator = RAGEvaluator(metrics=metrics)
        results = evaluator.evaluate(test_dataset)
        
        failed = evaluator.get_failed_samples(
            results,
            "Recall@2",
            threshold=0.9
        )
        
        assert isinstance(failed, list)


class TestCaching:
    """Test caching functionality."""
    
    def test_memory_cache(self):
        """Test memory-based cache."""
        cache = LLMCache(backend="memory", ttl=3600)
        
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        
        assert value == "test_value"
        assert cache.size() == 1
    
    def test_file_cache(self, tmp_path):
        """Test file-based cache."""
        cache = LLMCache(
            backend="file",
            cache_dir=str(tmp_path / "cache"),
            ttl=3600
        )
        
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        
        assert value == "test_value"
        assert cache.size() == 1
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = LLMCache(backend="memory", ttl=1)  # 1 second TTL
        
        cache.set("test_key", "test_value")
        
        import time
        time.sleep(2)  # Wait for expiration
        
        value = cache.get("test_key")
        assert value is None
    
    def test_cached_metric(self):
        """Test cached metric wrapper."""
        cache = LLMCache(backend="memory")
        metric = RecallAtK(k=5)
        cached_metric = CachedLLMMetric(metric, cache)
        
        sample = RAGSample(
            query="test",
            retrieved_docs=["doc1"],
            generated_answer="answer",
            relevant_docs=["doc1"]
        )
        
        # First call - cache miss
        result1 = cached_metric.compute(sample)
        assert cached_metric.cache_misses == 1
        
        # Second call - cache hit
        result2 = cached_metric.compute(sample)
        assert cached_metric.cache_hits == 1
        
        assert result1.score == result2.score


class TestStatisticalAnalysis:
    """Test statistical analysis functions."""
    
    def test_compute_statistics(self):
        """Test statistics computation."""
        scores = [0.7, 0.8, 0.85, 0.9, 0.75, 0.82, 0.88]
        stats = compute_statistics(scores)
        
        assert "mean" in stats
        assert "std" in stats
        assert "median" in stats
        assert "p95" in stats
        assert stats["count"] == 7
        assert 0 < stats["mean"] < 1
    
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        scores = [0.7, 0.8, 0.85, 0.9, 0.75, 0.82, 0.88]
        ci = compute_confidence_intervals(scores, confidence=0.95)
        
        assert "mean" in ci
        assert "median" in ci
        assert len(ci["mean"]) == 2  # (lower, upper)
        assert ci["mean"][0] < ci["mean"][1]
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        scores = [0.8, 0.85, 0.9, 0.75, 0.88, 0.1, 0.95]  # 0.1 is outlier
        outliers = detect_outliers(scores, method="iqr")
        
        assert "outlier_indices" in outliers
        assert "num_outliers" in outliers
        assert outliers["num_outliers"] > 0
    
    def test_distribution_comparison(self):
        """Test distribution comparison."""
        scores1 = [0.7, 0.75, 0.8, 0.85, 0.9]
        scores2 = [0.72, 0.78, 0.83, 0.88, 0.92]
        
        result = compare_distributions(scores1, scores2, test="ttest")
        
        assert "p_value" in result
        assert "statistic" in result
        assert "significant" in result
        assert isinstance(result["significant"], bool)


class TestFailureAnalysis:
    """Test failure analysis and grouping."""
    
    @pytest.fixture
    def results_with_failures(self):
        """Create results with some failures."""
        from rag_eval.dataset.schema import SampleResult, MetricResult
        
        results = []
        for i in range(20):
            sample = RAGSample(
                query=f"Query {i}",
                retrieved_docs=[f"Doc {i}"],
                generated_answer=f"Answer {i}",
                metadata={
                    "category": "easy" if i < 10 else "hard",
                    "index": i
                }
            )
            
            # Create varying scores
            score = 0.9 if i < 10 else 0.6
            
            results.append(SampleResult(
                sample=sample,
                metrics={
                    "test_metric": MetricResult(
                        metric_name="test_metric",
                        score=score
                    )
                },
                overall_score=score
            ))
        
        return results
    
    def test_group_by_metadata(self, results_with_failures):
        """Test grouping by metadata."""
        grouped = group_by_metadata(results_with_failures, "category")
        
        assert "easy" in grouped
        assert "hard" in grouped
        assert len(grouped["easy"]) == 10
        assert len(grouped["hard"]) == 10
    
    def test_cluster_failures(self, results_with_failures):
        """Test failure clustering."""
        clusters = cluster_failures(
            results_with_failures,
            "test_metric",
            threshold=0.8,
            n_clusters=3
        )
        
        assert isinstance(clusters, list)
        if clusters:
            assert "size" in clusters[0]
            assert "samples" in clusters[0]
    
    def test_identify_failure_patterns(self, results_with_failures):
        """Test failure pattern identification."""
        patterns = identify_failure_patterns(
            results_with_failures,
            min_support=3
        )
        
        assert isinstance(patterns, list)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_complete_evaluation_workflow(self, tmp_path):
        """Test complete evaluation workflow from dataset to results."""
        # 1. Create dataset
        samples = [
            RAGSample(
                query=f"What is topic {i}?",
                retrieved_docs=[f"Topic {i} is about X.", f"More info on {i}."],
                generated_answer=f"Topic {i} is about X and Y.",
                ground_truth=f"Topic {i} is about X.",
                relevant_docs=[f"Topic {i} is about X."],
                metadata={"category": "test", "difficulty": "easy"}
            )
            for i in range(5)
        ]
        dataset = RAGDataset(name="test_workflow", samples=samples)
        
        # 2. Save dataset
        dataset_path = tmp_path / "dataset.json"
        save_dataset(dataset, dataset_path)
        
        # 3. Load dataset
        loaded_dataset = load_dataset(dataset_path)
        assert len(loaded_dataset) == 5
        
        # 4. Validate dataset
        validation_report = validate_dataset(loaded_dataset)
        assert validation_report["valid"]
        
        # 5. Create evaluator with multiple metrics
        metrics = [
            RecallAtK(k=2),
            PrecisionAtK(k=2),
            F1Score(k=2)
        ]
        evaluator = RAGEvaluator(metrics=metrics, verbose=False)
        
        # 6. Run evaluation
        results = evaluator.evaluate(loaded_dataset)
        
        # 7. Verify results
        assert len(results.results) == 5
        assert len(results.aggregated_metrics) == 3
        
        # 8. Compute statistics
        for metric_name in results.aggregated_metrics:
            scores = results.get_metric_scores(metric_name)
            stats = compute_statistics(scores)
            assert stats["count"] == 5
        
        # 9. Group by metadata
        grouped = group_by_metadata(results.results, "category")
        assert "test" in grouped
        
        # 10. Save results
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(results.to_dict(), f, default=str)
        
        # 11. Verify saved results
        assert results_path.exists()
        with open(results_path) as f:
            loaded_results = json.load(f)
        assert loaded_results["dataset_name"] == "test_workflow"
    
    @pytest.mark.asyncio
    async def test_async_workflow_with_caching(self, tmp_path):
        """Test async evaluation with caching."""
        # Create dataset
        samples = [
            RAGSample(
                query=f"Query {i}",
                retrieved_docs=[f"Doc {i}"],
                generated_answer=f"Answer {i}",
                relevant_docs=[f"Doc {i}"]
            )
            for i in range(10)
        ]
        dataset = RAGDataset(samples=samples)
        
        # Create cache
        cache = LLMCache(
            backend="file",
            cache_dir=str(tmp_path / "cache")
        )
        
        # Create metrics with caching
        base_metric = RecallAtK(k=1)
        cached_metric = CachedLLMMetric(base_metric, cache)
        
        # Create async evaluator
        evaluator = AsyncRAGEvaluator(
            metrics=[cached_metric],
            max_concurrent=5
        )
        
        # First evaluation - populate cache
        results1 = await evaluator.evaluate_async(dataset)
        assert len(results1.results) == 10
        
        # Second evaluation - use cache
        results2 = await evaluator.evaluate_async(dataset)
        assert len(results2.results) == 10
        
        # Verify cache was used
        assert cached_metric.cache_hits > 0


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

