"""Tests for optional provider and backend behavior."""

from rag_eval.dataset.schema import RAGSample
from rag_eval.metrics.llm_metrics import FaithfulnessMetric
from rag_eval.metrics.semantic import SemanticSimilarity


def test_semantic_similarity_lexical_backend():
    """Lexical backend should work offline without model downloads."""
    sample = RAGSample(
        query="What is machine learning?",
        retrieved_docs=["Machine learning is a subset of AI."],
        generated_answer="Machine learning is a type of AI.",
        ground_truth="Machine learning is a subset of artificial intelligence.",
    )

    metric = SemanticSimilarity(
        backend="lexical",
        compare_to="ground_truth",
    )
    result = metric.compute(sample)

    assert result.score is not None
    assert 0 <= result.score <= 1
    assert result.details["fallback"] == "lexical_jaccard"


def test_llm_metric_mock_provider():
    """Mock provider should support deterministic tests and examples."""
    sample = RAGSample(
        query="What is Python?",
        retrieved_docs=["Python is a programming language."],
        generated_answer="Python is a programming language.",
    )

    metric = FaithfulnessMetric(provider="mock", mock_response="0.9")
    result = metric.compute(sample)

    assert result.score == 0.9
    assert result.details["provider"] == "mock"


def test_llm_metric_missing_openai_key_returns_clear_error(monkeypatch):
    """Provider-backed metrics should fail clearly when credentials are missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    sample = RAGSample(
        query="What is Python?",
        retrieved_docs=["Python is a programming language."],
        generated_answer="Python is a programming language.",
    )

    metric = FaithfulnessMetric(provider="openai")
    result = metric.compute(sample)

    assert result.score is None
    assert "OPENAI_API_KEY" in result.error
