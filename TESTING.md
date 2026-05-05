# Testing Guide for RAG Eval Pro

This guide provides quick instructions for running and understanding the test suite.

## Quick Start

### 1. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### 2. Run All Tests

```bash
pytest
```

### 3. Run with Coverage

```bash
pytest --cov=rag_eval --cov-report=html --cov-report=term
```

View the HTML report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Test Organization

### End-to-End Tests (`tests/test_end_to_end.py`)

Comprehensive tests covering all library features:

```bash
# Run all E2E tests
pytest tests/test_end_to_end.py -v

# Run specific test class
pytest tests/test_end_to_end.py::TestDatasetOperations -v

# Run specific test
pytest tests/test_end_to_end.py::TestDatasetOperations::test_create_rag_sample -v
```

**Test Coverage:**
- ✅ Dataset operations (create, load, save, validate)
- ✅ Retrieval metrics (Recall@K, Precision@K, NDCG, MRR, F1)
- ✅ Semantic similarity metrics
- ✅ Hallucination detection
- ✅ Sync and async evaluation pipelines
- ✅ Caching (memory, file, Redis)
- ✅ Statistical analysis
- ✅ Failure analysis and grouping
- ✅ Complete workflows

### Unit Tests

```bash
# Dataset tests
pytest tests/test_dataset.py -v

# Metrics tests
pytest tests/test_metrics.py -v

# Pipeline tests
pytest tests/test_pipeline.py -v
```

## Test Markers

### Skip Slow Tests

Some tests load ML models and take longer:

```bash
# Skip slow tests (recommended for quick iteration)
pytest -m "not slow"

# Run only slow tests
pytest -m slow
```

### Run Only Integration Tests

```bash
pytest -m integration
```

### Run Async Tests

```bash
pytest -m asyncio
```

## Common Test Scenarios

### 1. Quick Validation (Fast Tests Only)

```bash
pytest -m "not slow" --tb=short
```

**Use case:** Quick validation during development

### 2. Full Test Suite

```bash
pytest -v --tb=short
```

**Use case:** Before committing code

### 3. Coverage Report

```bash
pytest --cov=rag_eval --cov-report=term-missing
```

**Use case:** Identify untested code

### 4. Parallel Execution

```bash
pip install pytest-xdist
pytest -n auto
```

**Use case:** Speed up test execution

### 5. Debug Mode

```bash
# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Verbose logging
pytest --log-cli-level=DEBUG
```

**Use case:** Debugging test failures

## Test Examples

### Example 1: Testing Dataset Creation

```python
from rag_eval.dataset.schema import RAGSample, RAGDataset

def test_create_dataset():
    samples = [
        RAGSample(
            query="What is Python?",
            retrieved_docs=["Python is a programming language."],
            generated_answer="Python is a high-level language.",
            ground_truth="Python is a programming language.",
            relevant_docs=["Python is a programming language."]
        )
    ]
    dataset = RAGDataset(samples=samples)
    assert len(dataset) == 1
```

### Example 2: Testing Metrics

```python
from rag_eval.metrics.retrieval import RecallAtK

def test_recall_metric():
    metric = RecallAtK(k=5)
    sample = RAGSample(
        query="test",
        retrieved_docs=["doc1", "doc2", "doc3"],
        relevant_docs=["doc1", "doc4"]
    )
    result = metric.compute(sample)
    assert result.score == 0.5  # 1 out of 2 relevant docs retrieved
```

### Example 3: Testing Evaluation Pipeline

```python
from rag_eval.pipeline.runner import RAGEvaluator
from rag_eval.metrics.retrieval import RecallAtK

def test_evaluation():
    metrics = [RecallAtK(k=5)]
    evaluator = RAGEvaluator(metrics=metrics)
    results = evaluator.evaluate(dataset)
    assert len(results.results) == len(dataset)
```

## CI/CD Integration

Tests run automatically on GitHub Actions:

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: |
    pytest --cov=rag_eval --cov-report=xml
```

### Local CI Simulation

```bash
# Run the same tests as CI
pytest --cov=rag_eval --cov-report=xml --cov-report=term
```

## Performance Testing

### Show Slowest Tests

```bash
pytest --durations=10
```

### Profile Tests

```bash
pip install pytest-profiling
pytest --profile
```

## Test Data

Sample datasets for testing:

```bash
datasets/
├── sample_dataset.json      # Basic test dataset
└── golden_dataset.json      # Reference dataset
```

Load in tests:

```python
from rag_eval.dataset.loader import load_dataset

dataset = load_dataset("datasets/sample_dataset.json")
```

## Troubleshooting

### Issue: Import Errors

**Solution:**
```bash
pip install -e .
```

### Issue: Missing Optional Dependencies

**Solution:**
```bash
# For semantic metrics
pip install sentence-transformers

# For hallucination detection
pip install transformers

# For statistical analysis
pip install scipy
```

### Issue: Async Tests Failing

**Solution:**
```bash
pip install pytest-asyncio
```

### Issue: Tests Hanging

**Solution:**
```bash
# Add timeout
pip install pytest-timeout
pytest --timeout=300
```

## Best Practices

### 1. Run Fast Tests During Development

```bash
pytest -m "not slow" --tb=short
```

### 2. Run Full Suite Before Committing

```bash
pytest -v
```

### 3. Check Coverage Regularly

```bash
pytest --cov=rag_eval --cov-report=term-missing
```

### 4. Use Fixtures for Common Setup

```python
@pytest.fixture
def sample_dataset():
    return RAGDataset(samples=[...])

def test_something(sample_dataset):
    # Use sample_dataset
    pass
```

### 5. Parametrize for Multiple Cases

```python
@pytest.mark.parametrize("k,expected", [
    (1, 0.5),
    (2, 0.75),
    (5, 1.0)
])
def test_recall_at_k(k, expected):
    # Test with different k values
    pass
```

## Coverage Goals

| Module | Target Coverage |
|--------|----------------|
| `rag_eval/dataset/` | > 90% |
| `rag_eval/metrics/` | > 90% |
| `rag_eval/pipeline/` | > 85% |
| `rag_eval/aggregation/` | > 85% |
| `rag_eval/reporting/` | > 80% |
| **Overall** | **> 80%** |

## Continuous Testing

### Watch Mode (with pytest-watch)

```bash
pip install pytest-watch
ptw
```

Automatically runs tests when files change.

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
pytest -m "not slow" --tb=short
```

## Resources

- **Test Documentation**: `tests/README.md`
- **Contributing Guide**: `CONTRIBUTING.md`
- **CI Configuration**: `.github/workflows/ci.yml`
- **Pytest Configuration**: `pytest.ini`

## Getting Help

If you encounter issues:

1. Check `tests/README.md` for detailed test documentation
2. Review `CONTRIBUTING.md` for contribution guidelines
3. Open an issue on GitHub with test output

## Summary

```bash
# Quick validation
pytest -m "not slow" --tb=short

# Full test suite
pytest -v

# With coverage
pytest --cov=rag_eval --cov-report=html

# Parallel execution
pytest -n auto

# Debug mode
pytest --pdb -s
```

Happy testing! 🧪