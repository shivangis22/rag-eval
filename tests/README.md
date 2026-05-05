# RAG Eval Pro - Test Suite

This directory contains comprehensive tests for the RAG Eval Pro library.

## Test Structure

```
tests/
├── README.md                    # This file
├── test_end_to_end.py          # Comprehensive E2E tests
├── test_metrics.py             # Unit tests for metrics
├── test_pipeline.py            # Pipeline tests
└── test_dataset.py             # Dataset tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
# Run only end-to-end tests
pytest tests/test_end_to_end.py

# Run only metric tests
pytest tests/test_metrics.py
```

### Run Specific Test Classes

```bash
# Run dataset operation tests
pytest tests/test_end_to_end.py::TestDatasetOperations

# Run retrieval metric tests
pytest tests/test_end_to_end.py::TestRetrievalMetrics
```

### Run Specific Test Functions

```bash
pytest tests/test_end_to_end.py::TestDatasetOperations::test_create_rag_sample
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Coverage

```bash
pytest --cov=rag_eval --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run Tests in Parallel

```bash
pip install pytest-xdist
pytest -n auto
```

## Test Categories

### 1. Dataset Operations (`TestDatasetOperations`)

Tests for dataset creation, loading, validation, and statistics:

- `test_create_rag_sample`: Creating individual RAG samples
- `test_create_rag_dataset`: Creating datasets from samples
- `test_dataset_validation`: Validating dataset integrity
- `test_dataset_statistics`: Computing dataset statistics
- `test_save_and_load_dataset`: Serialization and deserialization

**Run:**
```bash
pytest tests/test_end_to_end.py::TestDatasetOperations -v
```

### 2. Retrieval Metrics (`TestRetrievalMetrics`)

Tests for retrieval-based evaluation metrics:

- `test_recall_at_k`: Recall@K metric
- `test_precision_at_k`: Precision@K metric
- `test_ndcg`: Normalized Discounted Cumulative Gain
- `test_mrr`: Mean Reciprocal Rank
- `test_f1_score`: F1 Score metric

**Run:**
```bash
pytest tests/test_end_to_end.py::TestRetrievalMetrics -v
```

### 3. Semantic Metrics (`TestSemanticMetrics`)

Tests for semantic similarity evaluation:

- `test_semantic_similarity`: Embedding-based similarity

**Note:** These tests are marked as `slow` because they load ML models.

**Run:**
```bash
pytest tests/test_end_to_end.py::TestSemanticMetrics -v
```

**Skip slow tests:**
```bash
pytest -m "not slow"
```

### 4. Hallucination Detection (`TestHallucinationDetection`)

Tests for hallucination detection metrics:

- `test_keyword_hallucination_detection`: Keyword-based detection
- `test_grounding_score`: Context grounding evaluation

**Run:**
```bash
pytest tests/test_end_to_end.py::TestHallucinationDetection -v
```

### 5. Evaluation Pipeline (`TestEvaluationPipeline`)

Tests for the evaluation orchestration:

- `test_sync_evaluation`: Synchronous evaluation
- `test_async_evaluation`: Asynchronous evaluation
- `test_evaluation_with_threshold`: Threshold-based pass/fail
- `test_failed_samples_identification`: Identifying failed samples

**Run:**
```bash
pytest tests/test_end_to_end.py::TestEvaluationPipeline -v
```

### 6. Caching (`TestCaching`)

Tests for LLM response caching:

- `test_memory_cache`: In-memory cache
- `test_file_cache`: File-based cache
- `test_cache_expiration`: TTL expiration
- `test_cached_metric`: Cached metric wrapper

**Run:**
```bash
pytest tests/test_end_to_end.py::TestCaching -v
```

### 7. Statistical Analysis (`TestStatisticalAnalysis`)

Tests for statistical computations:

- `test_compute_statistics`: Mean, std, median, percentiles
- `test_confidence_intervals`: Confidence interval computation
- `test_outlier_detection`: Outlier identification
- `test_distribution_comparison`: Statistical tests

**Run:**
```bash
pytest tests/test_end_to_end.py::TestStatisticalAnalysis -v
```

### 8. Failure Analysis (`TestFailureAnalysis`)

Tests for failure pattern identification:

- `test_group_by_metadata`: Grouping results by metadata
- `test_cluster_failures`: Clustering similar failures
- `test_identify_failure_patterns`: Pattern identification

**Run:**
```bash
pytest tests/test_end_to_end.py::TestFailureAnalysis -v
```

### 9. End-to-End Workflows (`TestEndToEndWorkflow`)

Complete workflow tests:

- `test_complete_evaluation_workflow`: Full evaluation pipeline
- `test_async_workflow_with_caching`: Async evaluation with caching

**Run:**
```bash
pytest tests/test_end_to_end.py::TestEndToEndWorkflow -v
```

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.slow`: Tests that take longer to run (e.g., loading ML models)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.asyncio`: Async tests

### Run Only Fast Tests

```bash
pytest -m "not slow"
```

### Run Only Integration Tests

```bash
pytest -m integration
```

## Continuous Integration

Tests are automatically run on every push via GitHub Actions. See `.github/workflows/ci.yml`.

## Test Coverage Goals

- **Overall Coverage**: > 80%
- **Core Modules**: > 90%
  - `rag_eval/dataset/`
  - `rag_eval/metrics/`
  - `rag_eval/pipeline/`

## Writing New Tests

### Test Structure

```python
import pytest
from rag_eval.dataset.schema import RAGSample

class TestMyFeature:
    """Test my new feature."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture for test data."""
        return RAGSample(
            query="test",
            retrieved_docs=["doc"],
            generated_answer="answer"
        )
    
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        # Arrange
        expected = "result"
        
        # Act
        result = my_function(sample_data)
        
        # Assert
        assert result == expected
```

### Best Practices

1. **Use descriptive test names**: `test_recall_at_k_with_empty_relevant_docs`
2. **Use fixtures for common setup**: Avoid code duplication
3. **Test edge cases**: Empty inputs, None values, invalid data
4. **Test error handling**: Verify exceptions are raised correctly
5. **Use parametrize for multiple cases**:
   ```python
   @pytest.mark.parametrize("k,expected", [(1, 0.5), (2, 0.75)])
   def test_recall_at_k(self, k, expected):
       ...
   ```

## Troubleshooting

### Import Errors

If you see import errors, ensure you've installed the package:

```bash
pip install -e .
```

### Missing Dependencies

Some tests require optional dependencies:

```bash
# For semantic metrics
pip install sentence-transformers

# For hallucination detection
pip install transformers

# For statistical analysis
pip install scipy

# For caching
pip install redis
```

### Async Test Failures

Ensure `pytest-asyncio` is installed:

```bash
pip install pytest-asyncio
```

### Slow Tests

Skip slow tests during development:

```bash
pytest -m "not slow"
```

## Test Data

Sample test datasets are available in `datasets/`:

- `sample_dataset.json`: Basic test dataset
- `golden_dataset.json`: Golden reference dataset

## Performance Testing

For performance testing, use:

```bash
pytest --durations=10
```

This shows the 10 slowest tests.

## Debugging Tests

### Run with PDB

```bash
pytest --pdb
```

Drops into debugger on failure.

### Print Output

```bash
pytest -s
```

Shows print statements.

### Verbose Logging

```bash
pytest --log-cli-level=DEBUG
```

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure all tests pass
3. Maintain > 80% coverage
4. Update this README if adding new test categories

## Questions?

See `CONTRIBUTING.md` for more information on contributing to the project.