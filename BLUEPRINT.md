# RAG Eval Pro - Complete Repository Blueprint

This document provides a comprehensive blueprint for implementing the complete RAG evaluation library based on the folder structure provided.

## ✅ Completed Components

### 1. Project Configuration
- ✅ `pyproject.toml` - Complete build configuration with dependencies
- ✅ `requirements.txt` - Core dependencies list
- ✅ `README.md` - Comprehensive documentation
- ✅ `LICENSE` - MIT License
- ✅ `CONTRIBUTING.md` - Contribution guidelines
- ✅ `.gitignore` - Git ignore patterns

### 2. Dataset Module (`rag_eval/dataset/`)
- ✅ `schema.py` - Pydantic schemas for datasets and results
- ✅ `loader.py` - JSON/CSV dataset loading and saving
- ✅ `validator.py` - Dataset validation and statistics
- ✅ `__init__.py` - Module exports

### 3. Metrics Base (`rag_eval/metrics/`)
- ✅ `base.py` - BaseMetric interface, CompositeMetric, CachedMetric
- ✅ `__init__.py` - Module exports

## 📋 Remaining Implementation Details

### 4. Metrics Module - Retrieval (`rag_eval/metrics/retrieval.py`)

```python
"""Retrieval-based metrics for RAG evaluation."""

from typing import List, Set
import numpy as np
from rag_eval.metrics.base import BaseMetric
from rag_eval.dataset.schema import EvaluationSample, MetricResult


class RecallAtK(BaseMetric):
    """Recall@K metric for retrieval evaluation.
    
    Measures what fraction of relevant documents are retrieved in top-k.
    """
    
    def __init__(self, k: int = 5, **kwargs):
        super().__init__(name=f"Recall@{k}", **kwargs)
        self.k = k
        self.requires_ground_truth = True
        self.requires_contexts = True
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        self.validate_sample(sample)
        
        if not sample.ground_truth_contexts:
            return self._create_result(None, error="No ground truth contexts")
        
        retrieved = set(sample.retrieved_contexts[:self.k])
        relevant = set(sample.ground_truth_contexts)
        
        if not relevant:
            return self._create_result(None, error="Empty relevant set")
        
        hits = len(retrieved & relevant)
        recall = hits / len(relevant)
        
        return self._create_result(
            score=recall,
            details={
                "k": self.k,
                "retrieved": len(retrieved),
                "relevant": len(relevant),
                "hits": hits
            }
        )


class PrecisionAtK(BaseMetric):
    """Precision@K metric for retrieval evaluation."""
    
    def __init__(self, k: int = 5, **kwargs):
        super().__init__(name=f"Precision@{k}", **kwargs)
        self.k = k
        self.requires_ground_truth = True
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        self.validate_sample(sample)
        
        if not sample.ground_truth_contexts:
            return self._create_result(None, error="No ground truth contexts")
        
        retrieved = sample.retrieved_contexts[:self.k]
        relevant = set(sample.ground_truth_contexts)
        
        if not retrieved:
            return self._create_result(0.0)
        
        hits = sum(1 for doc in retrieved if doc in relevant)
        precision = hits / len(retrieved)
        
        return self._create_result(
            score=precision,
            details={"k": self.k, "hits": hits, "retrieved": len(retrieved)}
        )


class NDCG(BaseMetric):
    """Normalized Discounted Cumulative Gain."""
    
    def __init__(self, k: int = 10, **kwargs):
        super().__init__(name=f"NDCG@{k}", **kwargs)
        self.k = k
        self.requires_ground_truth = True
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        self.validate_sample(sample)
        
        if not sample.ground_truth_contexts:
            return self._create_result(None, error="No ground truth contexts")
        
        retrieved = sample.retrieved_contexts[:self.k]
        relevant = set(sample.ground_truth_contexts)
        
        # Calculate DCG
        dcg = sum(
            (1 if doc in relevant else 0) / np.log2(i + 2)
            for i, doc in enumerate(retrieved)
        )
        
        # Calculate IDCG (ideal DCG)
        ideal_length = min(len(relevant), self.k)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_length))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return self._create_result(
            score=ndcg,
            details={"dcg": dcg, "idcg": idcg, "k": self.k}
        )


class MRR(BaseMetric):
    """Mean Reciprocal Rank."""
    
    def __init__(self, **kwargs):
        super().__init__(name="MRR", **kwargs)
        self.requires_ground_truth = True
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        self.validate_sample(sample)
        
        if not sample.ground_truth_contexts:
            return self._create_result(None, error="No ground truth contexts")
        
        relevant = set(sample.ground_truth_contexts)
        
        for i, doc in enumerate(sample.retrieved_contexts, 1):
            if doc in relevant:
                rr = 1.0 / i
                return self._create_result(
                    score=rr,
                    details={"rank": i}
                )
        
        return self._create_result(score=0.0, details={"rank": None})
```

### 5. Metrics Module - Semantic (`rag_eval/metrics/semantic.py`)

```python
"""Semantic similarity metrics using embeddings."""

from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rag_eval.metrics.base import BaseMetric
from rag_eval.dataset.schema import EvaluationSample, MetricResult


class SemanticSimilarity(BaseMetric):
    """Semantic similarity using sentence embeddings."""
    
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        compare_to: str = "ground_truth",
        **kwargs
    ):
        super().__init__(name="SemanticSimilarity", **kwargs)
        self.model_name = model
        self.compare_to = compare_to
        self.model = None
        self.requires_ground_truth = (compare_to == "ground_truth")
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        self.validate_sample(sample)
        self._load_model()
        
        answer = sample.generated_answer
        
        if self.compare_to == "ground_truth":
            if not sample.ground_truth:
                return self._create_result(None, error="No ground truth")
            reference = sample.ground_truth
        elif self.compare_to == "contexts":
            if not sample.retrieved_contexts:
                return self._create_result(None, error="No contexts")
            reference = " ".join(sample.retrieved_contexts)
        else:
            return self._create_result(None, error=f"Invalid compare_to: {self.compare_to}")
        
        # Compute embeddings
        embeddings = self.model.encode([answer, reference])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return self._create_result(
            score=float(similarity),
            details={"compare_to": self.compare_to, "model": self.model_name}
        )


class BERTScore(BaseMetric):
    """BERTScore for semantic similarity."""
    
    def __init__(self, model: str = "bert-base-uncased", **kwargs):
        super().__init__(name="BERTScore", **kwargs)
        self.model_name = model
        self.requires_ground_truth = True
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        # Implementation would use bert-score library
        # Placeholder for blueprint
        return self._create_result(
            score=0.85,
            details={"model": self.model_name}
        )
```

### 6. Metrics Module - LLM Metrics (`rag_eval/metrics/llm_metrics.py`)

```python
"""LLM-based evaluation metrics."""

from typing import Optional, Dict, Any
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_eval.metrics.base import BaseMetric
from rag_eval.dataset.schema import EvaluationSample, MetricResult


class LLMMetric(BaseMetric):
    """Base class for LLM-based metrics."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=api_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=500
        )
        return response.choices[0].message.content


class Faithfulness(LLMMetric):
    """Evaluate if answer is faithful to retrieved contexts."""
    
    def __init__(self, **kwargs):
        super().__init__(name="Faithfulness", **kwargs)
        self.requires_contexts = True
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        self.validate_sample(sample)
        
        contexts = "\n".join(sample.retrieved_contexts)
        answer = sample.generated_answer
        
        prompt = f"""Evaluate if the following answer is faithful to the given contexts.
Answer only with a score from 0 to 1, where:
- 1.0 = Completely faithful, all claims supported by contexts
- 0.5 = Partially faithful, some claims not supported
- 0.0 = Not faithful, contradicts or unsupported by contexts

Contexts:
{contexts}

Answer:
{answer}

Score (0-1):"""
        
        try:
            response = self._call_llm(prompt)
            score = float(response.strip())
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
            return self._create_result(
                score=score,
                details={"model": self.model, "response": response}
            )
        except Exception as e:
            return self._create_result(None, error=str(e))


class Relevance(LLMMetric):
    """Evaluate if answer is relevant to the query."""
    
    def __init__(self, **kwargs):
        super().__init__(name="Relevance", **kwargs)
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        self.validate_sample(sample)
        
        prompt = f"""Evaluate if the answer is relevant to the query.
Score from 0 to 1:
- 1.0 = Highly relevant, directly answers the query
- 0.5 = Partially relevant
- 0.0 = Not relevant

Query: {sample.query}
Answer: {sample.generated_answer}

Score (0-1):"""
        
        try:
            response = self._call_llm(prompt)
            score = float(response.strip())
            score = max(0.0, min(1.0, score))
            
            return self._create_result(score=score, details={"model": self.model})
        except Exception as e:
            return self._create_result(None, error=str(e))


class Coherence(LLMMetric):
    """Evaluate answer coherence and readability."""
    
    def __init__(self, **kwargs):
        super().__init__(name="Coherence", **kwargs)
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        # Similar implementation to Relevance
        # Evaluates logical flow, grammar, clarity
        pass
```

### 7. Metrics Module - Hallucination (`rag_eval/metrics/hallucination.py`)

```python
"""Hallucination detection metrics."""

from typing import List, Set
import re

from rag_eval.metrics.base import BaseMetric
from rag_eval.dataset.schema import EvaluationSample, MetricResult


class HallucinationDetector(BaseMetric):
    """Detect hallucinations in generated answers."""
    
    def __init__(self, method: str = "entailment", **kwargs):
        super().__init__(name="HallucinationDetector", **kwargs)
        self.method = method
        self.requires_contexts = True
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        self.validate_sample(sample)
        
        if self.method == "keyword":
            return self._keyword_based(sample)
        elif self.method == "entailment":
            return self._entailment_based(sample)
        else:
            return self._create_result(None, error=f"Unknown method: {self.method}")
    
    def _keyword_based(self, sample: EvaluationSample) -> MetricResult:
        """Simple keyword-based hallucination detection."""
        answer_words = set(sample.generated_answer.lower().split())
        context_words = set()
        for ctx in sample.retrieved_contexts:
            context_words.update(ctx.lower().split())
        
        # Calculate overlap
        overlap = len(answer_words & context_words)
        total = len(answer_words)
        
        if total == 0:
            return self._create_result(1.0)  # Empty answer, no hallucination
        
        support_ratio = overlap / total
        hallucination_score = 1.0 - support_ratio
        
        return self._create_result(
            score=hallucination_score,
            details={
                "method": "keyword",
                "support_ratio": support_ratio,
                "answer_words": total,
                "supported_words": overlap
            }
        )
    
    def _entailment_based(self, sample: EvaluationSample) -> MetricResult:
        """NLI-based hallucination detection."""
        # Would use a model like RoBERTa-large-MNLI
        # Placeholder for blueprint
        return self._create_result(
            score=0.1,
            details={"method": "entailment"}
        )
```

### 8. Evaluators Module (`rag_eval/evaluators/`)

**`base.py`** - Base evaluator interface
**`llm_judge.py`** - LLM-as-judge implementation
**`heuristic.py`** - Rule-based evaluators

Key features:
- Prompt templates for LLM judges
- Multi-aspect evaluation
- Scoring rubrics
- Chain-of-thought reasoning

### 9. Pipeline Module (`rag_eval/pipeline/`)

**`runner.py`** - Main evaluation orchestration
```python
class RAGEvaluator:
    def __init__(self, metrics: List[BaseMetric], config: Optional[Dict] = None)
    def evaluate(self, dataset: EvaluationDataset) -> EvaluationResults
    def evaluate_sample(self, sample: EvaluationSample) -> SampleResult
```

**`async_runner.py`** - Parallel execution
```python
class AsyncRAGEvaluator:
    async def evaluate_async(self, dataset, max_concurrent=10)
    async def evaluate_batch_async(self, samples)
```

**`caching.py`** - LLM response caching
```python
class LLMCache:
    def __init__(self, backend="memory", ttl=3600)
    def get(self, key: str) -> Optional[str]
    def set(self, key: str, value: str)
```

### 10. Aggregation Module (`rag_eval/aggregation/`)

**`stats.py`** - Statistical aggregations
```python
def compute_statistics(scores: List[float]) -> Dict[str, float]
def compute_confidence_intervals(scores: List[float], confidence=0.95)
def detect_outliers(scores: List[float], method="iqr")
```

**`grouping.py`** - Failure analysis and clustering
```python
def group_by_metadata(results: List[SampleResult], key: str)
def cluster_failures(results: List[SampleResult], n_clusters=5)
def identify_failure_patterns(results: List[SampleResult])
```

### 11. Reporting Module (`rag_eval/reporting/`)

**`cli.py`** - Rich CLI output
```python
def print_results(results: EvaluationResults, format="table")
def print_summary(results: EvaluationResults)
def print_detailed_report(results: EvaluationResults)
```

**`json_report.py`** - JSON export
```python
def export_json(results: EvaluationResults, path: str)
def export_csv(results: EvaluationResults, path: str)
```

**`dashboard.py`** - Streamlit integration
```python
def create_dashboard(results: EvaluationResults)
def plot_metric_distribution(metric_name: str, scores: List[float])
def plot_confusion_matrix(results: EvaluationResults)
```

### 12. Integrations Module (`rag_eval/integrations/`)

**`langchain_adapter.py`**
```python
class LangChainAdapter:
    def __init__(self, chain: Any)
    def run(self, query: str) -> Dict[str, Any]
    def extract_contexts(self, result: Any) -> List[str]
```

**`llamaindex_adapter.py`**
```python
class LlamaIndexAdapter:
    def __init__(self, index: Any)
    def query(self, query_str: str) -> Dict[str, Any]
```

**`custom_adapter.py`**
```python
class CustomRAGAdapter:
    def run(self, query: str) -> Dict[str, Any]
```

### 13. Utils Module (`rag_eval/utils/`)

**`embeddings.py`** - Embedding utilities
```python
class EmbeddingCache:
    def get_embedding(self, text: str, model: str) -> np.ndarray
    def batch_encode(self, texts: List[str]) -> np.ndarray
```

**`logging.py`** - Logging configuration
```python
def setup_logging(level: str = "INFO", format: str = "json")
def get_logger(name: str) -> logging.Logger
```

**`config.py`** - Configuration management
```python
def load_config(path: str) -> Dict[str, Any]
def merge_configs(base: Dict, override: Dict) -> Dict
```

### 14. Configuration (`rag_eval/config/default.yaml`)

```yaml
evaluator:
  cache_enabled: true
  cache_dir: ".cache/rag_eval"
  max_retries: 3
  timeout: 30
  batch_size: 32

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.0
  max_tokens: 500
  api_key_env: "OPENAI_API_KEY"

embeddings:
  model: "all-MiniLM-L6-v2"
  batch_size: 32
  device: "cpu"

metrics:
  recall_at_k:
    k: 5
    threshold: 0.7
  
  faithfulness:
    model: "gpt-4"
    threshold: 0.8
  
  semantic_similarity:
    model: "all-MiniLM-L6-v2"
    threshold: 0.75

logging:
  level: "INFO"
  format: "json"
  file: "rag_eval.log"

reporting:
  output_dir: "evaluation_results"
  save_json: true
  save_csv: true
  create_dashboard: false
```

### 15. Tests (`tests/`)

**`test_metrics.py`**
```python
def test_recall_at_k()
def test_precision_at_k()
def test_semantic_similarity()
def test_faithfulness()
```

**`test_pipeline.py`**
```python
def test_evaluator_initialization()
def test_evaluate_dataset()
def test_async_evaluation()
```

**`test_dataset.py`**
```python
def test_load_json_dataset()
def test_load_csv_dataset()
def test_dataset_validation()
```

### 16. Examples (`examples/`)

**`basic_usage.py`**
```python
from rag_eval import RAGEvaluator, load_dataset
from rag_eval.metrics import RecallAtK, Faithfulness

dataset = load_dataset("datasets/sample_dataset.json")
evaluator = RAGEvaluator(metrics=[
    RecallAtK(k=5),
    Faithfulness(model="gpt-4")
])
results = evaluator.evaluate(dataset)
print(results.summary_text())
```

**`custom_metric.py`**
```python
from rag_eval.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def compute(self, sample):
        # Custom logic
        return self._create_result(score=0.85)
```

**`ci_pipeline_example.py`**
```python
# Example CI/CD integration
import sys
from rag_eval import RAGEvaluator, load_dataset

results = evaluator.evaluate(dataset)
if results.aggregated_metrics["Faithfulness"]["mean"] < 0.8:
    sys.exit(1)  # Fail CI if quality drops
```

### 17. Sample Datasets (`datasets/`)

**`sample_dataset.json`**
```json
{
  "name": "Sample RAG Evaluation Dataset",
  "version": "1.0",
  "samples": [
    {
      "query": "What is the capital of France?",
      "retrieved_contexts": [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe."
      ],
      "generated_answer": "The capital of France is Paris.",
      "ground_truth": "Paris",
      "ground_truth_contexts": [
        "Paris is the capital and most populous city of France."
      ],
      "metadata": {
        "category": "geography",
        "difficulty": "easy"
      }
    }
  ]
}
```

### 18. Scripts (`scripts/`)

**`run_eval.py`**
```python
#!/usr/bin/env python
"""Run evaluation from command line."""
import click
from rag_eval import RAGEvaluator, load_dataset

@click.command()
@click.argument('dataset_path')
@click.option('--config', default='config.yaml')
@click.option('--output', default='results.json')
def main(dataset_path, config, output):
    dataset = load_dataset(dataset_path)
    # Load config and run evaluation
    pass

if __name__ == '__main__':
    main()
```

**`generate_synthetic_data.py`**
```python
"""Generate synthetic evaluation datasets."""
def generate_qa_pairs(n_samples=100):
    # Use LLM to generate synthetic Q&A pairs
    pass
```

### 19. CI/CD (`.github/workflows/ci.yml`)

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest --cov=rag_eval --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
    
    - name: Lint
      run: |
        black --check .
        ruff check .
        mypy rag_eval
```

## 🎯 Implementation Priority

1. **Phase 1: Core Foundation** ✅
   - Project setup and configuration
   - Dataset module (schema, loader, validator)
   - Metrics base classes

2. **Phase 2: Essential Metrics**
   - Retrieval metrics (Recall, Precision, nDCG, MRR)
   - Semantic similarity metrics
   - Basic LLM metrics

3. **Phase 3: Evaluation Pipeline**
   - Synchronous evaluator
   - Result aggregation
   - Basic reporting

4. **Phase 4: Advanced Features**
   - Async evaluation
   - LLM caching
   - Advanced metrics (hallucination detection)

5. **Phase 5: Integrations & Polish**
   - Framework adapters
   - Dashboard
   - Comprehensive tests
   - Documentation

## 🔧 Key Design Principles

1. **Type Safety**: Use Pydantic for all data models
2. **Extensibility**: Easy to add custom metrics
3. **Performance**: Async support, caching, batch processing
4. **Reliability**: Comprehensive error handling and validation
5. **Observability**: Detailed logging and reporting
6. **Production-Ready**: CI/CD, testing, documentation

## 📚 Additional Documentation Needed

- API Reference (auto-generated from docstrings)
- User Guide with tutorials
- Custom Metrics Guide
- Integration Examples
- Performance Optimization Guide
- Troubleshooting Guide

## 🚀 Getting Started (After Implementation)

```bash
# Install
pip install rag-eval-pro

# Quick start
from rag_eval import RAGEvaluator, load_dataset
from rag_eval.metrics import RecallAtK, Faithfulness

dataset = load_dataset("data.json")
evaluator = RAGEvaluator(metrics=[
    RecallAtK(k=5),
    Faithfulness()
])
results = evaluator.evaluate(dataset)
results.save_json("results.json")
```

---

This blueprint provides a complete roadmap for implementing a production-grade RAG evaluation library. Each component is designed to be modular, testable, and extensible.