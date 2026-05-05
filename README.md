# RAG Eval Pro 🎯

[![CI](https://github.com/yourusername/rag-eval-pro/workflows/CI/badge.svg)](https://github.com/yourusername/rag-eval-pro/actions)
[![PyPI version](https://badge.fury.io/py/rag-eval-pro.svg)](https://badge.fury.io/py/rag-eval-pro)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-Grade Evaluation Framework for RAG Systems**

A modular, extensible, and production-ready evaluation framework for Retrieval-Augmented Generation (RAG) systems.

## 🎯 What It Evaluates

- 📄 **Retrieval Quality** - Recall@k, Precision@k, nDCG, MRR
- 🧾 **Generation Quality** - Semantic similarity, relevance, completeness
- 🔗 **Grounding / Faithfulness** - Answer grounding in retrieved context
- ⚙️ **System Performance** - Latency, token usage, cost estimation

## 🚀 Why This Library Exists

Most RAG systems fail silently because:
- ❌ Retrieval is not evaluated
- ❌ Hallucinations go undetected
- ❌ Metrics are inconsistent across teams

**This library solves that by providing:**
- ✅ Unified evaluation across retrieval + generation + grounding
- ✅ Pluggable metric system
- ✅ CI/CD integration for regression detection
- ✅ LLM-as-a-judge paradigm with consistent scoring

## 🏗️ Architecture Overview

```
Dataset → RAG Execution → Metrics → Aggregation → Reporting → CI/CD
```

## 🔧 Works With

- Custom RAG pipelines
- LangChain / LlamaIndex
- Any LLM provider (OpenAI, Anthropic, Cohere, etc.)

## 📦 Installation

```bash
# Basic offline installation
pip install rag-eval-pro

# Local semantic metrics
pip install rag-eval-pro[semantic]

# LLM-backed metrics
pip install rag-eval-pro[llm]

# Everything
pip install rag-eval-pro[all]
```

### Installation Modes

- `rag-eval-pro`: fully offline core library for datasets, retrieval metrics, aggregation, caching, and reporting
- `rag-eval-pro[semantic]`: enables embedding-backed semantic similarity
- `rag-eval-pro[llm]`: enables provider-backed LLM judge metrics
- `rag-eval-pro[all]`: installs every optional feature

## 🎯 Quick Start

```python
from rag_eval.pipeline.runner import RAGEvaluator
from rag_eval.metrics.retrieval import RecallAtK
from rag_eval.metrics.semantic import SemanticSimilarity
from rag_eval.dataset.schema import RAGDataset, RAGSample

# Create dataset
dataset = RAGDataset(
    name="demo",
    version="v1",
    samples=[
        RAGSample(
            query="What is refund policy?",
            retrieved_docs=["refund doc"],
            generated_answer="Refund allowed in 30 days",
            ground_truth="Refund allowed in 30 days",
            relevant_docs=["refund doc"]
        )
    ]
)

# Initialize evaluator with metrics
metrics = [
    RecallAtK(k=5),
    SemanticSimilarity(
        backend="lexical",
        compare_to="ground_truth"
    )
]

evaluator = RAGEvaluator(metrics)
results = evaluator.evaluate(dataset)

print(results)
```

### LLM Metrics

```python
from rag_eval.metrics.llm_metrics import FaithfulnessMetric

metric = FaithfulnessMetric(
    provider="openai",
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### Semantic Backends

```python
from rag_eval.metrics.semantic import SemanticSimilarity

# Offline fallback
SemanticSimilarity(backend="lexical")

# Prefer local/downloaded sentence-transformers model, but fall back if unavailable
SemanticSimilarity(
    backend="auto",
    model="all-MiniLM-L6-v2"
)

# Require embedding model explicitly
SemanticSimilarity(
    backend="sentence-transformers",
    model="all-MiniLM-L6-v2",
    allow_fallback=False
)
```

### Output Format

```json
{
  "mean": {
    "faithfulness": 0.89,
    "recall@5": 0.76
  },
  "p95": {
    "faithfulness": 0.95
  }
}
```

## 📊 Dataset Format

The library is **dataset-first**. All evaluations start with a structured dataset.

### Required Schema

```json
{
  "query": "What is refund policy?",
  "retrieved_docs": ["doc1", "doc2"],
  "generated_answer": "You can request refund within 30 days.",
  "ground_truth": "Refund allowed within 30 days.",
  "relevant_docs": ["doc1"]
}
```

### Python Schema

```python
from rag_eval.dataset.schema import RAGSample

sample = RAGSample(
    query="What is refund policy?",
    retrieved_docs=["doc1", "doc2"],
    generated_answer="You can request refund within 30 days.",
    ground_truth="Refund allowed within 30 days.",
    relevant_docs=["doc1"]
)
```

## 🔧 Advanced Usage

### Custom Metrics

```python
from rag_eval.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def compute(self, sample):
        # Your custom logic
        score = your_evaluation_logic(sample)
        return {"score": score, "details": {...}}
    
    def aggregate(self, scores):
        return {"mean": np.mean(scores), "std": np.std(scores)}
```

### Async Evaluation

```python
from rag_eval.pipeline import AsyncRAGEvaluator

evaluator = AsyncRAGEvaluator(
    metrics=[...],
    max_concurrent=10,
    cache_enabled=True
)

results = await evaluator.evaluate_async(dataset)
```

### LangChain Integration

```python
from rag_eval.integrations import LangChainAdapter
from langchain.chains import RetrievalQA

# Your LangChain RAG system
qa_chain = RetrievalQA.from_chain_type(...)

# Wrap with adapter
adapter = LangChainAdapter(qa_chain)

# Evaluate
results = evaluator.evaluate(dataset, rag_system=adapter)
```

## 📈 Metrics Overview

### Retrieval Metrics
- **Recall@K**: Measures retrieval coverage
- **Precision@K**: Measures retrieval accuracy
- **nDCG**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

### Generation Metrics
- **Faithfulness**: Answer grounding in retrieved context
- **Relevance**: Answer relevance to query
- **Coherence**: Answer logical consistency
- **Semantic Similarity**: Embedding-based similarity

### Runtime Requirements
- Retrieval metrics work offline after `pip install rag-eval-pro`
- `SemanticSimilarity(backend="sentence-transformers")` needs `pip install rag-eval-pro[semantic]`
- LLM judge metrics need `pip install rag-eval-pro[llm]` plus provider credentials
- For local-only demos and CI, use `provider="mock"` for LLM metrics

### Hallucination Detection
- **Context Contradiction**: Detects contradictions with context
- **Factual Consistency**: Checks factual alignment
- **Unsupported Claims**: Identifies unsupported statements

## 🛠️ Configuration

Create a `config.yaml`:

```yaml
evaluator:
  cache_enabled: true
  cache_dir: ".cache/rag_eval"
  max_retries: 3
  timeout: 30

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.0
  max_tokens: 500

embeddings:
  model: "all-MiniLM-L6-v2"
  batch_size: 32

logging:
  level: "INFO"
  format: "json"
```

Load configuration:

```python
from rag_eval.utils import load_config

config = load_config("config.yaml")
evaluator = RAGEvaluator.from_config(config)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rag_eval --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

## 🚢 Publishing Notes

- Keep the base package lightweight and offline-safe
- Treat cloud-backed metrics and embedding models as optional extras
- Use `provider="mock"` in tests, examples, and CI when you do not want live API calls
- Document required environment variables for provider-backed metrics


## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup development environment
git clone https://github.com/shivangis22/rag-eval.git
cd rag-eval-pro
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest

# Format code
black .
ruff check --fix .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Inspired by RAGAS, TruLens, and other RAG evaluation frameworks
- Built with modern Python best practices
- Community-driven development

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/shivangis22/rag-eval.git)
- Email: shivangis2208@gmail.com

## 🗺️ Roadmap

- [ ] Support for more LLM providers (Cohere, Gemini)
- [ ] Multi-modal RAG evaluation
- [ ] Real-time evaluation dashboard
- [ ] A/B testing framework
- [ ] Automated dataset generation
- [ ] Integration with MLOps platforms

---

**Star ⭐ this repo if you find it useful!**
