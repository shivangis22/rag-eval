# Contributing to RAG Eval Pro

Thank you for your interest in contributing to RAG Eval Pro! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- **Bug Reports**: Report bugs via GitHub Issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Improve or expand documentation
- **Examples**: Add usage examples or tutorials
- **Testing**: Write tests or improve test coverage

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/shivangis22/rag-eval.git
cd rag-eval-pro

# Add upstream remote
git remote add upstream https://github.com/shivangis22/rag-eval.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 💻 Development Workflow

### Code Style

We use automated tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking

```bash
# Format code
black .

# Lint code
ruff check --fix .

# Type check
mypy rag_eval
```

Pre-commit hooks will automatically run these checks before each commit.

### Writing Code

1. **Follow PEP 8**: Python style guide
2. **Type Hints**: Add type hints to all functions
3. **Docstrings**: Use Google-style docstrings
4. **Error Handling**: Handle errors gracefully with informative messages
5. **Logging**: Use the logging module for debug/info messages

Example:

```python
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def process_samples(
    samples: List[dict],
    batch_size: int = 32,
    verbose: bool = False
) -> List[dict]:
    """Process evaluation samples in batches.
    
    Args:
        samples: List of evaluation samples to process
        batch_size: Number of samples to process at once
        verbose: Whether to log detailed progress
        
    Returns:
        List of processed samples with results
        
    Raises:
        ValueError: If samples list is empty
        
    Example:
        >>> samples = [{"query": "test", "context": "..."}]
        >>> results = process_samples(samples, batch_size=16)
    """
    if not samples:
        raise ValueError("Samples list cannot be empty")
    
    logger.info(f"Processing {len(samples)} samples in batches of {batch_size}")
    # Implementation...
    return processed_samples
```

### Testing

Write tests for all new features and bug fixes:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rag_eval --cov-report=html

# Run specific test file
pytest tests/test_metrics.py

# Run specific test
pytest tests/test_metrics.py::test_recall_at_k

# Run only fast tests
pytest -m "not slow"
```

Test structure:

```python
import pytest
from rag_eval.metrics import RecallAtK

class TestRecallAtK:
    """Test suite for RecallAtK metric."""
    
    def test_perfect_recall(self):
        """Test recall calculation with perfect retrieval."""
        metric = RecallAtK(k=5)
        sample = {
            "retrieved_contexts": ["doc1", "doc2", "doc3"],
            "ground_truth_contexts": ["doc1", "doc2", "doc3"]
        }
        result = metric.compute(sample)
        assert result["score"] == 1.0
    
    def test_zero_recall(self):
        """Test recall calculation with no relevant documents."""
        metric = RecallAtK(k=5)
        sample = {
            "retrieved_contexts": ["doc1", "doc2"],
            "ground_truth_contexts": ["doc3", "doc4"]
        }
        result = metric.compute(sample)
        assert result["score"] == 0.0
    
    @pytest.mark.parametrize("k,expected", [
        (1, 0.33),
        (3, 1.0),
        (5, 1.0),
    ])
    def test_recall_at_different_k(self, k, expected):
        """Test recall at different k values."""
        metric = RecallAtK(k=k)
        sample = {
            "retrieved_contexts": ["doc1", "doc2", "doc3"],
            "ground_truth_contexts": ["doc1", "doc2", "doc3"]
        }
        result = metric.compute(sample)
        assert abs(result["score"] - expected) < 0.01
```

### Adding New Metrics

1. Create a new file in `rag_eval/metrics/`
2. Inherit from `BaseMetric`
3. Implement required methods
4. Add tests
5. Update documentation

Example:

```python
from rag_eval.metrics.base import BaseMetric
from typing import Dict, Any, List

class CustomMetric(BaseMetric):
    """Custom evaluation metric.
    
    Args:
        threshold: Score threshold for binary classification
        **kwargs: Additional arguments passed to BaseMetric
    """
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
    
    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metric for a single sample.
        
        Args:
            sample: Evaluation sample containing required fields
            
        Returns:
            Dictionary with score and optional details
        """
        # Your implementation
        score = self._calculate_score(sample)
        
        return {
            "score": score,
            "passed": score >= self.threshold,
            "details": {
                "threshold": self.threshold,
                # Additional details
            }
        }
    
    def aggregate(self, scores: List[float]) -> Dict[str, float]:
        """Aggregate scores across multiple samples.
        
        Args:
            scores: List of individual scores
            
        Returns:
            Dictionary with aggregated statistics
        """
        import numpy as np
        
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
        }
```

## 📝 Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:

```bash
git commit -m "feat(metrics): add BLEU score metric"
git commit -m "fix(pipeline): handle empty dataset gracefully"
git commit -m "docs(readme): update installation instructions"
```

## 🔄 Pull Request Process

1. **Update Documentation**: Ensure README and docstrings are updated
2. **Add Tests**: Include tests for new features
3. **Run Tests**: Ensure all tests pass
4. **Update Changelog**: Add entry to CHANGELOG.md
5. **Create PR**: Submit pull request with clear description

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests added
```

## 🐛 Reporting Bugs

Use GitHub Issues with the bug report template:

**Title**: Clear, descriptive title

**Description**:
- Expected behavior
- Actual behavior
- Steps to reproduce
- Environment details (OS, Python version, package versions)
- Error messages/stack traces

## 💡 Feature Requests

Use GitHub Issues with the feature request template:

**Title**: Clear feature description

**Description**:
- Problem statement
- Proposed solution
- Alternative solutions considered
- Additional context

## 📚 Documentation

- Keep README.md up to date
- Add docstrings to all public APIs
- Update examples when adding features
- Create tutorials for complex features

## 🤝 Code Review

All submissions require review. We aim to:

- Provide constructive feedback
- Respond within 48 hours
- Merge approved PRs promptly

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

## 📧 Questions?

- Open a GitHub Discussion
- Email: your.email@example.com
- Join our community chat

Thank you for contributing to RAG Eval Pro! 🎉