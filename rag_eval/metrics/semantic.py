"""Semantic similarity metrics using embeddings."""

import re
from typing import Optional, List
import logging
import numpy as np

from rag_eval.metrics.base import BaseMetric
from rag_eval.dataset.schema import RAGSample, MetricResult

logger = logging.getLogger(__name__)


class SemanticSimilarity(BaseMetric):
    """Semantic similarity using sentence embeddings.
    
    Compares generated answer with ground truth or contexts using
    embedding-based cosine similarity.
    
    Args:
        model: Sentence transformer model name
        compare_to: What to compare against ("ground_truth" or "contexts")
        threshold: Optional threshold for pass/fail
    """
    
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        compare_to: str = "ground_truth",
        backend: str = "auto",
        allow_fallback: bool = True,
        threshold: Optional[float] = None,
        **kwargs
    ):
        super().__init__(name="SemanticSimilarity", threshold=threshold, **kwargs)
        if backend not in {"auto", "sentence-transformers", "lexical"}:
            raise ValueError(
                "backend must be one of: 'auto', 'sentence-transformers', 'lexical'"
            )
        self.model_name = model
        self.compare_to = compare_to
        self.backend = backend
        self.allow_fallback = allow_fallback
        self.model = None
        self.requires_ground_truth = (compare_to == "ground_truth")
        self.requires_contexts = (compare_to == "contexts")
        self.requires_answer = True
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            if self.backend == "lexical":
                self.model = False
                return
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                if self.backend == "sentence-transformers":
                    raise ImportError(
                        "sentence-transformers required for semantic similarity. "
                        "Install with: pip install rag-eval-pro[semantic]"
                    )
                if self.allow_fallback:
                    logger.warning(
                        "Falling back to lexical semantic similarity because "
                        "sentence-transformers is not installed."
                    )
                    self.model = False
                    return
                raise ImportError(
                    "sentence-transformers required for semantic similarity. "
                    "Install with: pip install rag-eval-pro[semantic]"
                )
            except Exception as e:
                if self.backend == "sentence-transformers" and not self.allow_fallback:
                    raise RuntimeError(
                        "Embedding model could not be loaded locally. "
                        "Download the model first or use backend='lexical'."
                    ) from e
                logger.warning(
                    "Falling back to lexical semantic similarity because the "
                    f"embedding model could not be loaded: {e}"
                )
                self.model = False

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text for lexical fallback similarity."""
        return set(re.findall(r"\w+", text.lower()))

    def _fallback_similarity(self, answer: str, reference: str) -> float:
        """Compute a simple lexical similarity when embeddings are unavailable."""
        answer_tokens = self._tokenize(answer)
        reference_tokens = self._tokenize(reference)
        if not answer_tokens or not reference_tokens:
            return 0.0

        intersection = len(answer_tokens & reference_tokens)
        union = len(answer_tokens | reference_tokens)
        return intersection / union if union else 0.0
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute semantic similarity for a sample.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with similarity score
        """
        try:
            self.validate_sample(sample)
            self._load_model()

            answer = sample.generated_answer

            if self.compare_to == "ground_truth":
                if not sample.ground_truth:
                    return self._create_result(None, error="No ground truth provided")
                reference = sample.ground_truth
            elif self.compare_to == "contexts":
                if not sample.retrieved_docs:
                    return self._create_result(None, error="No contexts provided")
                reference = " ".join(sample.retrieved_docs)
            else:
                return self._create_result(
                    None,
                    error=f"Invalid compare_to: {self.compare_to}"
                )

            if self.model is False:
                similarity = self._fallback_similarity(answer, reference)
                details = {
                    "compare_to": self.compare_to,
                    "model": self.model_name,
                    "fallback": "lexical_jaccard"
                }
            else:
                embeddings = self.model.encode([answer, reference])
                similarity = self._cosine_similarity(embeddings[0], embeddings[1])
                details = {
                    "compare_to": self.compare_to,
                    "model": self.model_name
                }

            return self._create_result(
                score=float(similarity),
                details=details
            )
        except Exception as e:
            return self._create_result(None, error=str(e))
    
    def compute_batch(self, samples: List[RAGSample]) -> List[MetricResult]:
        """Compute similarity for multiple samples efficiently.
        
        Args:
            samples: List of RAG samples
            
        Returns:
            List of metric results
        """
        results = []
        answers = []
        references = []
        valid_indices = []

        try:
            self._load_model()

            # Collect valid samples
            for idx, sample in enumerate(samples):
                try:
                    self.validate_sample(sample)

                    if self.compare_to == "ground_truth":
                        if sample.ground_truth:
                            answers.append(sample.generated_answer)
                            references.append(sample.ground_truth)
                            valid_indices.append(idx)
                    elif self.compare_to == "contexts":
                        if sample.retrieved_docs:
                            answers.append(sample.generated_answer)
                            references.append(" ".join(sample.retrieved_docs))
                            valid_indices.append(idx)
                except Exception:
                    pass

            if not answers or not references:
                return [
                    self._create_result(None, error="No valid samples")
                    for _ in samples
                ]

            if self.model is False:
                result_map = {}
                for idx, answer, reference in zip(valid_indices, answers, references):
                    result_map[idx] = self._create_result(
                        score=float(self._fallback_similarity(answer, reference)),
                        details={
                            "compare_to": self.compare_to,
                            "model": self.model_name,
                            "fallback": "lexical_jaccard"
                        }
                    )
            else:
                answer_embeddings = self.model.encode(answers, batch_size=32)
                reference_embeddings = self.model.encode(references, batch_size=32)
                similarities = [
                    self._cosine_similarity(ans_emb, ref_emb)
                    for ans_emb, ref_emb in zip(answer_embeddings, reference_embeddings)
                ]
                result_map = {}
                for idx, similarity in zip(valid_indices, similarities):
                    result_map[idx] = self._create_result(
                        score=float(similarity),
                        details={"compare_to": self.compare_to, "model": self.model_name}
                    )

            for idx, _sample in enumerate(samples):
                if idx in result_map:
                    results.append(result_map[idx])
                else:
                    results.append(self._create_result(None, error="Invalid sample"))

            return results
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            return [self._create_result(None, error=str(e)) for _ in samples]
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class BERTScore(BaseMetric):
    """BERTScore for semantic similarity.
    
    Uses contextual embeddings from BERT for more nuanced similarity.
    
    Args:
        model: BERT model name
        threshold: Optional threshold for pass/fail
    """
    
    def __init__(
        self,
        model: str = "bert-base-uncased",
        threshold: Optional[float] = None,
        **kwargs
    ):
        super().__init__(name="BERTScore", threshold=threshold, **kwargs)
        self.model_name = model
        self.requires_ground_truth = True
        self.requires_answer = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute BERTScore for a sample.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with BERTScore
        """
        self.validate_sample(sample)
        
        try:
            from bert_score import score
        except ImportError:
            return self._create_result(
                None,
                error="bert-score required. Install with: pip install bert-score"
            )
        
        if not sample.ground_truth:
            return self._create_result(None, error="No ground truth provided")
        
        try:
            # Calculate BERTScore
            P, R, F1 = score(
                [sample.generated_answer],
                [sample.ground_truth],
                model_type=self.model_name,
                verbose=False
            )
            
            return self._create_result(
                score=float(F1[0]),
                details={
                    "precision": float(P[0]),
                    "recall": float(R[0]),
                    "f1": float(F1[0]),
                    "model": self.model_name
                }
            )
        except Exception as e:
            return self._create_result(None, error=str(e))


class AnswerSimilarity(BaseMetric):
    """Answer similarity to ground truth using multiple methods.
    
    Combines multiple similarity approaches for robust evaluation.
    
    Args:
        methods: List of methods to use ("embedding", "bert", "rouge")
        threshold: Optional threshold for pass/fail
    """
    
    def __init__(
        self,
        methods: List[str] = ["embedding"],
        threshold: Optional[float] = None,
        **kwargs
    ):
        super().__init__(name="AnswerSimilarity", threshold=threshold, **kwargs)
        self.methods = methods
        self.requires_ground_truth = True
        self.requires_answer = True
        
        # Initialize sub-metrics
        self.sub_metrics = {}
        if "embedding" in methods:
            self.sub_metrics["embedding"] = SemanticSimilarity()
        if "bert" in methods:
            self.sub_metrics["bert"] = BERTScore()
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute combined similarity score.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with combined score
        """
        self.validate_sample(sample)
        
        scores = {}
        for method_name, metric in self.sub_metrics.items():
            try:
                result = metric.compute(sample)
                if result.score is not None:
                    scores[method_name] = result.score
            except Exception as e:
                logger.warning(f"Method {method_name} failed: {e}")
        
        if not scores:
            return self._create_result(None, error="All methods failed")
        
        # Average scores
        combined_score = sum(scores.values()) / len(scores)
        
        return self._create_result(
            score=combined_score,
            details={
                "methods": list(scores.keys()),
                "individual_scores": scores
            }
        )
