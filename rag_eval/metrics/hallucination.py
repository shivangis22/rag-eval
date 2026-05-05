"""Hallucination detection metrics."""

from typing import List, Set, Optional
import logging
import re

from rag_eval.metrics.base import BaseMetric
from rag_eval.dataset.schema import RAGSample, MetricResult

logger = logging.getLogger(__name__)


class HallucinationDetector(BaseMetric):
    """Detect hallucinations in generated answers.
    
    Uses multiple strategies to detect when answers contain information
    not supported by the retrieved contexts.
    
    Args:
        method: Detection method ("keyword", "entailment", "llm")
        threshold: Optional threshold (lower is better - less hallucination)
    """
    
    def __init__(
        self,
        method: str = "keyword",
        threshold: Optional[float] = None,
        **kwargs
    ):
        super().__init__(name="HallucinationDetector", threshold=threshold, **kwargs)
        self.method = method
        self.requires_contexts = True
        self.requires_answer = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute hallucination score for a sample.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with hallucination score (0 = no hallucination, 1 = high hallucination)
        """
        self.validate_sample(sample)
        
        if self.method == "keyword":
            return self._keyword_based(sample)
        elif self.method == "entailment":
            return self._entailment_based(sample)
        elif self.method == "llm":
            return self._llm_based(sample)
        else:
            return self._create_result(None, error=f"Unknown method: {self.method}")
    
    def _keyword_based(self, sample: RAGSample) -> MetricResult:
        """Simple keyword-based hallucination detection.
        
        Checks what fraction of answer words are not in contexts.
        """
        answer_words = self._extract_words(sample.generated_answer)
        context_words = set()
        for ctx in sample.retrieved_docs:
            context_words.update(self._extract_words(ctx))
        
        if not answer_words:
            return self._create_result(0.0, details={"method": "keyword"})
        
        # Calculate unsupported words
        unsupported = answer_words - context_words
        hallucination_score = len(unsupported) / len(answer_words)
        
        return self._create_result(
            score=hallucination_score,
            details={
                "method": "keyword",
                "total_words": len(answer_words),
                "unsupported_words": len(unsupported),
                "support_ratio": 1.0 - hallucination_score
            }
        )
    
    def _entailment_based(self, sample: RAGSample) -> MetricResult:
        """NLI-based hallucination detection.
        
        Uses natural language inference to check if answer is entailed by contexts.
        """
        try:
            from transformers import pipeline
        except ImportError:
            return self._create_result(
                None,
                error="transformers required. Install with: pip install transformers"
            )
        
        try:
            # Load NLI model (cached after first use)
            if not hasattr(self, '_nli_model'):
                self._nli_model = pipeline(
                    "text-classification",
                    model="microsoft/deberta-v3-base-mnli-fever-anli",
                    device=-1  # CPU
                )
            
            # Combine contexts
            context = " ".join(sample.retrieved_docs)
            answer = sample.generated_answer
            
            # Check entailment
            result = self._nli_model(f"{context} [SEP] {answer}")[0]
            
            # Convert to hallucination score
            # entailment = 0 (no hallucination)
            # neutral = 0.5 (some hallucination)
            # contradiction = 1.0 (high hallucination)
            label_to_score = {
                "ENTAILMENT": 0.0,
                "NEUTRAL": 0.5,
                "CONTRADICTION": 1.0
            }
            
            hallucination_score = label_to_score.get(result["label"].upper(), 0.5)
            
            return self._create_result(
                score=hallucination_score,
                details={
                    "method": "entailment",
                    "nli_label": result["label"],
                    "nli_score": result["score"]
                }
            )
        except Exception as e:
            logger.error(f"Entailment-based detection failed: {e}")
            return self._create_result(None, error=str(e))
    
    def _llm_based(self, sample: RAGSample) -> MetricResult:
        """LLM-based hallucination detection.
        
        Uses LLM to identify unsupported claims in the answer.
        """
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._create_result(None, error="No OPENAI_API_KEY provided")
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
        except ImportError:
            return self._create_result(
                None,
                error="openai required. Install with: pip install openai"
            )
        
        contexts = "\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(sample.retrieved_docs))
        answer = sample.generated_answer
        
        prompt = f"""Analyze if the answer contains hallucinations (unsupported claims).

Contexts:
{contexts}

Answer:
{answer}

Identify any claims in the answer that are:
1. Not supported by the contexts
2. Contradict the contexts
3. Add information not present in contexts

Provide a hallucination score from 0.0 to 1.0:
- 0.0 = No hallucination, fully supported
- 0.3 = Minor unsupported details
- 0.5 = Some hallucinated content
- 0.7 = Significant hallucination
- 1.0 = Completely hallucinated

Score (0.0-1.0):"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse score
            import re
            numbers = re.findall(r'0?\.\d+|[01]\.?\d*', response_text)
            if numbers:
                score = float(numbers[0])
                score = max(0.0, min(1.0, score))
            else:
                return self._create_result(
                    None,
                    error=f"Could not parse score from: {response_text}"
                )
            
            return self._create_result(
                score=score,
                details={
                    "method": "llm",
                    "model": "gpt-4",
                    "raw_response": response_text
                }
            )
        except Exception as e:
            return self._create_result(None, error=str(e))
    
    @staticmethod
    def _extract_words(text: str) -> Set[str]:
        """Extract normalized words from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of normalized words
        """
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        return {w for w in words if w not in stop_words and len(w) > 2}


class FactualConsistency(BaseMetric):
    """Check factual consistency between answer and contexts.
    
    Identifies specific factual claims and verifies them against contexts.
    """
    
    def __init__(self, threshold: Optional[float] = None, **kwargs):
        super().__init__(name="FactualConsistency", threshold=threshold, **kwargs)
        self.requires_contexts = True
        self.requires_answer = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute factual consistency score.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with consistency score
        """
        self.validate_sample(sample)
        
        # Extract claims from answer (simple sentence splitting)
        claims = self._extract_claims(sample.generated_answer)
        
        if not claims:
            return self._create_result(1.0, details={"num_claims": 0})
        
        # Check each claim against contexts
        context_text = " ".join(sample.retrieved_docs).lower()
        supported_claims = 0
        
        for claim in claims:
            # Simple keyword matching (can be improved with NLI)
            claim_words = set(claim.lower().split())
            if any(word in context_text for word in claim_words if len(word) > 3):
                supported_claims += 1
        
        consistency_score = supported_claims / len(claims)
        
        return self._create_result(
            score=consistency_score,
            details={
                "num_claims": len(claims),
                "supported_claims": supported_claims,
                "unsupported_claims": len(claims) - supported_claims
            }
        )
    
    @staticmethod
    def _extract_claims(text: str) -> List[str]:
        """Extract individual claims from text.
        
        Args:
            text: Input text
            
        Returns:
            List of claims (sentences)
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


class GroundingScore(BaseMetric):
    """Measure how well the answer is grounded in retrieved contexts.
    
    Combines multiple signals to assess grounding quality.
    """
    
    def __init__(self, threshold: Optional[float] = None, **kwargs):
        super().__init__(name="GroundingScore", threshold=threshold, **kwargs)
        self.requires_contexts = True
        self.requires_answer = True
        
        # Initialize sub-metrics
        self.hallucination_detector = HallucinationDetector(method="keyword")
        self.factual_consistency = FactualConsistency()
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute overall grounding score.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with grounding score
        """
        self.validate_sample(sample)
        
        # Get hallucination score (lower is better)
        hall_result = self.hallucination_detector.compute(sample)
        
        # Get factual consistency (higher is better)
        fact_result = self.factual_consistency.compute(sample)
        
        if hall_result.score is None or fact_result.score is None:
            return self._create_result(None, error="Sub-metric computation failed")
        
        # Combine scores: grounding = (1 - hallucination) * consistency
        grounding_score = (1.0 - hall_result.score) * fact_result.score
        
        return self._create_result(
            score=grounding_score,
            details={
                "hallucination_score": hall_result.score,
                "factual_consistency": fact_result.score,
                "grounding_score": grounding_score
            }
        )

