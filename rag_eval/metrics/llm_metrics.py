"""LLM-based evaluation metrics using LLM-as-a-judge paradigm."""

from typing import Optional, Dict, Any
import logging
import os

from rag_eval.metrics.base import BaseMetric
from rag_eval.dataset.schema import RAGSample, MetricResult

logger = logging.getLogger(__name__)


class LLMMetric(BaseMetric):
    """Base class for LLM-based metrics.
    
    Uses LLM-as-a-judge paradigm for evaluation.
    
    Args:
        model: LLM model name (e.g., "gpt-4", "gpt-3.5-turbo")
        api_key: API key for LLM provider
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens for LLM response
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        mock_response: str = "0.5",
        client: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if provider not in {"openai", "anthropic", "mock"}:
            raise ValueError("provider must be one of: 'openai', 'anthropic', 'mock'")
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client
        self.mock_response = mock_response

        env_var = self._default_api_env_var()
        self.api_key = api_key or os.getenv(env_var)

        if self.provider != "mock" and not self.api_key and self.client is None:
            logger.warning(
                f"{self.name}: No API key provided. Set {env_var} or pass api_key/client explicitly."
            )

    def _default_api_env_var(self) -> str:
        """Return the default environment variable for the configured provider."""
        if self.provider == "anthropic":
            return "ANTHROPIC_API_KEY"
        return "OPENAI_API_KEY"

    def _missing_provider_error(self) -> str:
        """Return a user-facing error for missing provider configuration."""
        if self.provider == "mock":
            return "Mock provider misconfigured"
        return (
            f"{self.provider} provider requires {self._default_api_env_var()} "
            "or an explicit api_key/client"
        )

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with the given prompt.
        
        Args:
            prompt: Prompt to send to LLM
            
        Returns:
            LLM response text
            
        Raises:
            ImportError: If openai package not installed
            Exception: If API call fails
        """
        if self.provider == "mock":
            return self.mock_response

        if self.provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI-backed metrics. "
                    "Install with: pip install rag-eval-pro[llm]"
                )

            try:
                client = self.client or openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"LLM API call failed: {e}")
                raise

        if self.provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required for Anthropic-backed metrics. "
                    "Install with: pip install rag-eval-pro[llm]"
                )

            try:
                client = self.client or anthropic.Anthropic(api_key=self.api_key)
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                first_block = response.content[0]
                return getattr(first_block, "text", str(first_block)).strip()
            except Exception as e:
                logger.error(f"LLM API call failed: {e}")
                raise

        raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _parse_score(self, response: str) -> Optional[float]:
        """Parse score from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed score (0-1) or None if parsing fails
        """
        import re
        
        # Try to extract a number from the response
        numbers = re.findall(r'0?\.\d+|[01]\.?\d*', response)
        if numbers:
            try:
                score = float(numbers[0])
                # Clamp to [0, 1]
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        return None


class FaithfulnessMetric(LLMMetric):
    """Evaluate if answer is faithful to retrieved contexts.
    
    Measures whether the generated answer is grounded in the retrieved documents.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="Faithfulness", **kwargs)
        self.requires_contexts = True
        self.requires_answer = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute faithfulness score.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with faithfulness score
        """
        self.validate_sample(sample)
        
        if self.provider != "mock" and not self.api_key and self.client is None:
            return self._create_result(None, error=self._missing_provider_error())
        
        contexts = "\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(sample.retrieved_docs))
        answer = sample.generated_answer
        
        prompt = f"""Evaluate if the following answer is faithful to the given contexts.

A faithful answer:
- Only makes claims supported by the contexts
- Does not contradict the contexts
- Does not add information not present in contexts

Contexts:
{contexts}

Answer:
{answer}

Provide a faithfulness score from 0.0 to 1.0:
- 1.0 = Completely faithful, all claims supported
- 0.7 = Mostly faithful, minor unsupported details
- 0.5 = Partially faithful, some unsupported claims
- 0.3 = Mostly unfaithful, many unsupported claims
- 0.0 = Completely unfaithful, contradicts or fabricates

Score (0.0-1.0):"""
        
        try:
            response = self._call_llm(prompt)
            score = self._parse_score(response)
            
            if score is None:
                return self._create_result(
                    None,
                    error=f"Could not parse score from response: {response}"
                )
            
            return self._create_result(
                score=score,
                details={
                    "provider": self.provider,
                    "model": self.model,
                    "raw_response": response,
                    "num_contexts": len(sample.retrieved_docs)
                }
            )
        except Exception as e:
            return self._create_result(None, error=str(e))


class AnswerRelevanceMetric(LLMMetric):
    """Evaluate if answer is relevant to the query.
    
    Measures how well the answer addresses the user's question.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="AnswerRelevance", **kwargs)
        self.requires_answer = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute answer relevance score.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with relevance score
        """
        self.validate_sample(sample)
        
        if self.provider != "mock" and not self.api_key and self.client is None:
            return self._create_result(None, error=self._missing_provider_error())
        
        prompt = f"""Evaluate if the answer is relevant to the query.

A relevant answer:
- Directly addresses the question asked
- Provides information the user is seeking
- Stays on topic

Query:
{sample.query}

Answer:
{sample.generated_answer}

Provide a relevance score from 0.0 to 1.0:
- 1.0 = Highly relevant, directly answers the query
- 0.7 = Mostly relevant, addresses main points
- 0.5 = Partially relevant, some useful information
- 0.3 = Barely relevant, mostly off-topic
- 0.0 = Not relevant, completely off-topic

Score (0.0-1.0):"""
        
        try:
            response = self._call_llm(prompt)
            score = self._parse_score(response)
            
            if score is None:
                return self._create_result(
                    None,
                    error=f"Could not parse score from response: {response}"
                )
            
            return self._create_result(
                score=score,
                details={
                    "provider": self.provider,
                    "model": self.model,
                    "raw_response": response
                }
            )
        except Exception as e:
            return self._create_result(None, error=str(e))


class CoherenceMetric(LLMMetric):
    """Evaluate answer coherence and readability.
    
    Measures logical flow, grammar, and clarity of the answer.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="Coherence", **kwargs)
        self.requires_answer = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute coherence score.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with coherence score
        """
        self.validate_sample(sample)
        
        if self.provider != "mock" and not self.api_key and self.client is None:
            return self._create_result(None, error=self._missing_provider_error())
        
        prompt = f"""Evaluate the coherence and readability of the following answer.

A coherent answer:
- Has logical flow and structure
- Uses proper grammar and punctuation
- Is clear and easy to understand
- Connects ideas smoothly

Answer:
{sample.generated_answer}

Provide a coherence score from 0.0 to 1.0:
- 1.0 = Excellent coherence, very clear and well-structured
- 0.7 = Good coherence, minor issues
- 0.5 = Moderate coherence, some confusing parts
- 0.3 = Poor coherence, hard to follow
- 0.0 = Incoherent, incomprehensible

Score (0.0-1.0):"""
        
        try:
            response = self._call_llm(prompt)
            score = self._parse_score(response)
            
            if score is None:
                return self._create_result(
                    None,
                    error=f"Could not parse score from response: {response}"
                )
            
            return self._create_result(
                score=score,
                details={
                    "provider": self.provider,
                    "model": self.model,
                    "raw_response": response
                }
            )
        except Exception as e:
            return self._create_result(None, error=str(e))


class CompletenessMetric(LLMMetric):
    """Evaluate if answer is complete and comprehensive.
    
    Measures whether the answer fully addresses all aspects of the query.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="Completeness", **kwargs)
        self.requires_answer = True
    
    def compute(self, sample: RAGSample) -> MetricResult:
        """Compute completeness score.
        
        Args:
            sample: RAG sample
            
        Returns:
            MetricResult with completeness score
        """
        self.validate_sample(sample)
        
        if self.provider != "mock" and not self.api_key and self.client is None:
            return self._create_result(None, error=self._missing_provider_error())
        
        prompt = f"""Evaluate if the answer completely addresses the query.

A complete answer:
- Addresses all parts of the question
- Provides sufficient detail
- Doesn't leave important aspects unanswered

Query:
{sample.query}

Answer:
{sample.generated_answer}

Provide a completeness score from 0.0 to 1.0:
- 1.0 = Fully complete, addresses everything
- 0.7 = Mostly complete, minor gaps
- 0.5 = Partially complete, missing some aspects
- 0.3 = Incomplete, missing major aspects
- 0.0 = Very incomplete, barely addresses query

Score (0.0-1.0):"""
        
        try:
            response = self._call_llm(prompt)
            score = self._parse_score(response)
            
            if score is None:
                return self._create_result(
                    None,
                    error=f"Could not parse score from response: {response}"
                )
            
            return self._create_result(
                score=score,
                details={
                    "provider": self.provider,
                    "model": self.model,
                    "raw_response": response
                }
            )
        except Exception as e:
            return self._create_result(None, error=str(e))


# Aliases for backward compatibility
Faithfulness = FaithfulnessMetric
Relevance = AnswerRelevanceMetric
Coherence = CoherenceMetric
