import asyncio
import numpy as np
from collections import defaultdict
from src.models.clients import LLMClient, LLMResponse

P_TRUE_PROMPT = (
    "Question: {q}\nProposed answer: {a}\n"
    "Is the proposed answer correct and complete? "
    "Reply with a single token: YES or NO."
)

class CISCVerifier:
    def __init__(self, client: LLMClient, n_samples: int = 3):
        self.client = client
        self.n_samples = n_samples

    async def _sample(self, messages: list[dict], temperature: float) -> LLMResponse:
        return await self.client.generate(messages, temperature=temperature, max_tokens=512)

    async def _p_true(self, question: str, answer: str) -> float:
        try:
            resp = await self.client.generate(
                [{"role": "user", "content": P_TRUE_PROMPT.format(q=question, a=answer)}],
                temperature=0.0, max_tokens=10,
            )
            text = resp.text.strip().upper()
            return 1.0 if "YES" in text else 0.0
        except Exception:
            # Fallback if model call fails
            return 0.5

    async def verify(self, question: str, gen_messages: list[dict]) -> dict:
        """
        Sample N local answers, score each with P-true, and aggregate using weighted voting.
        """
        # 1. Sample N diverse completions
        samples = await asyncio.gather(*[
            self._sample(gen_messages, temperature=0.7) for _ in range(self.n_samples)
        ])
        
        # 2. Score each answer's confidence using P-true
        p_trues = await asyncio.gather(*[
            self._p_true(question, s.text) for s in samples
        ])
        
        # 3. Cluster answers by normalized text, weight by P-true
        clusters = defaultdict(float)
        best_answer = ""
        best_weight = -1.0
        
        for s, p in zip(samples, p_trues):
            key = self._normalize(s.text)
            clusters[key] += p
            if clusters[key] > best_weight:
                best_weight = clusters[key]
                best_answer = s.text
                
        # If all samples had a P-true of 0.0, pick the first sample's text as fallback
        if not best_answer and samples:
            best_answer = samples[0].text
            
        # 4. Agreement = top cluster weight / total weight of all clusters
        total = sum(clusters.values())
        agreement = (best_weight / total) if total > 0 else (1.0 / self.n_samples)
        
        return {
            "answer": best_answer,
            "agreement": min(1.0, max(0.0, agreement)),  # Bound between 0 and 1
            "mean_p_true": float(np.mean(p_trues)) if p_trues else 0.0,
        }

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize answer text for clustering by removing punctuation and limiting length."""
        return "".join(c.lower() for c in text if c.isalnum() or c.isspace()).strip()[:200]
