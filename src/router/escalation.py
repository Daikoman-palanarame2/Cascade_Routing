from src.models.clients import LLMClient

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question accurately and concisely. "
    "If the question requires reasoning, show your work step by step."
)

class RemoteEscalator:
    def __init__(self, client: LLMClient):
        self.client = client

    async def escalate(self, task: str, cache_key: str = "cascade-shared-prefix") -> dict:
        """
        Escalate the task to the remote model on Fireworks, requesting prompt caching.
        """
        resp = await self.client.generate(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ],
            temperature=0.0,
            max_tokens=1024,
            cache_key=cache_key,   # Stable cache key helps trigger Fireworks prompt caching
        )
        
        # Fireworks token scoring
        prompt_tokens = resp.prompt_tokens
        cached_tokens = resp.cached_tokens
        completion_tokens = resp.completion_tokens
        
        # By default, compute total raw token count (uncached + cached + completion)
        # Note: If the leaderboard counts the discounted cost (where cached tokens are billed at 50%),
        # we can adjust this calculation. For now, report total raw tokens:
        tokens_paid = prompt_tokens + completion_tokens
        
        return {
            "answer": resp.text,
            "tokens_paid": tokens_paid,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
        }
