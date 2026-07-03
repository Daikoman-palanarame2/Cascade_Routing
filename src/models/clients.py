from openai import AsyncOpenAI
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import src.config as config

@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int

class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        # Allow passing empty api_key if not needed (e.g. local vllm)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self.model = model

    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.0,
        max_tokens: int = 1024, 
        cache_key: Optional[str] = None
    ) -> LLMResponse:
        if config.MOCK_LLM:
            import random
            prompt_content = messages[-1]["content"]
            
            # 1. CISC Verifier P-true check
            if "Is the proposed answer correct and complete?" in prompt_content:
                # Return YES for Madrid/Paris to satisfy tests, otherwise 75% YES
                if any(x in prompt_content for x in ["Madrid", "Paris", "Spain", "France"]):
                    text = "YES"
                elif any(x in prompt_content for x in ["Barcelona", "Rome", "Berlin"]):
                    text = "NO"
                else:
                    text = "YES" if random.random() < 0.75 else "NO"
                return LLMResponse(text=text, prompt_tokens=15, completion_tokens=1, cached_tokens=0)
                
            # 2. LLM-as-Judge score check
            elif "You are an accuracy evaluator." in prompt_content:
                # Return a score from 3 to 5
                score = str(random.choice([3, 4, 5]))
                return LLMResponse(text=score, prompt_tokens=25, completion_tokens=1, cached_tokens=0)
                
            # 3. Self-Refine critique
            elif "You are a strict reviewer." in prompt_content:
                text = "- Review comments: The response looks solid.\n- Verify specific historical details if needed."
                return LLMResponse(text=text, prompt_tokens=30, completion_tokens=15, cached_tokens=0)
                
            # 4. Self-Refine update
            elif "Improve your answer based on the critique." in prompt_content:
                # Extract draft answer from prompt if possible
                text = "Madrid" if "Madrid" in prompt_content else "Paris" if "Paris" in prompt_content else "Refined mock answer."
                return LLMResponse(text=text, prompt_tokens=50, completion_tokens=20, cached_tokens=0)
                
            # 5. General task solver
            else:
                text = "Madrid" if "Spain" in prompt_content else "Paris" if "France" in prompt_content else f"Mock answer to: '{prompt_content[:50]}'"
                return LLMResponse(text=text, prompt_tokens=20, completion_tokens=10, cached_tokens=0)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Fireworks supports prompt caching key
        if cache_key:
            kwargs["extra_body"] = {"prompt_cache_key": cache_key}
            
        resp = await self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        
        # Get cached tokens if present in response usage metadata
        cached_tokens = 0
        if usage:
            cached_tokens = getattr(usage, "prompt_cached_tokens", 0) or 0
            # Also handle prompt_cache_hit_tokens or similar depending on client details
            if not cached_tokens and hasattr(usage, "extra_fields"):
                extra = getattr(usage, "extra_fields", {})
                cached_tokens = extra.get("prompt_cached_tokens", 0)
                
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            cached_tokens=cached_tokens,
        )

# Local client initialization (simulated or real local vLLM)
if config.SIMULATE_LOCAL:
    # If simulating, route to Fireworks with the local model ID (e.g. google/gemma-3-4b-it)
    # The actual API key is required
    local_client = LLMClient(
        base_url=config.FIREWORKS_BASE_URL,
        api_key=config.FIREWORKS_API_KEY,
        model=config.LOCAL_MODEL,
    )
else:
    local_client = LLMClient(
        base_url=config.LOCAL_VLLM_URL,
        api_key="",
        model=config.LOCAL_MODEL,
    )

# Remote client initialization (Fireworks)
remote_client = LLMClient(
    base_url=config.FIREWORKS_BASE_URL,
    api_key=config.FIREWORKS_API_KEY,
    model=config.REMOTE_MODEL,
)
