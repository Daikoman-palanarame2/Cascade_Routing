import pytest
import asyncio
from unittest.mock import AsyncMock
from src.router.verifier import CISCVerifier
from src.models.clients import LLMClient, LLMResponse

class MockLLMClient:
    def __init__(self):
        # List of sample answers we want our mock to return
        self.sample_idx = 0
        self.sample_responses = []
        
    async def generate(self, messages, temperature=0.0, max_tokens=1024, cache_key=None):
        content = messages[-1]["content"]
        
        # If it is a P-true confidence check prompt
        if "Is the proposed answer correct and complete?" in content:
            # If the answer is "Madrid" return YES, otherwise return NO
            if "Madrid" in content:
                return LLMResponse(text="YES", prompt_tokens=10, completion_tokens=1, cached_tokens=0)
            return LLMResponse(text="NO", prompt_tokens=10, completion_tokens=1, cached_tokens=0)
            
        # Standard sample generation call
        if self.sample_responses:
            resp_text = self.sample_responses[self.sample_idx % len(self.sample_responses)]
            self.sample_idx += 1
            return LLMResponse(text=resp_text, prompt_tokens=20, completion_tokens=5, cached_tokens=0)
            
        return LLMResponse(text="Madrid", prompt_tokens=20, completion_tokens=5, cached_tokens=0)

@pytest.mark.asyncio
async def test_verifier_consensus():
    mock_client = MockLLMClient()
    mock_client.sample_responses = ["Madrid", "Madrid", "Barcelona"]
    
    # 3 samples: 2 say Madrid (which gets YES -> P_true 1.0), 1 says Barcelona (gets NO -> P_true 0.0)
    verifier = CISCVerifier(client=mock_client, n_samples=3)
    
    result = await verifier.verify("What is the capital of Spain?", [{"role": "user", "content": "What is the capital of Spain?"}])
    
    assert result["answer"] == "Madrid"
    # Weight of Madrid = 1.0 + 1.0 = 2.0. Weight of Barcelona = 0.0.
    # Agreement = 2.0 / 2.0 = 1.0
    assert result["agreement"] == 1.0
    assert result["mean_p_true"] == pytest.approx(2/3) # 2/3 YES, 1/3 NO

@pytest.mark.asyncio
async def test_verifier_no_consensus():
    mock_client = MockLLMClient()
    mock_client.sample_responses = ["Paris", "Berlin", "Rome"]
    
    # All get NO from the mock (P_true 0.0)
    verifier = CISCVerifier(client=mock_client, n_samples=3)
    
    result = await verifier.verify("What is the capital of Spain?", [{"role": "user", "content": "What is the capital of Spain?"}])
    
    # Fallback to the first sample "Paris"
    assert result["answer"] == "Paris"
    assert result["agreement"] == pytest.approx(1/3) # weight 0.0 / total 0.0 defaults to 1/3
    assert result["mean_p_true"] == 0.0
