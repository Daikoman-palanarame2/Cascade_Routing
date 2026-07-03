import pytest
from unittest.mock import AsyncMock, MagicMock
from src.router.escalation import RemoteEscalator
from src.models.clients import LLMClient, LLMResponse

@pytest.mark.asyncio
async def test_escalator_successful_escalation():
    mock_client = MagicMock(spec=LLMClient)
    
    # Mock remote model response with token usage metadata
    resp = LLMResponse(
        text="This is a detailed reasoning answer from Gemma 27B.",
        prompt_tokens=120,
        completion_tokens=80,
        cached_tokens=40
    )
    mock_client.generate = AsyncMock(return_value=resp)
    
    escalator = RemoteEscalator(client=mock_client)
    result = await escalator.escalate("Solve the Riemann Hypothesis.")
    
    assert result["answer"] == "This is a detailed reasoning answer from Gemma 27B."
    assert result["prompt_tokens"] == 120
    assert result["completion_tokens"] == 80
    assert result["cached_tokens"] == 40
    # tokens_paid should be prompt_tokens + completion_tokens (120 + 80 = 200)
    assert result["tokens_paid"] == 200
    mock_client.generate.assert_called_once_with(
        messages=[
            {"role": "system", "content": "You are a precise assistant. Answer the user's question accurately and concisely. If the question requires reasoning, show your work step by step."},
            {"role": "user", "content": "Solve the Riemann Hypothesis."}
        ],
        temperature=0.0,
        max_tokens=1024,
        cache_key="cascade-shared-prefix"
    )
