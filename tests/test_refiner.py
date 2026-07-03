import pytest
from unittest.mock import AsyncMock, MagicMock
from src.router.refiner import SelfRefiner
from src.models.clients import LLMClient, LLMResponse

@pytest.mark.asyncio
async def test_refiner_empty_draft():
    # If the draft is empty or whitespace, it should return immediately without calling LLM
    mock_client = MagicMock(spec=LLMClient)
    refiner = SelfRefiner(client=mock_client, max_rounds=1)
    
    result = await refiner.refine("What is the capital of Spain?", "  ")
    
    assert result["answer"] == "  "
    assert result["rounds"] == 0
    mock_client.generate.assert_not_called()

@pytest.mark.asyncio
async def test_refiner_successful_refine():
    mock_client = MagicMock(spec=LLMClient)
    
    # Mock critique response
    critique_resp = LLMResponse(
        text="- Grammatical error: capital of Spain is Madrid, not Barcelona.",
        prompt_tokens=30,
        completion_tokens=15,
        cached_tokens=0
    )
    # Mock revised response
    revised_resp = LLMResponse(
        text="Madrid",
        prompt_tokens=50,
        completion_tokens=5,
        cached_tokens=0
    )
    
    # Configure mock client to return responses in sequence
    mock_client.generate = AsyncMock(side_effect=[critique_resp, revised_resp])
    
    refiner = SelfRefiner(client=mock_client, max_rounds=1)
    
    result = await refiner.refine("What is the capital of Spain?", "Barcelona")
    
    assert result["answer"] == "Madrid"
    assert result["rounds"] == 1
    assert mock_client.generate.call_count == 2
