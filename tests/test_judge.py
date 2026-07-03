import pytest
from unittest.mock import AsyncMock, MagicMock
from src.router.judge import JudgeGate
from src.models.clients import LLMClient, LLMResponse

@pytest.mark.asyncio
async def test_judge_empty_answer():
    mock_client = MagicMock(spec=LLMClient)
    judge = JudgeGate(client=mock_client)
    
    score = await judge.score("What is 10 + 10?", " ")
    assert score == 0.0
    mock_client.generate.assert_not_called()

@pytest.mark.asyncio
async def test_judge_successful_score():
    mock_client = MagicMock(spec=LLMClient)
    
    # Mock response containing the digit 4
    resp = LLMResponse(
        text="The answer is correct. Score: 4",
        prompt_tokens=25,
        completion_tokens=5,
        cached_tokens=0
    )
    mock_client.generate = AsyncMock(return_value=resp)
    
    judge = JudgeGate(client=mock_client)
    score = await judge.score("What is the capital of France?", "Paris")
    
    # 4 / 5.0 = 0.8
    assert score == 0.8
    mock_client.generate.assert_called_once()

@pytest.mark.asyncio
async def test_judge_no_digit_fallback():
    mock_client = MagicMock(spec=LLMClient)
    
    # Mock response containing no digit
    resp = LLMResponse(
        text="Incorrect answer with no score provided.",
        prompt_tokens=25,
        completion_tokens=5,
        cached_tokens=0
    )
    mock_client.generate = AsyncMock(return_value=resp)
    
    judge = JudgeGate(client=mock_client)
    score = await judge.score("What is the capital of France?", "Berlin")
    
    # Should fallback to 0.0
    assert score == 0.0
    mock_client.generate.assert_called_once()
