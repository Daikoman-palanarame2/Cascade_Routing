import pytest
import os
from unittest.mock import AsyncMock, MagicMock
from src.router.pipeline import RoutingPipeline, RoutingResult
from src.cache.semantic_cache import SemanticCache
from src.router.classifier import DifficultyClassifier
from src.router.verifier import CISCVerifier
from src.router.refiner import SelfRefiner
from src.router.judge import JudgeGate
from src.router.escalation import RemoteEscalator

@pytest.fixture
def mock_pipeline():
    cache = MagicMock(spec=SemanticCache)
    classifier = MagicMock(spec=DifficultyClassifier)
    verifier = MagicMock(spec=CISCVerifier)
    refiner = MagicMock(spec=SelfRefiner)
    judge = MagicMock(spec=JudgeGate)
    escalator = MagicMock(spec=RemoteEscalator)
    
    pipeline = RoutingPipeline(
        cache=cache,
        classifier=classifier,
        verifier=verifier,
        refiner=refiner,
        judge=judge,
        escalator=escalator,
        threshold=0.70
    )
    return pipeline

@pytest.mark.asyncio
async def test_pipeline_cache_hit(mock_pipeline):
    # Set up cache hit
    mock_pipeline.cache.lookup.return_value = "Cached Answer"
    
    result = await mock_pipeline.solve("Some query", "t_001")
    
    assert result.answer == "Cached Answer"
    assert result.tier == "cache"
    assert result.tokens_paid == 0
    mock_pipeline.cache.lookup.assert_called_once_with("Some query")
    mock_pipeline.classifier.predict_p_easy.assert_not_called()

@pytest.mark.asyncio
async def test_pipeline_local_pass(mock_pipeline):
    # Cache miss
    mock_pipeline.cache.lookup.return_value = None
    
    # High confidence signals: p_easy=0.8, agreement=0.8, judge_score=0.8
    # combined = 0.3*0.8 + 0.3*0.8 + 0.4*0.8 = 0.8 >= 0.70 (threshold)
    mock_pipeline.classifier.predict_p_easy.return_value = 0.8
    mock_pipeline.verifier.verify = AsyncMock(return_value={"answer": "Local Draft", "agreement": 0.8})
    mock_pipeline.refiner.refine = AsyncMock(return_value={"answer": "Refined Local Answer", "rounds": 1})
    mock_pipeline.judge.score = AsyncMock(return_value=0.8)
    
    result = await mock_pipeline.solve("Explain photosynthesis", "t_002")
    
    assert result.answer == "Refined Local Answer"
    assert result.tier == "local"
    assert result.tokens_paid == 0
    assert result.confidence == 0.8
    mock_pipeline.escalator.escalate.assert_not_called()
    mock_pipeline.cache.store.assert_called_once_with("Explain photosynthesis", "Refined Local Answer", 0.8)

@pytest.mark.asyncio
async def test_pipeline_escalation(mock_pipeline):
    # Cache miss
    mock_pipeline.cache.lookup.return_value = None
    
    # Low confidence signals: p_easy=0.4, agreement=0.4, judge_score=0.4
    # combined = 0.3*0.4 + 0.3*0.4 + 0.4*0.4 = 0.4 < 0.70 (threshold)
    mock_pipeline.classifier.predict_p_easy.return_value = 0.4
    mock_pipeline.verifier.verify = AsyncMock(return_value={"answer": "Weak Draft", "agreement": 0.4})
    mock_pipeline.refiner.refine = AsyncMock(return_value={"answer": "Refined Weak Answer", "rounds": 1})
    mock_pipeline.judge.score = AsyncMock(return_value=0.4)
    
    # Escalation result
    mock_pipeline.escalator.escalate = AsyncMock(return_value={
        "answer": "Fireworks Strong Answer",
        "tokens_paid": 150,
        "prompt_tokens": 100,
        "cached_tokens": 50,
        "completion_tokens": 50
    })
    
    result = await mock_pipeline.solve("Derive general relativity", "t_003")
    
    assert result.answer == "Fireworks Strong Answer"
    assert result.tier == "escalated"
    assert result.tokens_paid == 150
    assert result.confidence == 0.4
    mock_pipeline.escalator.escalate.assert_called_once_with("Derive general relativity")
    mock_pipeline.cache.store.assert_called_once_with("Derive general relativity", "Fireworks Strong Answer", 0.4)
