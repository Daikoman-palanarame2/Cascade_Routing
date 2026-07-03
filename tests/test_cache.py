import pytest
import os
import shutil
from src.cache.semantic_cache import SemanticCache

@pytest.fixture
def temp_cache():
    # Set up a test database file
    db_path = "./test_cache.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    cache = SemanticCache(db_path=db_path)
    yield cache
    
    # Tear down
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

def test_cache_miss(temp_cache):
    # Lookup in empty cache should return None
    assert temp_cache.lookup("What is the capital of Sweden?") is None

def test_cache_hit_exact(temp_cache):
    # Store an answer
    query = "What is the capital of Spain?"
    answer = "Madrid"
    temp_cache.store(query, answer, 1.0)
    
    # Exact lookup
    hit = temp_cache.lookup(query)
    assert hit == answer

def test_cache_hit_similar(temp_cache):
    import src.config as config
    original_threshold = config.CACHE_SIMILARITY_THRESHOLD
    config.CACHE_SIMILARITY_THRESHOLD = 0.90
    try:
        # Store an answer
        query = "What is the capital of France?"
        answer = "Paris"
        temp_cache.store(query, answer, 1.0)
        
        # Semantic similarity lookup (near match)
        hit = temp_cache.lookup("what is the capital of France")
        assert hit == answer
    finally:
        config.CACHE_SIMILARITY_THRESHOLD = original_threshold

def test_cache_miss_different(temp_cache):
    # Store an answer
    query = "What is the capital of Italy?"
    answer = "Rome"
    temp_cache.store(query, answer, 1.0)
    
    # Semantic similarity lookup (not a match)
    hit = temp_cache.lookup("What is the population of Rome?")
    assert hit is None
