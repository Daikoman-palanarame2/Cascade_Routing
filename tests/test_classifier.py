import pytest
import os
from src.router.classifier import DifficultyClassifier

@pytest.fixture
def temp_classifier_path():
    path = "./test_classifier.joblib"
    if os.path.exists(path):
        os.remove(path)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

def test_classifier_fit_and_predict(temp_classifier_path):
    clf = DifficultyClassifier()
    
    # Simple training data: 
    # queries about math/grammar are easy (1), queries about advanced topics are hard (0)
    queries = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "How do you spell dynamic?",
        "Define gravity",
        "Derive the black scholes partial differential equation",
        "Explain the biochemical pathways of the citric acid cycle",
        "Implement a lock-free queue in Rust using atomic pointers",
        "Prove the Riemann hypothesis from scratch"
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    
    # Train
    clf.fit(queries, labels)
    
    # Predict easy query
    p_easy = clf.predict_p_easy("What is 3 + 3?")
    # Predict hard query
    p_hard = clf.predict_p_easy("Prove Goldbach conjecture using analytic number theory")
    
    assert p_easy > p_hard

def test_save_and_load(temp_classifier_path):
    clf = DifficultyClassifier()
    queries = ["Simple math question", "Complex quantum mechanics proof"]
    labels = [1, 0]
    clf.fit(queries, labels)
    
    clf.save(temp_classifier_path)
    assert os.path.exists(temp_classifier_path)
    
    new_clf = DifficultyClassifier()
    new_clf.load(temp_classifier_path)
    
    p_easy = new_clf.predict_p_easy("Simple math question")
    assert p_easy is not None
    assert 0.0 <= p_easy <= 1.0
