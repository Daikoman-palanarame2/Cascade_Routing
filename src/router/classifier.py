import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os

class DifficultyClassifier:
    def __init__(self, embedder_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedder_model)
        self.clf = None

    def fit(self, queries: list[str], labels: list[int]):
        """
        Train the classifier.
        labels: 1 if local model succeeded on task, 0 if it failed/needs escalation.
        """
        if not queries:
            return
        X = self.embedder.encode(queries, normalize_embeddings=True)
        # Use balanced weights because of potentially skewed distributions of easy/hard tasks
        base = LogisticRegression(max_iter=1000, class_weight="balanced")
        
        # Check class distribution to determine calibration cross-validation strategy
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2 or np.min(counts) < 2:
            # Fallback to direct LogisticRegression if data is too small to calibrate via CV
            self.clf = base
            self.clf.fit(X, labels)
        else:
            # CalibratedClassifierCV ensures predicted probability is a reliable P(local correct)
            cv_folds = int(min(5, np.min(counts)))
            self.clf = CalibratedClassifierCV(estimator=base, cv=cv_folds, method="isotonic")
            self.clf.fit(X, labels)

    def predict_p_easy(self, query: str) -> float:
        """
        Return P(local model correct) in [0, 1].
        If not trained, return 0.5 (neutral prior).
        """
        if self.clf is None:
            return 0.5
        try:
            x = self.embedder.encode([query], normalize_embeddings=True)
            return float(self.clf.predict_proba(x)[0, 1])
        except Exception:
            return 0.5

    def save(self, path: str):
        """Save the trained classifier to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.clf, path)

    def load(self, path: str):
        """Load a trained classifier from a file."""
        if os.path.exists(path):
            self.clf = joblib.load(path)
        else:
            self.clf = None
