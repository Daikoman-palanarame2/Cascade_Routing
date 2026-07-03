import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import src.config as config

class SemanticCache:
    def __init__(self, db_path: str = "./cache.db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        # SentenceTransformer handles downloading and caching the model locally
        self.model = SentenceTransformer(model_name)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT UNIQUE,
                embedding TEXT,
                answer TEXT,
                confidence REAL
            )
        """)
        conn.commit()
        conn.close()

    def lookup(self, query: str) -> str | None:
        """Return cached answer if similarity >= threshold, else None."""
        try:
            # Encode query and normalize for cosine similarity via dot product
            query_emb = self.model.encode(query, normalize_embeddings=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT query, embedding, answer FROM semantic_cache")
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return None
                
            best_sim = -1.0
            best_answer = None
            
            for row_query, emb_str, answer in rows:
                emb = np.array(json.loads(emb_str))
                # Normalized vectors dot product is exactly the cosine similarity
                sim = float(np.dot(query_emb, emb))
                if sim > best_sim:
                    best_sim = sim
                    best_answer = answer
                    
            if best_sim >= config.CACHE_SIMILARITY_THRESHOLD:
                return best_answer
            return None
        except Exception:
            return None

    def store(self, query: str, answer: str, confidence: float):
        """Store query, normalized embedding, answer, and confidence in the cache."""
        try:
            query_emb = self.model.encode(query, normalize_embeddings=True).tolist()
            emb_str = json.dumps(query_emb)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO semantic_cache (query, embedding, answer, confidence) VALUES (?, ?, ?, ?)",
                (query, emb_str, answer, confidence)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
