import os
from dataclasses import dataclass
from typing import Dict, Any
import src.config as config
from src.cache.semantic_cache import SemanticCache
from src.router.classifier import DifficultyClassifier
from src.router.verifier import CISCVerifier
from src.router.refiner import SelfRefiner
from src.router.judge import JudgeGate
from src.router.escalation import RemoteEscalator
from src.models.clients import local_client, remote_client

@dataclass
class RoutingResult:
    answer: str
    tier: str                 # "cache" | "local" | "escalated"
    tokens_paid: int
    confidence: float
    trace: Dict[str, Any]

class RoutingPipeline:
    def __init__(
        self, 
        cache: SemanticCache, 
        classifier: DifficultyClassifier, 
        verifier: CISCVerifier, 
        refiner: SelfRefiner, 
        judge: JudgeGate, 
        escalator: RemoteEscalator, 
        threshold: float
    ):
        self.cache = cache
        self.classifier = classifier
        self.verifier = verifier
        self.refiner = refiner
        self.judge = judge
        self.escalator = escalator
        self.threshold = threshold

    @classmethod
    def from_env(cls):
        cache = SemanticCache()
        clf = DifficultyClassifier()
        clf_path = "src/calibration/classifier.joblib"
        try:
            if os.path.exists(clf_path):
                clf.load(clf_path)
        except Exception:
            pass  # Fallback to untrained neutral prior if load fails
            
        return cls(
            cache=cache,
            classifier=clf,
            verifier=CISCVerifier(local_client, n_samples=config.SELF_CONSISTENCY_N),
            refiner=SelfRefiner(local_client, max_rounds=config.SELF_REFINE_ROUNDS),
            judge=JudgeGate(local_client),
            escalator=RemoteEscalator(remote_client),
            threshold=config.ESCALATION_THRESHOLD,
        )

    async def solve(self, task: str, task_id: str) -> RoutingResult:
        # 1. Semantic cache lookup (0 tokens paid, 0 local tokens)
        cached = self.cache.lookup(task)
        if cached is not None:
            return RoutingResult(
                answer=cached, 
                tier="cache", 
                tokens_paid=0, 
                confidence=1.0,
                trace={"stage": "cache_hit"}
            )

        # 2. Difficulty classifier prediction (free local CPU)
        p_easy = self.classifier.predict_p_easy(task)

        # 3. Local generation + CISC voting (free local tokens)
        gen_messages = [{"role": "user", "content": task}]
        cisc = await self.verifier.verify(task, gen_messages)
        draft = cisc["answer"]

        # 4. Self-Refine critique and update round (free local tokens)
        refined = await self.refiner.refine(task, draft)
        answer = refined["answer"]

        # 5. LLM-as-Judge scoring (free local tokens)
        judge_score = await self.judge.score(task, answer)

        # 6. Calculate combined confidence score to decide if we escalate
        agreement = cisc["agreement"]
        combined = 0.3 * p_easy + 0.3 * agreement + 0.4 * judge_score

        if combined >= self.threshold:
            # We trust the local model's output; store in cache and return
            self.cache.store(task, answer, combined)
            return RoutingResult(
                answer=answer, 
                tier="local", 
                tokens_paid=0,
                confidence=combined,
                trace={
                    "p_easy": p_easy, 
                    "agreement": agreement,
                    "judge": judge_score, 
                    "combined": combined,
                    "escalated": False
                },
            )
            
        # 7. Escalate to remote (paid Fireworks API tokens)
        esc = await self.escalator.escalate(task)
        
        # Store remote response in cache with our low combined confidence tag
        self.cache.store(task, esc["answer"], combined)
        
        return RoutingResult(
            answer=esc["answer"], 
            tier="escalated",
            tokens_paid=esc["tokens_paid"], 
            confidence=combined,
            trace={
                "p_easy": p_easy, 
                "agreement": agreement,
                "judge": judge_score, 
                "combined": combined,
                "escalated": True, 
                "remote_tokens": esc["tokens_paid"],
                "prompt_tokens": esc["prompt_tokens"],
                "cached_tokens": esc["cached_tokens"],
                "completion_tokens": esc["completion_tokens"]
            },
        )
