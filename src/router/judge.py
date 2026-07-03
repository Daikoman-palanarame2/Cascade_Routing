import re
from src.models.clients import LLMClient

# Guidelines say output only the digit, but search for it to be robust
JUDGE_PROMPT = (
    "You are an accuracy evaluator. Score the proposed answer to the question.\n"
    "Return ONLY a single integer 0-5. Do not write explanation, context, or markdown. Output only the digit.\n"
    "0 = completely wrong, 1 = mostly wrong, 2 = partially correct, "
    "3 = mostly correct with gaps, 4 = correct, 5 = correct and complete.\n\n"
    "Question: {q}\n\nProposed answer: {a}\n\nScore:"
)

class JudgeGate:
    def __init__(self, client: LLMClient):
        self.client = client

    async def score(self, question: str, answer: str) -> float:
        """
        Evaluate the answer's quality, returning a float from 0.0 to 1.0.
        """
        # If answer is blank, it's completely wrong (score 0.0)
        if not answer.strip():
            return 0.0
            
        try:
            resp = await self.client.generate(
                [{"role": "user", "content": JUDGE_PROMPT.format(q=question, a=answer)}],
                temperature=0.0, max_tokens=10,
            )
            text = resp.text.strip()
            
            # Robust extraction of the first digit from 0 to 5 in the output text
            match = re.search(r"\b([0-5])\b", text)
            if not match:
                # Fallback to search any occurrence of 0-5 if word bounds fail
                match = re.search(r"([0-5])", text)
                
            if match:
                raw = int(match.group(1))
            else:
                raw = 0
        except Exception:
            # Safe default on failure
            raw = 0
            
        return max(0.0, min(1.0, raw / 5.0))
