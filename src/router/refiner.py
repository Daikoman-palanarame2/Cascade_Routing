from src.models.clients import LLMClient

CRITIQUE_PROMPT = (
    "You are a strict reviewer. Given the question and your draft answer, "
    "identify any errors, missing information, or unclear reasoning. "
    "Be concise (max 3 bullet points).\n\n"
    "Question: {q}\n\nDraft answer: {a}\n\nCritique:"
)

REFINE_PROMPT = (
    "Improve your answer based on the critique. Output only the final revised answer.\n\n"
    "Question: {q}\n\nDraft answer: {a}\n\nCritique: {c}\n\nRevised answer:"
)

class SelfRefiner:
    def __init__(self, client: LLMClient, max_rounds: int = 1):
        self.client = client
        self.max_rounds = max_rounds

    async def refine(self, question: str, draft: str) -> dict:
        """
        Refine a draft answer using a critique-and-refine round loop.
        """
        # If draft is empty, return it directly or let the model refine it
        if not draft.strip():
            return {"answer": draft, "rounds": 0}
            
        answer = draft
        rounds_run = 0
        for _ in range(self.max_rounds):
            try:
                critique = await self.client.generate(
                    [{"role": "user", "content": CRITIQUE_PROMPT.format(q=question, a=answer)}],
                    temperature=0.0, max_tokens=256,
                )
                revised = await self.client.generate(
                    [{"role": "user", "content": REFINE_PROMPT.format(
                        q=question, a=answer, c=critique.text)}],
                    temperature=0.0, max_tokens=512,
                )
                answer = revised.text
                rounds_run += 1
            except Exception:
                # If model call fails, keep the current best answer
                break
                
        return {"answer": answer, "rounds": rounds_run}
