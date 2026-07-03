from pydantic import BaseModel, Field
from typing import Any, Optional, Dict

class TaskRequest(BaseModel):
    task_id: str
    prompt: str
    context: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    task_id: str
    answer: str
    tier_used: str                    # "cache" | "local" | "escalated"
    tokens_paid: int                  # Fireworks tokens only
    confidence: float
    trace: Dict[str, Any]             # full decision trace for dashboard
