from fastapi import FastAPI, HTTPException
from src.api.schemas import TaskRequest, TaskResponse
from src.router.pipeline import RoutingPipeline

app = FastAPI(title="Free-Verify Cascade Routing Agent")

# Initialize routing pipeline
try:
    pipeline = RoutingPipeline.from_env()
except Exception as e:
    # Fallback to None if config/clients aren't fully set up during early startup
    pipeline = None

@app.post("/solve", response_model=TaskResponse)
async def solve(req: TaskRequest) -> TaskResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Routing pipeline not fully initialized.")
    try:
        result = await pipeline.solve(req.prompt, req.task_id)
        return TaskResponse(
            task_id=req.task_id,
            answer=result.answer,
            tier_used=result.tier,
            tokens_paid=result.tokens_paid,
            confidence=result.confidence,
            trace=result.trace,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing execution failed: {str(e)}")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline_initialized": pipeline is not None
    }
