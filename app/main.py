# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
import secrets
import threading
import time
import json
import shutil
from pathlib import Path
from collections import defaultdict, deque
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.config import settings
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard

from pydantic import BaseModel
from pydantic import Field


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

RATE_LIMIT_REQUESTS = 120
RATE_LIMIT_WINDOW_SECONDS = 60
_request_times: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = threading.Lock()


@app.on_event("startup")
def startup_event():
    settings.validate_runtime_settings()
    initialize_rails()


class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=4_000)
    thread_id: str = Field(default="default_user", min_length=1, max_length=128)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require an API key when one is configured; keep local development simple."""
    if settings.API_KEY and not secrets.compare_digest(x_api_key or "", settings.API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def enforce_rate_limit(request: Request) -> None:
    """Small dependency-free per-process limiter for the public query endpoint."""
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_limit_lock:
        timestamps = _request_times[client]
        while timestamps and timestamps[0] <= now - RATE_LIMIT_WINDOW_SECONDS:
            timestamps.popleft()
        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again shortly.",
            )
        timestamps.append(now)
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/health")
def health():
    return {"status": "ok", "environment": settings.ENVIRONMENT}


@app.get("/eval/dataset", dependencies=[Depends(verify_api_key)])
def evaluation_dataset():
    """Expose evaluation coverage for the UI without importing the heavy RAGAS stack."""
    from evals.pipeline import load_golden_dataset

    dataset = load_golden_dataset()
    return {
        "rag_samples": dataset["rag_samples"],
        "guardrails_samples": dataset["guardrails_samples"],
    }


@app.post("/eval/run", dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])
def run_evaluation():
    """Run live retrieval and guardrail evaluation from a controlled UI action."""
    from evals.guardrails_eval import compute_guardrails_metrics, run_guardrails_eval
    from evals.pipeline import load_golden_dataset, run_pipeline

    dataset = run_pipeline(load_golden_dataset())
    guardrails = run_guardrails_eval(dataset["guardrails_samples"])
    return {
        "rag_samples": dataset["rag_samples"],
        "guardrails": compute_guardrails_metrics(guardrails),
        "guardrail_results": guardrails,
    }


@app.post("/ingest", dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])
def ingest_documents(
    files: list[UploadFile] = File(...),
    source_type: str = Form(default="general"),
):
    """Accept a small batch of supported documents and add them to the knowledge base."""
    allowed_extensions = {".pdf", ".html", ".htm", ".txt", ".docx", ".pptx"}
    safe_source_type = "".join(char for char in source_type.lower() if char.isalnum() or char in "_-")[:40] or "general"
    upload_dir = Path("uploaded_data") / safe_source_type
    upload_dir.mkdir(parents=True, exist_ok=True)
    results = []

    from app.ingestion.processor import process_file

    for upload in files:
        filename = Path(upload.filename or "upload").name
        if Path(filename).suffix.lower() not in allowed_extensions:
            results.append({"filename": filename, "status": "skipped", "reason": "Unsupported file type"})
            continue
        destination = upload_dir / filename
        with destination.open("wb") as target:
            shutil.copyfileobj(upload.file, target)
        outcome = process_file(str(destination), filename, safe_source_type)
        results.append(outcome or {"filename": filename, "status": "indexed"})

    return {"results": results}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query", dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)
        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to process the request. Please try again later.",
        ) from e


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/query/stream", dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])
def stream_query(request: QueryRequest):
    """Server-sent events endpoint for clients that render response tokens live."""
    def events():
        q = request.q
        thread_id = request.thread_id
        yield _sse("progress", {"label": "Checking request safety", "value": 12})

        try:
            rail_fired, rail_response = guard(q)
            if rail_fired:
                yield _sse("progress", {"label": "Request handled by guardrails", "value": 100})
                yield _sse("answer", {"text": rail_response, "sources": [], "plan": ["Intent: Guardrails Fired", "Retrieval: Skipped"]})
                yield _sse("done", {})
                return

            yield _sse("progress", {"label": "Planning retrieval strategy", "value": 30})
            yield _sse("progress", {"label": "Searching enterprise knowledge", "value": 55})
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = {
                "messages": [{"role": "user", "content": q}],
                "current_query": q,
                "documents": [],
                "plan": ["Start"],
                "status": "Initializing Graph...",
            }
            final_output = rag_agent.invoke(initial_state, config=config)
            answer = final_output.get("final_answer") or "I could not generate an answer."
            yield _sse("progress", {"label": "Synthesizing response", "value": 82})
            for start in range(0, len(answer), 18):
                yield _sse("delta", {"text": answer[start:start + 18]})
            yield _sse("answer", {
                "text": answer,
                "sources": final_output.get("documents", []),
                "plan": final_output.get("plan", []),
                "status": final_output.get("status", "Response generated."),
            })
            yield _sse("progress", {"label": "Response ready", "value": 100})
            yield _sse("done", {})
        except Exception as e:
            logfire.error(f"❌ Streaming backend execution failed: {e}")
            yield _sse("error", {"message": "Unable to process the request. Please try again later."})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
