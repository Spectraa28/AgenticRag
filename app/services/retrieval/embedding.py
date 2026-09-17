import time
import os
import re
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

# The free Gemini embedding quota is easy to exhaust with a 50-chunk burst.
# Smaller, paced batches make local ingestion dependable (and resumable).
BATCH_SIZE = int(os.getenv("GEMINI_EMBED_BATCH_SIZE", "10"))
MIN_BATCH_INTERVAL_SECONDS = float(os.getenv("GEMINI_EMBED_BATCH_INTERVAL_SECONDS", "4"))
MAX_RATE_LIMIT_RETRIES = 5
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  

_active_model = None
_model_type: str | None = None
_next_gemini_request_at = 0.0


def _provider_retry_delay(error: Exception, attempt: int) -> float:
    """Use Google's supplied cooldown when present, with exponential fallback."""
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(error))
    provider_delay = float(match.group(1)) if match else 0.0
    return max(provider_delay, float(2 ** attempt))


def _wait_for_gemini_slot() -> None:
    global _next_gemini_request_at
    wait = _next_gemini_request_at - time.monotonic()
    if wait > 0:
        logfire.info(f"Pacing Gemini embedding batch for {wait:.1f}s.")
        time.sleep(wait)

def _probe_gemini():
    """Try one embed call to verify gemini is reachable . Returns  model or None"""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY
        )
        model.embed_query("probe")
        logfire.info("GEmini embedding ready (gemini-embedding-2-preview,  3072 dim")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed : {e} . will use sentence transformer fallback")
        return None
    
    
def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence transformer fallback (all-mpnet-base-v2, 768 dim)")
    return SentenceTransformer("all-mpnet-base-v2")

def _init():
    """Initialise embedding model once per process . CAlled Lazily on first use"""
    global _active_model, _model_type
    if _active_model is not None:
        return
    
    gemini  = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"
 

def get_embedding_dim() -> int:
    """Returns the vector dimension for the active model . call after the init()"""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


def _embed_batch(batch: list[str]) -> list[list[float]]:
    global _next_gemini_request_at
    if _model_type == "gemini":
        # Respect the free-tier throughput and the provider's RetryInfo hint.
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            try:
                _wait_for_gemini_slot()
                embeddings = _active_model.embed_documents(batch)
                _next_gemini_request_at = time.monotonic() + MIN_BATCH_INTERVAL_SECONDS
                return embeddings
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < MAX_RATE_LIMIT_RETRIES - 1:
                    wait = _provider_retry_delay(e, attempt)
                    _next_gemini_request_at = time.monotonic() + wait
                    logfire.warning(
                        f"Gemini rate limit hit — retrying in {wait:g}s "
                        f"(attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES})."
                    )
                    _wait_for_gemini_slot()
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError(f"Gemini rate limit persisted after {MAX_RATE_LIMIT_RETRIES} attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
