import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI
from app.config import settings

# ============================================================
# Production gateway config (saved in Portkey dashboard):
#   - Fallback: @rag1/llama-3.3-70b-versatile → @rag1/llama-3.1-8b-instant on failure
#   - Cache: simple mode
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target
#
# NOTE: Inline configs are disabled on this Portkey account. The config
# above is saved in the Portkey dashboard and referenced here by its
# slug (PORTKEY_CONFIG_SLUG) instead of being passed as a raw dict.
# To edit the fallback/cache/retry behavior, update the saved config
# in the Portkey dashboard directly.
# ============================================================
PORTKEY_CONFIG_SLUG = settings.PORTKEY_CONFIG_SLUG  # optional dashboard configuration
MODEL = f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag1/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=MODEL,
        temperature=0,
        # Groq can briefly reject bursts from an evaluation run.  Retrying at
        # the model boundary preserves the normal LangChain interface and does
        # not require callers to understand provider-specific errors.
        max_retries=3,
        timeout=90,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
