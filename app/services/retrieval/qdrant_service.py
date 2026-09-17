import logfire
import time
from app.config import settings
from app.services.retrieval.embedding import embed_query
from app.services.retrieval.qdrant_client import get_qdrant_client


# Initialize Qdrant Client
client = get_qdrant_client()

def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        query_vector = embed_query(query)
    except Exception as e:
        logfire.error(f"❌ Query embedding failed: {e}")
        return []

    # Cloud connections can be reset transiently. Reuse the already-produced
    # embedding and retry only the inexpensive vector search.
    for attempt in range(3):
        try:
            response = client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                limit=limit,
                with_payload=True,  # JSON
            )
            return [
                {
                    "content": res.payload.get("text", ""),
                    "source": res.payload.get("source", "Unknown"),
                    "score": res.score,
                }
                for res in response.points
            ]
        except Exception as e:
            if attempt == 2:
                logfire.error(f"❌ Qdrant Search Failed after 3 attempts: {e}")
                return []
            wait = 1 + attempt
            logfire.warning(f"Qdrant search attempt {attempt + 1} failed; retrying in {wait}s: {e}")
            time.sleep(wait)
