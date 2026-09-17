"""One Qdrant client configuration shared by retrieval and ingestion."""

from qdrant_client import QdrantClient

from app.config import settings


def get_qdrant_client() -> QdrantClient:
    """Use persistent local storage for demos; use Qdrant Cloud when configured."""
    if settings.QDRANT_MODE == "local":
        return QdrantClient(path=settings.QDRANT_LOCAL_PATH)
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
