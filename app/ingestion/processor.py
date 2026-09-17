import os 
import sys 
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts , get_embedding_dim
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service")

PROCESSED_DATA_DIR = "processed_data"

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def _source_filter(source_type: str, filename: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key="source", match=models.MatchValue(value=filename)),
            models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type)),
        ]
    )


def _assert_collection_dimension() -> None:
    """Reject model/collection dimension drift before indexing corrupts a run."""
    collection = qdrant_client.get_collection(settings.QDRANT_COLLECTION)
    vectors = collection.config.params.vectors
    configured_size = vectors.size if hasattr(vectors, "size") else None
    active_size = get_embedding_dim()
    if configured_size != active_size:
        raise RuntimeError(
            f"Collection '{settings.QDRANT_COLLECTION}' uses {configured_size}-dim vectors, "
            f"but the active embedding model produces {active_size}. Re-ingest with --wipe "
            "after selecting one embedding provider."
        )

def save_processed_locally(data:dict,source_type:str,filename:str)->str:
    folder = os.path.join(PROCESSED_DATA_DIR,source_type)
    os.makedirs(folder,exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str):
    """Parse → chunk → save locally → embed → index in Qdrant."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return {"filename": filename, "status": "skipped", "reason": "Unsupported file type"}

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return {"filename": filename, "status": "skipped", "reason": "No extractable text"}

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return {"filename": filename, "status": "skipped", "reason": "No chunks created"}

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")
            with logfire.span("Vectorizing & Indexing"):
                _assert_collection_dimension()
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        # Stable IDs make re-ingestion idempotent for unchanged chunks.
                        id=str(uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{source_type}:{filename}:{index}:{chunk}",
                        )),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for index, (chunk, vector) in enumerate(zip(chunks, embeddings))
                ]

                # Replace a source atomically enough for this single-writer CLI:
                # old chunks disappear before the new set is upserted, preventing
                # duplicates when a file changes or the chunking strategy changes.
                qdrant_client.delete(
                    collection_name=settings.QDRANT_COLLECTION,
                    points_selector=models.FilterSelector(
                        filter=_source_filter(source_type, filename)
                    ),
                )

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")
                return {"filename": filename, "status": "indexed", "chunks": len(points)}

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            return {"filename": filename, "status": "error", "reason": str(e)}
            
                     
def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)

def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Recreate collection — dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )
        

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)

if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    # Fix: Extract the specific string element from the list
    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    try:
        run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
        logfire.info("Ingestion job completed.")
    finally:
        # Closes the connection pool cleanly, removing the RuntimeWarning
        qdrant_client.close()
