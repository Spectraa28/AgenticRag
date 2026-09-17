from typing import List
import logfire

def chunk_text(text:str, chunk_size: int =  1500) -> List[str]:
    """Paragraph-aware splitter that never emits an oversized chunk."""
    with logfire.span("✂️ Text Splitter",text_length=len(text)):
        if not text.strip():
            return []
        
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # A long paragraph is split on word boundaries so it cannot blow the
            # embedding request or model context budget on its own.
            pieces = []
            while len(paragraph) > chunk_size:
                split_at = paragraph.rfind(" ", 0, chunk_size)
                split_at = split_at if split_at > 0 else chunk_size
                pieces.append(paragraph[:split_at].strip())
                paragraph = paragraph[split_at:].strip()
            pieces.append(paragraph)

            for piece in pieces:
                separator = "\n\n" if current_chunk else ""
                if len(current_chunk) + len(separator) + len(piece) <= chunk_size:
                    current_chunk += separator + piece
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = piece
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
