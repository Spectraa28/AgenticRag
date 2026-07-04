import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "RAG"
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = "rag1"
    GROQ_SLUG_2 = "rag1"
    PORTKEY_CONFIG_SLUG=os.getenv("PORTKEY_CONFIG_SLUG")
    
    
settings = Settings()
    