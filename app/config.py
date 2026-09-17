import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Keep provider names consistent. GROK_API_KEY was an accidental typo that
    # left the guardrail model without credentials in normal deployments.
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL", "openai/gpt-oss-20b")
    
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "RAG"
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = "rag1"
    GROQ_SLUG_2 = "rag1"
    PORTKEY_CONFIG_SLUG=os.getenv("PORTKEY_CONFIG_SLUG")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
    API_KEY = os.getenv("API_KEY")
    ALLOWED_ORIGINS = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
        if origin.strip()
    ]

    def validate_runtime_settings(self) -> None:
        """Fail fast with actionable configuration errors at API startup."""
        required = {
            "GROQ_API_KEY": self.GROQ_API_KEY,
            "PORTKEY_API_KEY": self.PORTKEY_API_KEY,
            "QDRANT_CLUSTER_ENDPOINT": self.QDRANT_URL,
            "QDRANT_API_KEY": self.QDRANT_API_KEY,
        }
        missing = [name for name, value in required.items() if not value]
        if self.ENVIRONMENT == "production" and not self.API_KEY:
            missing.append("API_KEY")
        if missing:
            raise RuntimeError("Missing required environment variables: " + ", ".join(missing))
    
    
settings = Settings()
