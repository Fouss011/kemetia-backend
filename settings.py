from pydantic import BaseModel
import os

class Settings(BaseModel):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    EMBED_TOPK: int = int(os.getenv("EMBED_TOPK", "8"))
    RERANK_WEIGHT: float = float(os.getenv("RERANK_WEIGHT", "0.25"))

SET = Settings()
