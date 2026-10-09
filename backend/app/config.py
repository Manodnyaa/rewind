# backend/app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    chunk_window_seconds: int = 45
    chunk_overlap_seconds: int = 10
    min_rerank_score: float = -2.0
    top_k: int = 5
    min_similarity: float = 0.25
    embedding_model: str = "all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "openai/gpt-oss-120b"    # strong model, for complex questions
    fast_model: str = "openai/gpt-oss-20b"    # cheap, fast model, for short questions

    class Config:
        env_file = ".env"

settings = Settings()