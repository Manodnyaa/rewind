from app.vectorstore import vector_search
from app.config import settings


def retrieve(query: str) -> list[dict]:
    results = vector_search(query, k=settings.top_k)
    if not results or results[0]["score"] < settings.min_similarity:
        return []          # empty list = "nothing relevant in these lectures"
    return results