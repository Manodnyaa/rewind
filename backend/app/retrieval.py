from app.vectorstore import vector_search, keyword_search
from app.config import settings

RRF_K = 60   # standard smoothing constant


def reciprocal_rank_fusion(*ranked_lists, k=RRF_K):
    scores, lookup = {}, {}
    for results in ranked_lists:
        for rank, chunk in enumerate(results):
            key = (chunk["video_id"], chunk["start_seconds"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            lookup.setdefault(key, chunk)
    fused = []
    for key in sorted(scores, key=scores.get, reverse=True):
        chunk = dict(lookup[key])
        chunk["score"] = scores[key]
        fused.append(chunk)
    return fused


def retrieve(query: str) -> list[dict]:
    vec = vector_search(query, k=10)
    if not vec or vec[0]["score"] < settings.min_similarity:
        return []                       # guardrail uses vector similarity
    kw = [c for c in keyword_search(query, k=10) if c["score"] > 0]
    return reciprocal_rank_fusion(vec, kw)[: settings.top_k]