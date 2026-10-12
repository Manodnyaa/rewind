import json
import os
import re
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"   # silence telemetry noise

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import settings

DATA_PATH = Path(__file__).parent.parent / "data" / "transcripts.json"

_embedder = None
_collection = None
_bm25 = None
_chunks = []


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())   # words only: strips . , ? ! etc.


def normalize_chunk(chunk: dict, index: int) -> dict:
    text = chunk.get("text") or chunk.get("evidence") or chunk.get("question") or ""
    return {
        **chunk,
        "text": text,
        "title": chunk.get("title") or "Generated eval item",
        "video_id": chunk.get("video_id") or f"eval-{index}",
        "start_seconds": chunk.get("start_seconds", 0),
    }


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def build_index():
    global _collection, _bm25, _chunks

    with open(DATA_PATH, encoding="utf-8") as f:
        _chunks = [normalize_chunk(c, i) for i, c in enumerate(json.load(f))]

    texts = [c["text"] for c in _chunks]

    client = chromadb.Client()
    try:
        client.delete_collection("lectures")
    except Exception:
        pass
    _collection = client.create_collection(
        "lectures", metadata={"hnsw:space": "cosine"}
    )
    _collection.add(
        ids=[str(i) for i in range(len(_chunks))],
        embeddings=get_embedder().encode(texts).tolist(),
        documents=texts,
    )

    _bm25 = BM25Okapi([tokenize(t) for t in texts])


def vector_search(query, k=5):
    q = get_embedder().encode([query]).tolist()
    res = _collection.query(query_embeddings=q, n_results=k)
    out = []
    for id_, dist in zip(res["ids"][0], res["distances"][0]):
        chunk = dict(_chunks[int(id_)])
        chunk["score"] = 1 - dist          # cosine similarity
        out.append(chunk)
    return out


def keyword_search(query, k=5):
    scores = _bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**_chunks[i], "score": float(scores[i])} for i in top]
