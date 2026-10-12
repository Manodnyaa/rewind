import argparse
import json
import re
from pathlib import Path

from app.config import settings
from app.vectorstore import build_index, vector_search
from app.retrieval import retrieve

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "eval_questions.json"


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def contains(text: str, phrase: str) -> bool:
    return f" {normalize(phrase)} " in f" {normalize(text)} "


def chunk_end_seconds(chunk: dict) -> float:
    return float(chunk.get("end_seconds", chunk["start_seconds"] + settings.chunk_window_seconds))


def is_hit(chunk: dict, item: dict) -> bool:
    if "evidence" in item:                                   # playlist-independent labels
        return contains(chunk["text"], item["evidence"])
    return (chunk["video_id"] == item["video_id"]            # legacy time-window labels
            and float(chunk["start_seconds"]) < float(item["start_max"])
            and chunk_end_seconds(chunk) > float(item["start_min"]))


def run(path: Path):
    build_index()
    items = json.load(open(path, encoding="utf-8"))
    hits, rr, answerable, offtopic, refused = 0, [], 0, 0, 0
    ans_scores, off_scores = [], []

    for it in items:
        top = vector_search(it["question"], k=1)
        top_score = top[0]["score"] if top else 0.0          # same score the guardrail uses
        results = retrieve(it["question"])

        if it.get("should_refuse"):
            offtopic += 1
            off_scores.append(top_score)
            ok = results == []
            refused += ok
            print(f"{'PASS' if ok else 'FAIL'} [refuse] (top sim {top_score:.2f}) {it['question']}")
            continue

        answerable += 1
        ans_scores.append(top_score)
        rank = next((i + 1 for i, r in enumerate(results) if is_hit(r, it)), None)
        hits += bool(rank)
        rr.append(1 / rank if rank else 0)
        print(f"{'PASS rank ' + str(rank) if rank else 'FAIL'} [answer] (top sim {top_score:.2f}) {it['question']}")

    if answerable:
        print(f"\nHit Rate@K : {hits}/{answerable} = {hits / answerable:.0%}")
        print(f"MRR        : {sum(rr) / len(rr):.3f}")
    if offtopic:
        print(f"Refusal    : {refused}/{offtopic} off-topic questions correctly refused")

    if ans_scores and off_scores:
        lo, hi = min(ans_scores), max(off_scores)
        print(f"\nThreshold check: lowest on-topic top score {lo:.3f}, highest off-topic {hi:.3f}, "
              f"current min_similarity {settings.min_similarity}")
        if lo > hi:
            print(f"Suggested MIN_SIMILARITY is about {(lo + hi) / 2:.2f} (set it in .env)")
        else:
            print("Warning: on-topic and off-topic scores overlap, so no single threshold separates them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(DEFAULT_PATH))
    run(Path(parser.parse_args().file))