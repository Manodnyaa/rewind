import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from groq import Groq

from app.config import settings
from app.evaluate import contains

DATA = Path(__file__).parent.parent / "data"

# Generic off-topic questions. If your playlist covers any of these topics, edit the list.
OFF_TOPIC = [
    "What is the capital of France?",
    "Give me a good pizza recipe",
    "How do I train for a marathon?",
    "How does photosynthesis work?",
    "Who won the football world cup in 2010?",
]

PROMPT = """Below is an excerpt from a lecture transcript.
1. Write ONE question a student might ask that this excerpt answers. Paraphrase: do NOT reuse
   distinctive wording from the excerpt.
2. Copy ONE exact contiguous quote of 5 to 12 words from the excerpt that contains the key answer.
Return only JSON: {"question": "...", "evidence": "..."}

EXCERPT:
"""


def pick_chunks(chunks, n, seed):
    """Round-robin across videos so every video is covered; skip very short chunks."""
    rng = random.Random(seed)
    by_video = defaultdict(list)
    for c in chunks:
        if len(c["text"].split()) >= 25:
            by_video[c["video_id"]].append(c)
    for group in by_video.values():
        rng.shuffle(group)
    picked = []
    while len(picked) < n and any(by_video.values()):
        for vid in list(by_video):
            if by_video[vid] and len(picked) < n:
                picked.append(by_video[vid].pop())
    return picked


def generate(chunk, client):
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": PROMPT + chunk["text"]}],
            temperature=0.3,
            response_format={"type": "json_object"},
            extra_body={"reasoning_effort": "low"},    # delete this line if the API rejects it
        )
        data = json.loads(resp.choices[0].message.content)
        question, evidence = data["question"].strip(), data["evidence"].strip()
    except Exception as e:
        print(f"  generation failed: {e}")
        return None
    if len(evidence.split()) < 4 or not contains(chunk["text"], evidence):   # quote must really exist
        print(f"  discarded (evidence not found in chunk): {evidence!r}")
        return None
    return {
        "question": question,
        "evidence": evidence,
        "text": chunk["text"],
        "title": chunk["title"],
        "video_id": chunk["video_id"],
        "start_seconds": chunk["start_seconds"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(DATA / "eval_generated.json"))
    args = parser.parse_args()

    chunks = json.load(open(DATA / "transcripts.json", encoding="utf-8"))
    client = Groq(api_key=settings.groq_api_key)
    items = [x for x in (generate(c, client) for c in pick_chunks(chunks, args.n, args.seed)) if x]
    items += [
        {
            "question": q,
            "should_refuse": True,
            "text": q,
            "title": "Off-topic eval question",
            "video_id": "off-topic",
            "start_seconds": 0,
        }
        for q in OFF_TOPIC
    ]

    Path(args.out).write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(items)} items to {args.out}. REVIEW THEM before trusting the numbers:\n")
    for it in items:
        print("-", it["question"])


if __name__ == "__main__":
    main()
