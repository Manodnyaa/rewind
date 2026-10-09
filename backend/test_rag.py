import asyncio
from app.vectorstore import build_index
from app.rag import answer_stream

build_index()


async def ask(q):
    print("\n\nQ:", q)
    async for ev in answer_stream(q, []):
        if ev["event"] == "citations":
            print(f"[citations] {len(ev['data'])} chunks")
        elif ev["event"] == "model":
            print(f"[model] {ev['data']}")
        elif ev["event"] == "token":
            print(ev["data"], end="", flush=True)
        else:
            print(f"\n[{ev['event']}] {str(ev['data'])[:80]}")


for q in ["what if I go blank",
          "why is less more in a short presentation",
          "what is the capital of France"]:
    asyncio.run(ask(q))