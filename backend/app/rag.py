#answer-generatuions
from groq import AsyncGroq
from app.config import settings
from app.retrieval import retrieve

client = AsyncGroq(api_key=settings.groq_api_key)

NOT_FOUND = "I couldn't find that in these lectures."
COMPLEX_HINTS = ("why", "compare", "difference", "explain", "versus", "summarize", "relate")


def choose_model(question: str) -> str:
    q = question.lower()
    if len(q.split()) > 15 or any(h in q for h in COMPLEX_HINTS):
        return settings.llm_model      # 120B for reasoning-heavy questions
    return settings.fast_model         # 20B for short factual ones


def build_system_prompt(chunks: list[dict]) -> str:
    sources = "\n\n".join(
        f"[{i+1}] ({c['title']}, {int(c['start_seconds'])}s): {c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        "You are a lecture assistant. Answer using ONLY the sources below. "
        "Do not add advice, examples, or frameworks that are not stated in the sources. "
        "If the sources only partly answer the question, answer that part and say the rest "
        "isn't covered. If they don't answer it at all, say you don't know.\n"
        "Format rules: write short plain-text paragraphs, no markdown, no asterisks, no headings. "
        "Put citations like [1] or [2] right after the sentence they support, "
        "using only plain square brackets. Keep the answer under 120 words.\n\n"
        f"SOURCES:\n{sources}"
    )


async def answer_stream(question: str, history: list[dict]):
    chunks = retrieve(question)
    if not chunks:
        yield {"event": "token", "data": NOT_FOUND}
        yield {"event": "done", "data": ""}
        return

    yield {"event": "citations", "data": chunks}          # a list, not a JSON string

    primary = choose_model(question)
    other = settings.fast_model if primary == settings.llm_model else settings.llm_model
    recent = [{"role": h["role"], "content": h["content"]} for h in history[-4:]]
    messages = (
        [{"role": "system", "content": build_system_prompt(chunks)}]
        + recent
        + [{"role": "user", "content": question}]
    )

    for model in (primary, other):
        started = False
        try:
            yield {"event": "model", "data": model}
            stream = await client.chat.completions.create(
                model=model, messages=messages, stream=True, temperature=0.2,
                extra_body={"reasoning_effort": "low"},   # delete this line if the API rejects it
            )
            async for event in stream:
                delta = event.choices[0].delta.content
                if delta:
                    started = True
                    yield {"event": "token", "data": delta}
            yield {"event": "done", "data": ""}
            return
        except Exception as e:
            print(f"Model {model} failed: {e}")
            if started:
                break            # don't retry after partial output, it would duplicate text

    yield {"event": "error", "data": "The language model is unavailable right now. Please try again."}