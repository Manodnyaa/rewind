from app.vectorstore import build_index, vector_search

build_index()
questions = [
    "what if I go blank",
    "how to structure a short presentation",
    "how do I handle a hard question",
    "what is the capital of France",
    "how does backpropagation work",
    "best pizza recipe",
]
for q in questions:
    top = vector_search(q, 1)[0]
    print(round(top["score"], 3), "|", q)