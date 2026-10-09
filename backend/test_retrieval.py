from app.vectorstore import build_index
from app.retrieval import retrieve

build_index()
for q in ["what if I go blank", "how to structure a short presentation",
          "how do I handle a hard question", "what is the capital of France",
          "how does backpropagation work", "best pizza recipe"]:
    print(len(retrieve(q)), "chunks |", q)