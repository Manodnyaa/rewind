from app.vectorstore import build_index, vector_search, keyword_search

build_index()
for q in ["how to structure a short presentation", "what if I go blank"]:
    print("\nQUERY:", q)
    print("-- vector")
    for r in vector_search(q, 3):
        print(round(r["score"], 3), int(r["start_seconds"]), r["text"][:90])
    print("-- keyword")
    for r in keyword_search(q, 3):
        print(round(r["score"], 3), int(r["start_seconds"]), r["text"][:90])