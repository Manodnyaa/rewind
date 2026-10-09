from rank_bm25 import BM25Okapi

texts = [
    "The cat sat on the mat",
    "Gradient descent minimizes the loss function",
    "A feline rested on the rug",
    "Backpropagation computes gradients layer by layer",
]

tokenized = [t.lower().split() for t in texts]   # each text -> list of words
bm25 = BM25Okapi(tokenized)                      # build the index

for query in ["gradient descent", "how does neural network training work"]:
    scores = bm25.get_scores(query.lower().split())
    print(query, "->", scores)