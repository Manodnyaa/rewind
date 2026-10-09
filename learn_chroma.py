import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "The cat sat on the mat",
    "Gradient descent minimizes the loss function",
    "A feline rested on the rug",
    "Backpropagation computes gradients layer by layer",
]

embeddings = embedder.encode(texts).tolist()

client = chromadb.Client()
collection = client.get_or_create_collection("test_collection")

collection.add(
    ids=["0","1","2","3"],
    embeddings=embeddings,
    documents=texts,
)

query = "how does neural network training work"
query_vector = embedder.encode([query]).tolist()

results = collection.query(query_embeddings=query_vector, n_results=2)
print(results["documents"])
