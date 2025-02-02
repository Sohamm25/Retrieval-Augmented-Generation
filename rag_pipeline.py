from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Example documents
documents = [
    "The Eiffel Tower is in Paris.",
    "The Great Wall of China is one of the Seven Wonders of the World.",
    "Python is a powerful programming language used in AI."
]

# Encode and index documents
doc_embeddings = embedder.encode(documents)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Initialize text generation model
generator = pipeline("text2text-generation", model="google/flan-t5-small")

# Query
query = "Where is the Eiffel Tower located?"
query_embedding = embedder.encode([query])

# Retrieve best matching document
D, I = index.search(np.array(query_embedding), k=1)
best_match = documents[I[0][0]]

# Generate response
input_text = f"Answer the question: {query} Context: {best_match}"
response = generator(input_text)

print(f"Query: {query}")
print(f"Best Match: {best_match}")
print(f"Generated Answer: {response[0]['generated_text']}")
