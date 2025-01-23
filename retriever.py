from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example docs
documents = [
    "The Eiffel Tower is in Paris.",
    "The Great Wall of China is visible from space.",
    "Python is a versatile programming language."
]
# Encode 
doc_embeddings = model.encode(documents)
# Initialize FAISS index -->
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

query = "Where is the Eiffel Tower located?"
query_embedding = model.encode([query])

D, I = index.search(np.array(query_embedding), k=1)
print(f"Query: {query}")
print(f"Best match: {documents[I[0][0]]}")
