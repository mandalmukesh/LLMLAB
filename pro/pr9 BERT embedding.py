from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model_sbert.encode([s[0] for s in sentences])
query = "Great product"
query_embedding = model_sbert.encode(query)
similarities = cosine_similarity([query_embedding], embeddings)
index = np.argmax(similarities)
print("Best Match:", sentences[index][0])