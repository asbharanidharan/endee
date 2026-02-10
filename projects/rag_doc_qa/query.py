from sentence_transformers import SentenceTransformer
from endee_client import EndeeVectorClient

model = SentenceTransformer("all-MiniLM-L6-v2")

def query_endee(endee, question):
    q_emb = model.encode([question])[0]
    results = endee.search(q_emb, top_k=2)
    return results
