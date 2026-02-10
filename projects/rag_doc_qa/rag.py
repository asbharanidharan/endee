from sentence_transformers import SentenceTransformer
from ingest import load_documents
from endee_client import EndeeVectorClient

model = SentenceTransformer("all-MiniLM-L6-v2")

if __name__ == "__main__":
    docs = load_documents()
    embeddings = model.encode(docs)

    endee = EndeeVectorClient()
    endee.insert(embeddings, docs)

    question = input("Ask a question: ")
    query_embedding = model.encode([question])[0]

    results = endee.search(query_embedding, top_k=3)

    print("\nRetrieved context:\n")
    for i, (score, text) in enumerate(results, 1):
        print(f"{i}. {text[:300]}...\n")
