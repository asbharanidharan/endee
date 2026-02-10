from pathlib import Path
from sentence_transformers import SentenceTransformer
from endee_client import EndeeVectorClient

DATA_DIR = Path(__file__).parent / "data"
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def load_documents():
    all_chunks = []

    for file in DATA_DIR.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

    return all_chunks

if __name__ == "__main__":
    docs = load_documents()
    embeddings = model.encode(docs)

    endee = EndeeVectorClient()
    endee.insert(embeddings, docs)

    print(f"Ingested {len(docs)} chunks into Endee vector store")
