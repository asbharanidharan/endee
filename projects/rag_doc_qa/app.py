import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.append(str(PROJECT_DIR))

import streamlit as st
from sentence_transformers import SentenceTransformer
from ingest import load_documents
from endee_client import EndeeVectorClient

st.set_page_config(page_title="RAG using Endee", layout="centered")

st.title("📄 RAG-based Document Q&A")
st.caption("Vector search powered by Endee (conceptual integration)")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

docs = load_documents()
embeddings = model.encode(docs)

endee = EndeeVectorClient()
endee.insert(embeddings, docs)

st.success(f"Ingested {len(docs)} chunks into vector store")

question = st.text_input("Ask a question")

if st.button("Search") and question:
    q_emb = model.encode([question])[0]
    results = endee.search(q_emb, top_k=3)

    st.subheader("🔎 Retrieved Context")
    for i, (score, text) in enumerate(results, 1):
        st.markdown(f"**Result {i}:**")
        st.write(text[:300] + "...")
