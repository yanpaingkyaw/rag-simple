"""Shared RAG functions: chunking, embedding, similarity, and generation.

This module contains the core building blocks of the RAG pipeline.
Both ingest.py and ask.py import from here.
"""

import json
import sys
from pathlib import Path

import numpy as np
import ollama

# --- Configuration ---
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "index"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

CHUNK_SIZE = 500  # max characters per chunk
OVERLAP = 100     # characters shared between consecutive chunks


def check_ollama():
    """Verify Ollama is running and required models are installed. Exit with a helpful message if not."""
    try:
        models = [m["model"].split(":")[0] for m in ollama.list()["models"]]
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama.\n{e}\n")
        print("Make sure Ollama is installed and running:")
        print("  1. Download from https://ollama.com")
        print("  2. Run: ollama serve")
        sys.exit(1)

    missing = [m for m in (EMBED_MODEL, CHAT_MODEL) if m not in models]
    if missing:
        print("ERROR: Missing Ollama models:")
        for m in missing:
            print(f"  ollama pull {m}")
        sys.exit(1)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks of roughly chunk_size characters.

    Overlap ensures a sentence spanning two chunks isn't lost from both.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def embed(text):
    """Convert text into an embedding vector using Ollama."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(response["embedding"], dtype=np.float32)


def cosine_similarity(a, b):
    """Measure how similar two vectors are (1.0 = identical meaning, 0.0 = unrelated)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def save_index(embeddings, chunks):
    """Save the embedding matrix and chunk metadata to the index/ folder."""
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    CHUNKS_FILE.write_text(json.dumps(chunks, indent=2), encoding="utf-8")


def load_index():
    """Load the embedding matrix and chunk metadata. Exit with a helpful message if missing."""
    if not EMBEDDINGS_FILE.exists() or not CHUNKS_FILE.exists():
        print("ERROR: No index found. Run this first:")
        print("  python ingest.py")
        sys.exit(1)
    embeddings = np.load(EMBEDDINGS_FILE)
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return embeddings, chunks


def retrieve(question, embeddings, chunks, top_k=3):
    """Find the top_k most relevant chunks for a question."""
    q_vector = embed(question)
    scores = [cosine_similarity(q_vector, emb) for emb in embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": chunks[i]["text"], "source": chunks[i]["source"], "score": float(scores[i])}
        for i in top_indices
    ]


def build_prompt(question, context_chunks):
    """Build a prompt that grounds the LLM in the retrieved context."""
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    return f"""You are a helpful teaching assistant. Answer the question using ONLY the context below.
If the context does not contain enough information to answer, say "I don't have enough information to answer that question."
Always mention which source document(s) your answer comes from.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def generate_answer(question, context_chunks):
    """Generate an answer using the LLM with retrieved context."""
    prompt = build_prompt(question, context_chunks)
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]
