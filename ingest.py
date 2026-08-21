"""Step 1 of the RAG pipeline: build the search index.

Loads documents from data/, splits them into chunks, embeds each chunk
with Ollama, and saves the result to index/.

Usage:
    python ingest.py
"""

import numpy as np

from rag import DATA_DIR, chunk_text, check_ollama, embed, save_index


def main():
    check_ollama()

    # 1. Load documents
    documents = []
    for filepath in sorted(DATA_DIR.glob("*.txt")):
        text = filepath.read_text(encoding="utf-8")
        documents.append({"source": filepath.name, "text": text})
        print(f"Loaded: {filepath.name} ({len(text)} characters)")

    if not documents:
        print(f"ERROR: No .txt files found in {DATA_DIR}")
        return

    # 2. Chunk each document
    all_chunks = []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "chunk_index": i,
            })
    print(f"\nCreated {len(all_chunks)} chunks from {len(documents)} documents")

    # 3. Embed every chunk (this is the slow part)
    print(f"\nEmbedding {len(all_chunks)} chunks...")
    embeddings = []
    for i, chunk in enumerate(all_chunks):
        embeddings.append(embed(chunk["text"]))
        if (i + 1) % 5 == 0 or i == len(all_chunks) - 1:
            print(f"  {i + 1}/{len(all_chunks)} done")
    embeddings = np.array(embeddings)

    # 4. Save the index to disk so ask.py can use it
    save_index(embeddings, all_chunks)
    print(f"\nIndex saved: {embeddings.shape[0]} chunks x {embeddings.shape[1]} dimensions")
    print("You can now ask questions:")
    print('  python ask.py "Which planet is the largest?"')


if __name__ == "__main__":
    main()
