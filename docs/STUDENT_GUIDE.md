# RAG From Scratch — Detailed Student Guide

A complete, code-level walkthrough of the `rag-simple` project: a Retrieval-Augmented
Generation (RAG) system built with only **Python, NumPy, and Ollama** — no frameworks,
no API keys, everything runs on your own machine.

> **How to use this guide:** read it side-by-side with the code. Every section points to
> the exact file and function it explains. Read in this order:
> `rag.py` → `ingest.py` → `ask.py`.

---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [Tech Stack](#2-tech-stack)
3. [Prerequisites & Setup](#3-prerequisites--setup)
4. [What Problem Does RAG Solve?](#4-what-problem-does-rag-solve)
5. [The Big Picture — Architecture](#5-the-big-picture--architecture)
6. [Key Concepts You Must Understand First](#6-key-concepts-you-must-understand-first)
7. [Project Structure](#7-project-structure)
8. [Code Walkthrough — `rag.py` (the toolbox)](#8-code-walkthrough--ragpy-the-toolbox)
9. [Code Walkthrough — `ingest.py` (build the index)](#9-code-walkthrough--ingestpy-build-the-index)
10. [Code Walkthrough — `ask.py` (ask questions)](#10-code-walkthrough--askpy-ask-questions)
11. [End-to-End Trace of One Question](#11-end-to-end-trace-of-one-question)
12. [Running the Project](#12-running-the-project)
13. [Exercises](#13-exercises)
14. [Glossary](#14-glossary)
15. [FAQ / Troubleshooting](#15-faq--troubleshooting)

---

## 1. Learning Goals

After studying this project you should be able to:

- Explain **why** LLMs need RAG (knowledge cutoff, private data, hallucination).
- Describe the two phases of RAG: **ingestion** (offline) and **querying** (online).
- Implement text **chunking** with overlap and explain why overlap matters.
- Explain what an **embedding** is and how **cosine similarity** compares meanings.
- Build a working **retrieval** step using nothing but NumPy.
- Write a **grounded prompt** that forces the LLM to answer only from context.

---

## 2. Tech Stack

The stack is deliberately minimal so every moving part is visible. There is **no
LangChain, no LlamaIndex, no cloud API, no database server** — just the pieces below.

| Layer | Technology | Role in this project |
|---|---|---|
| Language | **Python 3.10+** | All application code (~250 lines across 3 files). |
| Numerical library | **NumPy** (`numpy>=1.26.0`) | Stores the embedding matrix, computes dot products / norms for cosine similarity, `argsort` for ranking. |
| LLM runtime | **Ollama** (desktop app / server) | Runs both AI models locally on your machine and exposes them over a local HTTP API (`localhost:11434`). |
| Python client | **ollama** package (`ollama>=0.4.0`) | Thin Python wrapper used for the `ollama.list()`, `ollama.embeddings()`, and `ollama.chat()` calls in `rag.py`. |
| Embedding model | **nomic-embed-text** (~274 MB) | Converts text into 768-dimensional vectors. Used on every chunk at ingest time and on every question at ask time. |
| Chat model | **llama3.2** (3B parameters, ~2 GB) | Reads the retrieved context and writes the final answer. |
| "Vector database" | **Flat files** — `embeddings.npy` + `chunks.json` | The stored index. A NumPy binary file and a JSON file replace a real vector DB for teaching purposes. |
| Interface | **CLI** (`argparse`, `input()`) | `ingest.py` and `ask.py` are plain command-line scripts — no web server, no UI framework. |

Why this stack for teaching:

- **Everything runs locally** — no API keys, no cost per call, no data leaves your machine.
- **Two dependencies total** (`ollama`, `numpy`) — the entire `requirements.txt` is 2 lines.
- **No framework magic** — every RAG step (chunk, embed, search, prompt) is code you can read and modify.

---

## 3. Prerequisites & Setup

### 3.1 What you need before starting

| Requirement | Minimum | Notes |
|---|---|---|
| Operating system | Windows 10/11, macOS, or Linux | Ollama supports all three. |
| Python | 3.10 or newer | Check with `python --version`. |
| RAM | 8 GB (16 GB is comfortable) | llama3.2 (3B) needs roughly 4 GB free while answering. |
| Disk space | ~3 GB free | ~2 GB for llama3.2 + ~274 MB for nomic-embed-text + Ollama itself. |
| GPU | **Not required** | Everything runs on CPU; a GPU just makes it faster. |
| Internet | Only during setup | Needed once to download Ollama and pull the two models. After that, the whole system works offline. |
| Knowledge | Basic Python (functions, lists, dicts) | No machine-learning background needed — that's what this tutorial teaches. |

### 3.2 Step 1 — Install Ollama

Ollama is the engine that runs the AI models locally. Download the installer for your
OS from [https://ollama.com](https://ollama.com) and run it.

After installing, verify the server is running:

```bash
ollama --version    # prints the version
ollama list         # lists installed models (may be empty at first)
```

On Windows and macOS the Ollama app starts the server automatically in the background.
If `ollama list` says it cannot connect, start the server manually:

```bash
ollama serve
```

### 3.3 Step 2 — Pull the two models

```bash
ollama pull nomic-embed-text    # embedding model (~274 MB)
ollama pull llama3.2            # chat model (~2 GB — takes a few minutes)
```

Verify both appear:

```bash
ollama list
# NAME                    SIZE
# nomic-embed-text:latest 274 MB
# llama3.2:latest         2.0 GB
```

> `check_ollama()` in `rag.py` performs exactly this verification at startup and
> prints these same commands if anything is missing — see [section 8.2](#82-check_ollama--fail-fast-with-a-helpful-message).

### 3.4 Step 3 — Set up Python

From the `rag-simple/` folder:

```bash
# Create an isolated virtual environment
python -m venv .venv

# Activate it — Windows (PowerShell):
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install the two dependencies (ollama, numpy)
pip install -r requirements.txt
```

### 3.5 Step 4 — Verify everything works

```bash
python -c "import ollama, numpy; print('Python packages OK')"
python ingest.py     # should load 4 documents, embed ~28 chunks, save the index
python ask.py "Which planet is the largest?"
```

If all three commands succeed, your environment is ready. If not, jump to
[FAQ / Troubleshooting](#15-faq--troubleshooting).

---

## 4. What Problem Does RAG Solve?

A large language model (LLM) is trained once, on a fixed snapshot of public text.
That creates three problems:

| Problem | Example |
|---|---|
| **Knowledge cutoff** | The model doesn't know anything that happened after training. |
| **Private data** | It has never seen *your* class notes, company wiki, or PDFs. |
| **Hallucination** | When it doesn't know, it often invents a confident-sounding answer. |

**RAG (Retrieval-Augmented Generation)** solves all three with one idea:

> Before asking the LLM a question, first *search your own documents* for the most
> relevant passages, and paste them into the prompt as context. Then instruct the
> model: *"answer using ONLY this context."*

The model no longer needs to "know" the answer — it just needs to **read and
summarize** the context you handed it. That is much more reliable.

---

## 5. The Big Picture — Architecture

The system has **two separate phases** that run at different times:

```mermaid
flowchart TB
    subgraph Phase1["PHASE 1 — INGESTION (offline, run once) — ingest.py"]
        A[".txt files in data/"] --> B["chunk_text()<br/>split into 500-char pieces<br/>with 100-char overlap"]
        B --> C["embed()<br/>each chunk → 768-number vector<br/>(nomic-embed-text via Ollama)"]
        C --> D["save_index()<br/>index/embeddings.npy<br/>index/chunks.json"]
    end

    subgraph Phase2["PHASE 2 — QUERYING (online, every question) — ask.py"]
        E["User question"] --> F["embed()<br/>question → vector"]
        D -.->|load_index()| G
        F --> G["retrieve()<br/>cosine similarity vs every chunk<br/>keep top-k best matches"]
        G --> H["build_prompt()<br/>context + question + rules"]
        H --> I["generate_answer()<br/>llama3.2 via Ollama"]
        I --> J["Grounded answer<br/>with source citations"]
    end
```

Why two phases? **Embedding is slow** (one LLM call per chunk), so we do it once and
save the result to disk. Answering a question then only needs **one** embedding call
(for the question itself) plus one chat call.

---

## 6. Key Concepts You Must Understand First

### 6.1 Embeddings — "meaning as numbers"

An **embedding model** converts a piece of text into a fixed-length list of numbers
(a *vector*). This project uses `nomic-embed-text`, which produces **768 numbers**
per text.

The magic property: **texts with similar meaning get vectors that point in similar
directions**, even if they share no words.

```
"Jupiter is the largest planet"   → [0.12, -0.83, 0.44, ...]  (768 numbers)
"The biggest world orbiting Sun"  → [0.11, -0.79, 0.41, ...]  ← very close!
"How to bake chocolate cake"      → [-0.55, 0.20, -0.91, ...] ← far away
```

This is why RAG search works with *meaning*, not keyword matching.

### 6.2 Cosine similarity — "how aligned are two vectors?"

To compare two vectors we measure the **angle** between them:

\[ \text{similarity}(a, b) = \frac{a \cdot b}{\|a\| \, \|b\|} \]

- `1.0` → same direction → same meaning
- `0.0` → perpendicular → unrelated
- Typical scores in this project: **0.55–0.80** for a good match, below ~0.4 for noise.

We divide by the lengths (norms) so that only *direction* matters — a long document
and a short one about the same topic still score high.

### 6.3 Chunking — "why not embed the whole document?"

Two reasons we split documents into small pieces:

1. **Precision.** If you embed a whole 5-page document, its vector is an *average*
   of every topic inside it — the vector becomes blurry. Small chunks have sharp,
   specific meanings that match questions better.
2. **Prompt budget.** We paste retrieved chunks into the LLM prompt. Whole documents
   would not fit; small chunks do.

**Overlap** (100 chars here) means each chunk repeats the tail of the previous one, so
a sentence that would be cut in half at a boundary still appears complete in at least
one chunk.

---

## 7. Project Structure

```
rag-simple/
├── rag.py                 # Core toolbox: chunk, embed, similarity, retrieve, generate
├── ingest.py              # Phase 1: build the index from data/*.txt
├── ask.py                 # Phase 2: CLI to ask questions
├── requirements.txt       # ollama, numpy — that's all
├── data/                  # Source documents (4 sample .txt files)
│   ├── solar_system.txt
│   ├── photosynthesis.txt
│   ├── history_of_computers.txt
│   └── water_cycle.txt
├── index/                 # GENERATED by ingest.py — do not edit by hand
│   ├── embeddings.npy     # NumPy matrix: one 768-dim row per chunk
│   └── chunks.json        # The chunk texts + which file each came from
└── docs/
    ├── STUDENT_GUIDE.md   # This document
    └── RAG_Presentation.pptx
```

**Dependency direction:** `ingest.py` and `ask.py` both import from `rag.py`.
`rag.py` imports nothing from the other two. This keeps all "smart" logic in one place.

---

## 8. Code Walkthrough — `rag.py` (the toolbox)

This module holds every core building block. Nothing here runs by itself — it is a
library used by the two scripts.

### 8.1 Configuration constants

```14:24:rag.py
# --- Configuration ---
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "index"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

CHUNK_SIZE = 500  # max characters per chunk
OVERLAP = 100     # characters shared between consecutive chunks
```

Step by step:

- **`EMBED_MODEL`** — the Ollama model that turns text into vectors. Small (~274 MB), fast.
- **`CHAT_MODEL`** — the Ollama model that writes the final answer (llama3.2, 3B parameters).
- **`Path(__file__).parent`** — the folder containing `rag.py`. Using this instead of a
  relative string means the scripts work no matter which directory you run them from.
- **`CHUNK_SIZE = 500` / `OVERLAP = 100`** — the two knobs of chunking. Try changing
  them (Exercise 2) and re-running `ingest.py` to see how retrieval quality changes.

### 8.2 `check_ollama()` — fail fast with a helpful message

```27:43:rag.py
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
```

Step by step:

1. `ollama.list()` asks the local Ollama server which models are installed. If the
   server isn't running, this raises an exception → we print install instructions and exit.
2. Model names come back like `"llama3.2:latest"`, so `.split(":")[0]` strips the tag,
   leaving `"llama3.2"`.
3. We then check both required models are present; if not, we print the exact
   `ollama pull ...` commands the student needs to run.

**Design lesson:** *fail fast* at startup with an actionable message, instead of
crashing halfway through with a confusing stack trace.

### 8.3 `chunk_text()` — the sliding window

```46:59:rag.py
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
```

Step by step, with `chunk_size=500`, `overlap=100`:

1. Start a cursor at position `0`.
2. Take the slice `text[0:500]` → **chunk 1**.
3. Advance the cursor by `chunk_size - overlap = 400` → position `400`.
4. Take `text[400:900]` → **chunk 2**. Notice characters 400–500 appear in *both*
   chunks — that is the overlap.
5. Repeat until the cursor passes the end of the text. Python slicing is forgiving:
   `text[800:1300]` on a 1000-char text simply returns characters 800–999.
6. `if chunk.strip()` skips chunks that are pure whitespace; `.strip()` also removes
   leading/trailing whitespace from kept chunks.

Visualization of a 1,000-character document:

```
Position: 0         400       500       800  900       1000
          |----------|---------|---------|----|---------|
Chunk 1:  [======== 0 – 500 ========]
Chunk 2:             [======== 400 – 900 =======]
Chunk 3:                                 [== 800 – 1000 ==]
                     ^^^^ overlap ^^^^   ^^ overlap ^^
```

**Limitation to notice:** this splits at *character* positions, so it can cut a word
in half mid-chunk. Real systems split at sentence or paragraph boundaries — that is
Exercise 6.

### 8.4 `embed()` — text in, vector out

```62:65:rag.py
def embed(text):
    """Convert text into an embedding vector using Ollama."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(response["embedding"], dtype=np.float32)
```

Step by step:

1. Send the text to the local Ollama server with the `nomic-embed-text` model.
2. Ollama returns `{"embedding": [0.12, -0.83, ...]}` — a Python list of 768 floats.
3. Convert to a NumPy array of `float32` so later math (dot products) is fast and
   the saved file is half the size of `float64`.

The **same function** embeds document chunks (in `ingest.py`) and user questions
(in `retrieve()`). This is essential: both must live in the *same vector space* or
comparing them would be meaningless.

### 8.5 `cosine_similarity()` — one line of math

```68:70:rag.py
def cosine_similarity(a, b):
    """Measure how similar two vectors are (1.0 = identical meaning, 0.0 = unrelated)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

- `np.dot(a, b)` — multiply the vectors element-by-element and sum: \(\sum_i a_i b_i\).
- `np.linalg.norm(a)` — the vector's length: \(\sqrt{\sum_i a_i^2}\).
- Dividing by both lengths normalizes the result to the range roughly \([-1, 1]\),
  where only the **angle** between the vectors matters, not their size.

Tiny worked example with 2-D vectors:

```
a = [1, 0],  b = [1, 1]
dot(a, b)   = 1×1 + 0×1 = 1
|a| = 1,  |b| = √2 ≈ 1.414
similarity  = 1 / (1 × 1.414) ≈ 0.707     (45° angle)
```

### 8.6 `save_index()` and `load_index()` — the "database"

```73:88:rag.py
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
```

Two parallel files make up our "vector database":

| File | Contains | Format |
|---|---|---|
| `embeddings.npy` | One row of 768 floats per chunk (shape: `N × 768`) | NumPy binary |
| `chunks.json` | For each chunk: its `text`, `source` filename, `chunk_index` | JSON |

**Row `i` of the matrix corresponds to entry `i` of the JSON list.** That positional
pairing is the entire "database schema". Real systems (Chroma, Pinecone, pgvector)
do the same thing with more engineering around it.

`load_index()` shows the fail-fast pattern again: if you run `ask.py` before
`ingest.py`, you get told exactly what to do.

### 8.7 `retrieve()` — the heart of RAG

```91:99:rag.py
def retrieve(question, embeddings, chunks, top_k=3):
    """Find the top_k most relevant chunks for a question."""
    q_vector = embed(question)
    scores = [cosine_similarity(q_vector, emb) for emb in embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": chunks[i]["text"], "source": chunks[i]["source"], "score": float(scores[i])}
        for i in top_indices
    ]
```

Step by step:

1. **Embed the question** into the same 768-dim space as the chunks.
2. **Score every chunk**: compute cosine similarity between the question vector and
   each row of the embedding matrix. With ~30 chunks this brute-force loop is instant.
3. **`np.argsort(scores)`** returns the *indices* that would sort the scores
   ascending (worst first). Example: `argsort([0.2, 0.9, 0.5]) → [0, 2, 1]`.
4. **`[::-1]`** reverses it → descending (best first): `[1, 2, 0]`.
5. **`[:top_k]`** keeps the best `k` indices (default 3).
6. Build a result list carrying the chunk **text**, its **source** filename (so the
   answer can cite it), and the **score** (so students can inspect retrieval quality).

> **Scaling note:** at millions of chunks you would replace the loop with a vector
> database using approximate nearest-neighbor search (HNSW, FAISS). The *concept*
> stays identical.

### 8.8 `build_prompt()` — grounding the LLM

```102:116:rag.py
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
```

The prompt has three deliberate defenses against hallucination:

1. **"using ONLY the context below"** — forbids the model from using its general
   training knowledge.
2. **An explicit escape hatch** — the model is *told what to say* when the context
   is insufficient. Without this, LLMs tend to guess rather than admit ignorance.
3. **"mention which source document(s)"** — each chunk is labeled
   `[Source: filename]`, and the model must cite it. This gives users a way to verify.

Chunks are separated by `---` lines so the model can tell where one source ends and
the next begins.

### 8.9 `generate_answer()` — the final LLM call

```119:126:rag.py
def generate_answer(question, context_chunks):
    """Generate an answer using the LLM with retrieved context."""
    prompt = build_prompt(question, context_chunks)
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]
```

Step by step:

1. Assemble the grounded prompt.
2. Send it to `llama3.2` via Ollama's chat API as a single user message.
3. Return the model's text reply.

Note the **"G" in RAG is the smallest function in the file**. All the interesting
work happened before this point — that is the central lesson of the project.

---

## 9. Code Walkthrough — `ingest.py` (build the index)

Run with `python ingest.py`. Four numbered stages, matching the comments in the code.

### Stage 1 — Load documents

```18:27:ingest.py
    # 1. Load documents
    documents = []
    for filepath in sorted(DATA_DIR.glob("*.txt")):
        text = filepath.read_text(encoding="utf-8")
        documents.append({"source": filepath.name, "text": text})
        print(f"Loaded: {filepath.name} ({len(text)} characters)")

    if not documents:
        print(f"ERROR: No .txt files found in {DATA_DIR}")
        return
```

- `DATA_DIR.glob("*.txt")` finds every text file; `sorted(...)` makes the order
  deterministic across operating systems (important for reproducible chunk indices).
- Each document is stored as a dict `{"source": filename, "text": contents}` —
  the filename travels with the text so we can cite sources later.

### Stage 2 — Chunk each document

```29:38:ingest.py
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
```

Every chunk records **which file it came from** and **its position** within that file.
Documents are chunked independently — a chunk never spans two files.

### Stage 3 — Embed every chunk (the slow part)

```40:47:ingest.py
    # 3. Embed every chunk (this is the slow part)
    print(f"\nEmbedding {len(all_chunks)} chunks...")
    embeddings = []
    for i, chunk in enumerate(all_chunks):
        embeddings.append(embed(chunk["text"]))
        if (i + 1) % 5 == 0 or i == len(all_chunks) - 1:
            print(f"  {i + 1}/{len(all_chunks)} done")
    embeddings = np.array(embeddings)
```

- One Ollama call per chunk → this is why ingestion is a separate offline phase.
- Progress prints every 5 chunks (`(i + 1) % 5 == 0`) and always on the last one.
- `np.array(list_of_vectors)` stacks the individual 768-dim vectors into one
  `N × 768` **matrix** — the shape retrieval expects.

### Stage 4 — Save the index

```49:53:ingest.py
    # 4. Save the index to disk so ask.py can use it
    save_index(embeddings, all_chunks)
    print(f"\nIndex saved: {embeddings.shape[0]} chunks x {embeddings.shape[1]} dimensions")
    print("You can now ask questions:")
    print('  python ask.py "Which planet is the largest?"')
```

Expected output for the four sample documents (sizes are approximate):

```
Loaded: history_of_computers.txt (2418 characters)
Loaded: photosynthesis.txt (2093 characters)
Loaded: solar_system.txt (2214 characters)
Loaded: water_cycle.txt (2198 characters)

Created 28 chunks from 4 documents

Embedding 28 chunks...
  5/28 done
  ...
  28/28 done

Index saved: 28 chunks x 768 dimensions
```

**Remember:** re-run `python ingest.py` whenever you add, remove, or edit files in
`data/` — the index does not update itself.

---

## 10. Code Walkthrough — `ask.py` (ask questions)

### 10.1 `answer_question()` — retrieval + generation for one question

```16:25:ask.py
def answer_question(question, embeddings, chunks, top_k, show_context):
    context = retrieve(question, embeddings, chunks, top_k=top_k)

    if show_context:
        print("\nRetrieved chunks:")
        for c in context:
            print(f"  [{c['score']:.4f}] {c['source']}")

    answer = generate_answer(question, context)
    print(f"\nAnswer:\n{answer}\n")
```

1. Retrieve the `top_k` most relevant chunks.
2. Unless `--quiet` was passed, print each chunk's similarity score and source file —
   this **transparency** is the best debugging tool in RAG: if the answer is bad,
   first check whether retrieval picked the right chunks.
3. Generate and print the grounded answer.

### 10.2 `main()` — CLI arguments and the two modes

```28:37:ask.py
def main():
    parser = argparse.ArgumentParser(description="Ask questions about your documents using RAG.")
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    parser.add_argument("--quiet", action="store_true", help="Hide retrieved chunks and scores")
    args = parser.parse_args()

    check_ollama()
    embeddings, chunks = load_index()
    show_context = not args.quiet
```

- `nargs="?"` makes the question **optional** — with a question it answers once and
  exits; without one it drops into interactive mode.
- The index is loaded **once**, before the loop. Only the cheap operations
  (embed one question + one chat call) happen per question.

### 10.3 Interactive loop

```43:56:ask.py
    # Interactive mode
    print("RAG question-answering (type 'quit' to exit)")
    print("Example: Which planet is the largest?\n")
    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        answer_question(question, embeddings, chunks, args.top_k, show_context)
```

Small but polite CLI details worth copying in your own projects:

- `Ctrl+C` / `Ctrl+D` (`KeyboardInterrupt` / `EOFError`) exit cleanly instead of
  dumping a stack trace.
- Empty input is ignored (`continue`), and `quit` / `exit` / `q` all work.

---

## 11. End-to-End Trace of One Question

Command: `python ask.py "Which planet is the largest?"`

| # | Step | Function | What happens |
|---|---|---|---|
| 1 | Startup check | `check_ollama()` | Ollama is running, both models present. |
| 2 | Load index | `load_index()` | `28 × 768` matrix + 28 chunk records loaded from `index/`. |
| 3 | Embed question | `embed()` | `"Which planet is the largest?"` → 768-dim vector. |
| 4 | Score all chunks | `cosine_similarity()` ×28 | e.g. Jupiter chunk `0.74`, Saturn chunk `0.68`, photosynthesis chunk `0.31`… |
| 5 | Pick top 3 | `np.argsort(...)[::-1][:3]` | Three `solar_system.txt` chunks win. |
| 6 | Build prompt | `build_prompt()` | Context (3 labeled chunks) + rules + question. |
| 7 | Generate | `generate_answer()` | llama3.2 reads the context and writes the answer. |
| 8 | Output | `answer_question()` | *"Jupiter is the largest planet in our solar system… (Source: solar_system.txt)"* |

And the control case — `python ask.py "What is the capital of France?"`:
retrieval still returns the 3 "least bad" chunks, but scores are low and the context
contains nothing about France, so the prompt's escape hatch fires:
*"I don't have enough information to answer that question."*
**This is RAG succeeding, not failing** — the model refused to hallucinate.

---

## 12. Running the Project

One-time setup is covered in [Prerequisites & Setup](#3-prerequisites--setup). Day-to-day usage:

```bash
# Phase 1 — build the index (repeat after changing data/)
python ingest.py

# Phase 2 — ask questions
python ask.py "Which planet is the largest?"
python ask.py                                   # interactive mode
python ask.py "Tell me about computers" --top-k 5
python ask.py "What is photosynthesis?" --quiet
```

---

## 13. Exercises

1. **Add your own document.** Drop any `.txt` file into `data/`, re-run
   `python ingest.py`, and ask questions about it.
2. **Tune the chunking knobs.** Set `CHUNK_SIZE = 100` in `rag.py`, re-ingest, and
   compare answers. Then try `2000`. What breaks in each direction, and why?
3. **Vary `top_k`.** Compare `--top-k 1` vs `--top-k 5` on the same question. When
   does more context help, and when does it add noise?
4. **Inspect the scores.** Ask one on-topic and one off-topic question and note the
   top scores. Could you pick a threshold below which the system should refuse to
   answer *without* even calling the LLM?
5. **Swap the chat model.** `ollama pull mistral`, set `CHAT_MODEL = "mistral"`,
   and compare answer style and citation quality.
6. **(Challenge) Sentence-aware chunking.** Rewrite `chunk_text()` to split on
   sentence boundaries (e.g. after `. `) instead of raw character counts. Does
   retrieval improve?
7. **(Challenge) Break the grounding.** Remove the "ONLY the context" rule from
   `build_prompt()` and ask about France again. Watch the model answer from its
   training data — you have just witnessed why grounding matters.

---

## 14. Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation: retrieve relevant text, then generate an answer grounded in it. |
| **LLM** | Large Language Model — the neural network that generates text (here: llama3.2). |
| **Embedding** | A fixed-length vector of numbers representing the *meaning* of a text. |
| **Vector** | An ordered list of numbers; here 768 floats per text. |
| **Cosine similarity** | Angle-based similarity between two vectors; 1.0 = same direction/meaning. |
| **Chunk** | A small piece of a document (here ≤500 characters) that gets its own embedding. |
| **Overlap** | Characters repeated between consecutive chunks so boundary sentences survive. |
| **Index** | The stored embeddings + chunk metadata that make search possible. |
| **Ingestion** | The offline phase: load → chunk → embed → store. |
| **Retrieval** | Finding the chunks most similar to the question. |
| **top-k** | How many best-matching chunks to hand to the LLM. |
| **Grounding** | Restricting the LLM to answer only from supplied context. |
| **Hallucination** | An LLM confidently stating something false or unsupported. |
| **Ollama** | A tool that runs open-source LLMs locally and exposes them via an API. |

---

## 15. FAQ / Troubleshooting

| Symptom | Cause & Fix |
|---|---|
| `ERROR: Cannot connect to Ollama` | Ollama isn't running → start the Ollama app or run `ollama serve`. |
| `ERROR: Missing Ollama models` | Run the printed `ollama pull ...` commands. |
| `ERROR: No index found` | You ran `ask.py` before `ingest.py` → run `python ingest.py` first. |
| First answer is very slow | The model is loading into RAM on first use; later calls are fast. |
| Answers ignore my new document | You forgot to re-run `python ingest.py` after editing `data/`. |
| Answer says "I don't have enough information" | Either the answer truly isn't in your documents (correct behavior!) or retrieval missed it — check the printed scores, try a higher `--top-k`, or rephrase the question. |
| Out of memory | Use a smaller chat model, e.g. `ollama pull phi3` and set `CHAT_MODEL = "phi3"`. |
