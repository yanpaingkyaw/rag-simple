"""Step 2 of the RAG pipeline: ask questions about your documents.

Loads the index built by ingest.py, retrieves the most relevant chunks
for your question, and asks the LLM to answer using that context.

Usage:
    python ask.py "Which planet is the largest?"   # single question
    python ask.py                                   # interactive mode
"""

import argparse

from rag import check_ollama, generate_answer, load_index, retrieve


def answer_question(question, embeddings, chunks, top_k, show_context):
    context = retrieve(question, embeddings, chunks, top_k=top_k)

    if show_context:
        print("\nRetrieved chunks:")
        for c in context:
            print(f"  [{c['score']:.4f}] {c['source']}")

    answer = generate_answer(question, context)
    print(f"\nAnswer:\n{answer}\n")


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your documents using RAG.")
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    parser.add_argument("--quiet", action="store_true", help="Hide retrieved chunks and scores")
    args = parser.parse_args()

    check_ollama()
    embeddings, chunks = load_index()
    show_context = not args.quiet

    if args.question:
        answer_question(args.question, embeddings, chunks, args.top_k, show_context)
        return

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


if __name__ == "__main__":
    main()
