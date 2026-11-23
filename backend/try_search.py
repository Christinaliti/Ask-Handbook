"""Try the search in the terminal.

    python try_search.py                      # type questions, empty line to quit
    python try_search.py "HX-400 oil volume"  # search once and exit
"""

import sys

from search import search


def show(question):
    results = search(question)
    if not results:
        print("No matching chunks.")
        return
    for rank, hit in enumerate(results, start=1):
        preview = " ".join(hit["text"].split())[:200]
        cosine = "-" if hit["similarity"] is None else f"{hit['similarity']:.3f}"
        print(f"\n{rank}. {hit['document']} | page {hit['page']} | rrf {hit['score']:.4f} | "
              f"bm25 {hit['bm25_score']:.2f} | cosine {cosine} | {hit['chunk_id']}")
        print(f"   {preview}")


def main():
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return
    print("Search the handbook. Type a question, or press Enter on an empty line to quit.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        show(question)


if __name__ == "__main__":
    main()
