
import argparse
import json
import sqlite3
from pathlib import Path

from src.llm import answer_question
from src.retriever import retrieve


DEFAULT_DATABASE_PATH = Path("data") / "index.db"


def ask_question(
    question: str,
    database_path: Path = DEFAULT_DATABASE_PATH,
    top_k: int = 5,
) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks and generate a grounded answer."""
    if not database_path.is_file():
        raise FileNotFoundError(
            f"Index not found: {database_path}. Run the indexer first."
        )

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT source, page, chunk_index, text, embedding
            FROM indexed_chunks
            """
        ).fetchall()

    chunks = []
    embeddings = []

    for source, page, chunk_index, text, embedding_json in rows:
        chunks.append({
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "text": text,
        })
        embeddings.append(json.loads(embedding_json))

    if not chunks:
        return (
            "El índice está vacío. Ejecuta primero el indexador.",
            [],
        )

    results = retrieve(question, chunks, embeddings, top_k=top_k)
    answer = answer_question(question, results)

    return answer, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question about your indexed PDF documents."
    )
    parser.add_argument(
        "question",
        help="Question to ask about the documents.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to retrieve.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    answer, sources = ask_question(args.question, top_k=args.top_k)

    print("\nRESPUESTA\n")
    print(answer)

    if sources:
        print("\nFRAGMENTOS RECUPERADOS\n")
        for source in sources:
            print(
                f"- {source['source']}, página {source['page']} "
                f"(similitud: {source['similarity']:.3f})"
            )


if __name__ == "__main__":
    main()
