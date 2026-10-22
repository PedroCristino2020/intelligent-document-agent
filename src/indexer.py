
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from src.chunker import split_into_chunks
from src.document_loader import load_pdf
from src.embeddings import embed_texts


DEFAULT_DOCUMENTS_DIR = Path("documents")
DEFAULT_DATABASE_PATH = Path("data") / "index.db"


def file_hash(path: Path) -> str:
    """Calculate a SHA-256 hash to detect changed PDFs."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def initialise_database(connection: sqlite3.Connection) -> None:
    """Create the local table used to persist chunks and embeddings."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS indexed_chunks (
            source TEXT NOT NULL,
            page INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            PRIMARY KEY (source, page, chunk_index)
        )
        """
    )
    connection.commit()


def index_pdf(
    pdf_path: Path,
    connection: sqlite3.Connection,
    batch_size: int = 16,
    max_pages: int | None = None,
) -> int:
    """Index one PDF, saving batches so completed work can be reused."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    current_hash = file_hash(pdf_path)

    existing_hashes = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT file_hash FROM indexed_chunks WHERE source = ?",
            (pdf_path.name,),
        )
    }

    if existing_hashes and existing_hashes != {current_hash}:
        print(f"El archivo ha cambiado; reindexando: {pdf_path.name}")
        connection.execute(
            "DELETE FROM indexed_chunks WHERE source = ?",
            (pdf_path.name,),
        )
        connection.commit()

    page_docs = load_pdf(pdf_path, max_pages=max_pages)

    pending_chunks: list[dict] = []
    new_chunks = 0

    def save_batch() -> None:
        nonlocal new_chunks

        if not pending_chunks:
            return

        vectors = embed_texts(
            [chunk["text"] for chunk in pending_chunks],
            batch_size=batch_size,
        )

        rows = [
            (
                chunk["source"],
                chunk["page"],
                chunk["chunk_index"],
                chunk["text"],
                json.dumps(vector),
                current_hash,
            )
            for chunk, vector in zip(pending_chunks, vectors)
        ]

        connection.executemany(
            """
            INSERT OR REPLACE INTO indexed_chunks
            (source, page, chunk_index, text, embedding, file_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()

        new_chunks += len(rows)
        print(
            f"  Guardados {new_chunks} fragmentos nuevos de "
            f"{pdf_path.name}",
            flush=True,
        )
        pending_chunks.clear()

    for page_doc in page_docs:
        page_chunks = split_into_chunks([page_doc])

        for chunk in page_chunks:
            exists = connection.execute(
                """
                SELECT 1 FROM indexed_chunks
                WHERE source = ? AND page = ? AND chunk_index = ?
                """,
                (
                    chunk["source"],
                    chunk["page"],
                    chunk["chunk_index"],
                ),
            ).fetchone()

            if exists:
                continue

            pending_chunks.append(chunk)

            if len(pending_chunks) >= batch_size:
                save_batch()

    save_batch()
    return new_chunks


def index_directory(
    documents_dir: Path = DEFAULT_DOCUMENTS_DIR,
    database_path: Path = DEFAULT_DATABASE_PATH,
    batch_size: int = 16,
    max_pages: int | None = None,
    only_pdf: str | None = None,
) -> None:
    """Index PDFs in a directory and persist progress in SQLite."""
    if not documents_dir.is_dir():
        raise NotADirectoryError(
            f"Directory not found: {documents_dir}"
        )

    pdfs = sorted(documents_dir.glob("*.pdf"))

    if only_pdf:
        pdfs = [path for path in pdfs if path.name == only_pdf]

        if not pdfs:
            raise FileNotFoundError(
                f"PDF not found in {documents_dir}: {only_pdf}"
            )

    if not pdfs:
        print(f"No PDF files found in {documents_dir}")
        return

    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        initialise_database(connection)

        for pdf_path in pdfs:
            print(f"\nProcesando {pdf_path.name}...", flush=True)

            count = index_pdf(
                pdf_path,
                connection,
                batch_size=batch_size,
                max_pages=max_pages,
            )

            print(
                f"Finalizado: {pdf_path.name}; "
                f"fragmentos nuevos: {count}",
                flush=True,
            )

        total = connection.execute(
            "SELECT COUNT(*) FROM indexed_chunks"
        ).fetchone()[0]

        print(f"\nTotal de fragmentos almacenados: {total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a resumable local RAG index from PDF files."
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to process per PDF (for testing).",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Process only this PDF filename.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of text chunks per embedding batch.",
    )
    args = parser.parse_args()

    if args.max_pages is not None and args.max_pages <= 0:
        parser.error("--max-pages must be positive")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    index_directory(
        batch_size=args.batch_size,
        max_pages=args.max_pages,
        only_pdf=args.pdf,
    )


if __name__ == "__main__":
    main()
