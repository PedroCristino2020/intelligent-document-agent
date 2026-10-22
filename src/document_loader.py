from pathlib import Path

from pypdf import PdfReader


def load_pdf(file_path: str | Path, max_pages: int | None = None) -> list[dict]:
    """Extract text from a PDF, preserving its source and page number."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    reader = PdfReader(str(path))
    documents = []

    pages = reader.pages
    if max_pages is not None:
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        page_limit = min(len(pages), max_pages)
        pages_to_iterate = pages[:page_limit]
    else:
        pages_to_iterate = pages

    for page_number, page in enumerate(pages_to_iterate, start=1):
        text = (page.extract_text() or "").strip()

        if not text:
            continue

        documents.append(
            {
                "source": path.name,
                "page": page_number,
                "text": text,
            }
        )

    return documents


def load_pdfs(directory: str | Path) -> list[dict]:
    """Load all PDF files in a directory."""
    path = Path(directory)

    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {path}")

    documents = []

    for pdf_path in sorted(path.glob("*.pdf")):
        documents.extend(load_pdf(pdf_path))

    return documents
