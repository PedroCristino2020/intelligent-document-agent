from pathlib import Path

from pypdf import PdfReader


def load_pdf(file_path: str | Path) -> list[dict]:
    """Extract text from a PDF, preserving its source and page number."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    reader = PdfReader(str(path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
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
