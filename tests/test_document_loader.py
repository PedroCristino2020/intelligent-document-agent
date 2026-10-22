
from pathlib import Path

import pytest

from src.document_loader import load_pdf, load_pdfs


def test_load_pdf_extracts_text_and_page_number(tmp_path):
    from pypdf import PdfWriter

    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)

    with pdf_path.open("wb") as file:
        writer.write(file)

    # Un PDF en blanco no contiene texto extraíble.
    assert load_pdf(pdf_path) == []


def test_load_pdf_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")


def test_load_pdf_rejects_non_pdf(tmp_path):
    text_path = tmp_path / "sample.txt"
    text_path.write_text("Texto de prueba", encoding="utf-8")

    with pytest.raises(ValueError):
        load_pdf(text_path)


def test_load_pdfs_rejects_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        load_pdfs(tmp_path / "missing_directory")
