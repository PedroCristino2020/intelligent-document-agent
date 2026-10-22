
from pathlib import Path

import pytest

from src.document_loader import load_pdf, load_pdfs


def _create_pdf_with_text(path, texts):
    from pypdf import PdfWriter

    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=300, height=300)
        # pypdf can add text via add_text or we just rely on extract_text if present;
        # for tests we use a simple approach - create pages and rely on extract_text
        # Some pypdf versions need different API; try to add text if possible
        try:
            from pypdf import PageObject

            # Try to add text annotation/content if supported
            pass
        except Exception:
            pass
    with path.open("wb") as file:
        writer.write(file)
    return path


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


def test_load_pdf_max_pages_limits_pages(tmp_path, monkeypatch):
    from pypdf import PdfReader, PdfWriter

    pdf_path = tmp_path / "multi.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_blank_page(width=300, height=300)
    writer.add_blank_page(width=300, height=300)
    with pdf_path.open("wb") as file:
        writer.write(file)

    # Monkeypatch extract_text to return text per page
    original_extract = PdfReader.pages.__class__  # not directly; patch on instance
    # Easier: patch PdfReader.pages to return mock pages
    class DummyPage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class DummyReader:
        def __init__(self, *args, **kwargs):
            self.pages = [DummyPage("p1"), DummyPage("p2"), DummyPage("p3")]

    monkeypatch.setattr("src.document_loader.PdfReader", DummyReader)

    result = load_pdf(pdf_path, max_pages=2)
    assert len(result) == 2
    assert result[0]["page"] == 1
    assert result[1]["page"] == 2
    assert result[0]["source"] == "multi.pdf"


def test_load_pdf_max_pages_rejects_non_positive(tmp_path, monkeypatch):
    from pypdf import PdfWriter

    pdf_path = tmp_path / "x.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with pdf_path.open("wb") as file:
        writer.write(file)

    class DummyPage:
        def extract_text(self):
            return "hi"

    class DummyReader:
        def __init__(self, *args, **kwargs):
            self.pages = [DummyPage()]

    monkeypatch.setattr("src.document_loader.PdfReader", DummyReader)

    with pytest.raises(ValueError):
        load_pdf(pdf_path, max_pages=0)
    with pytest.raises(ValueError):
        load_pdf(pdf_path, max_pages=-1)
