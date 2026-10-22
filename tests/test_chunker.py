
import pytest

from src.chunker import split_into_chunks


def test_splits_text_with_overlap():
    documents = [
        {
            "source": "manual.pdf",
            "page": 1,
            "text": "ABCDEFGHIJ",
        }
    ]

    chunks = split_into_chunks(
        documents,
        chunk_size=6,
        overlap=2,
    )

    assert [chunk["text"] for chunk in chunks] == [
        "ABCDEF",
        "EFGHIJ",
    ]
    assert chunks[1]["source"] == "manual.pdf"
    assert chunks[1]["page"] == 1


def test_empty_documents_return_no_chunks():
    assert split_into_chunks(
        [{"source": "empty.pdf", "page": 1, "text": ""}]
    ) == []


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        split_into_chunks([], chunk_size=0)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_into_chunks([], chunk_size=10, overlap=10)
