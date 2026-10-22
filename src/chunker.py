
def split_into_chunks(documents: list[dict], chunk_size: int = 1000, overlap: int = 150,) -> list[dict]:
    """Split document text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks = []

    for document in documents:
        text = document["text"].strip()

        if not text:
            continue

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        **document,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                    }
                )
                chunk_index += 1

            if end == len(text):
                break

            start = end - overlap

    return chunks
