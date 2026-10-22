
import numpy as np

from src.embeddings import embed_query


def retrieve(
    query: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    top_k: int = 5,
) -> list[dict]:
    """Return the most relevant chunks using cosine similarity."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have one embedding")

    if not chunks:
        return []

    query_vector = np.asarray(embed_query(query), dtype=float)
    document_vectors = np.asarray(embeddings, dtype=float)

    if document_vectors.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array")

    if document_vectors.shape[1] != query_vector.shape[0]:
        raise ValueError("Query and document embeddings have different sizes")

    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)

    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("Embeddings must not be zero vectors")

    similarities = (
        document_vectors @ query_vector
    ) / (document_norms * query_norm)

    best_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in best_indices:
        result = dict(chunks[int(index)])
        result["similarity"] = float(similarities[index])
        results.append(result)

    return results
