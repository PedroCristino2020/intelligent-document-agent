import json
import math
import os
from pathlib import Path

from src.embeddings import embed_query


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def retrieve(
    query: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    top_k: int = 3,
) -> list[dict]:
    if not chunks or not embeddings:
        return []

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must have the same length")

    query_embedding = embed_query(query)
    similarities = [
        cosine_similarity(query_embedding, chunk_embedding)
        for chunk_embedding in embeddings
    ]

    indexed_similarities = list(enumerate(similarities))
    indexed_similarities.sort(key=lambda item: item[1], reverse=True)

    results = []
    for idx, score in indexed_similarities[:top_k]:
        chunk = chunks[idx]
        results.append(
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk.get("chunk_index"),
                "text": chunk["text"],
                "similarity": float(score),
            }
        )

    return results