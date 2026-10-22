
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


def _get_client() -> genai.Client:
    """Create a Gemini client using the API key in .env."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Configure it in your .env file."
        )

    return genai.Client(api_key=api_key)


def embed_texts(
    texts: list[str],
    batch_size: int = 16,
) -> list[list[float]]:
    """Generate document embeddings in batches, preserving input order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Texts must not be empty")

    model = os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "gemini-embedding-001",
    )
    embeddings = []

    with _get_client() as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            response = client.models.embed_content(
                model=model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )

            if not response.embeddings:
                raise RuntimeError("Gemini returned no embeddings")

            if len(response.embeddings) != len(batch):
                raise RuntimeError(
                    "Gemini returned an unexpected number of embeddings"
                )

            embeddings.extend(
                embedding.values for embedding in response.embeddings
            )

    return embeddings


def embed_query(query: str) -> list[float]:
    """Generate an embedding specifically for a search query."""
    if not query.strip():
        raise ValueError("Query must not be empty")

    model = os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "gemini-embedding-001",
    )

    with _get_client() as client:
        response = client.models.embed_content(
            model=model,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            ),
        )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no query embedding")

    return response.embeddings[0].values
