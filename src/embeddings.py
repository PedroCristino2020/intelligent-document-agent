
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

load_dotenv()


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Configure it in your .env file."
        )

    return genai.Client(api_key=api_key)


def _embed_with_retry(
    client: genai.Client,
    model: str,
    contents: list[str],
    task_type: str,
    max_retries: int = 5,
) -> list[list[float]]:
    for attempt in range(max_retries + 1):
        try:
            response = client.models.embed_content(
                model=model,
                contents=contents,
                config=types.EmbedContentConfig(task_type=task_type),
            )

            if not response.embeddings:
                raise RuntimeError("Gemini returned no embeddings")

            return [embedding.values for embedding in response.embeddings]

        except ClientError as exc:
            if exc.code != 429 or attempt >= max_retries:
                raise

            # Use the retry delay supplied by Gemini when available.
            match = re.search(
                r"retry in\s*(\d+(?:\.\d+)?)s",
                str(exc),
                re.IGNORECASE,
            )

            wait_seconds = (
                float(match.group(1)) + 2 if match else 40
            )

            print(
                f"Gemini quota reached. Waiting "
                f"{wait_seconds:.1f} seconds before retry "
                f"({attempt + 1}/{max_retries})..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Embedding retries exhausted")


def embed_texts(
    texts: list[str],
    batch_size: int = 16,
) -> list[list[float]]:
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
            batch = texts[start:start + batch_size]

            batch_embeddings = _embed_with_retry(
                client=client,
                model=model,
                contents=batch,
                task_type="RETRIEVAL_DOCUMENT",
            )

            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    "Gemini returned an unexpected number of embeddings"
                )

            embeddings.extend(batch_embeddings)

    return embeddings


def embed_query(query: str) -> list[float]: #Convierte una pregunta del usuario en un unico vector numerico
    if not query.strip():
        raise ValueError("Query must not be empty")

    model = os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "gemini-embedding-001",
    )

    with _get_client() as client:
        embeddings = _embed_with_retry(
            client=client,
            model=model,
            contents=[query],
            task_type="RETRIEVAL_QUERY",
        )

    return embeddings[0]
