import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import (
    AskRequest,
    AskResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.ask import DEFAULT_DATABASE_PATH, ask_question
from src.retriever import retrieve

router = APIRouter(tags=["Pregunta-Respuesta"])

def _load_index():
    db_path = DEFAULT_DATABASE_PATH
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Index not found: {db_path}. Run the indexer first."
        )

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT source, page, chunk_index, text, embedding
            FROM indexed_chunks
            """
        ).fetchall()

    chunks = []
    embeddings = []

    for source, page, chunk_index, text, embedding_json in rows:
        chunks.append(
            {
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
                "text": text,
            }
        )
        embeddings.append(json.loads(embedding_json))

    if not chunks:
        raise HTTPException(status_code=404, detail="El índice está vacío.")

    return chunks, embeddings


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        answer, sources = ask_question(req.question, top_k=req.top_k)
        return {"answer": answer, "sources": sources}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar respuesta: {e}")


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        chunks, embeddings = _load_index()
        results = retrieve(req.question, chunks, embeddings, top_k=req.top_k)

        out: list[SearchResult] = []
        for i, r in enumerate(results, start=1):
            out.append(
                SearchResult(
                    rank=i,
                    source=r["source"],
                    page=r["page"],
                    chunk_index=r.get("chunk_index"),
                    similarity=r["similarity"],
                    text=r["text"],
                )
            )

        return SearchResponse(results=out)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {e}")