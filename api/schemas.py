from pydantic import BaseModel, Field


class Source(BaseModel):
    source: str
    page: int
    similarity: float
    chunk_index: int | None = None
    text: str = Field(..., description="Contenido completo del chunk recuperado")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Pregunta a realizar sobre los documentos indexados")
    top_k: int = Field(default=3, ge=1, le=10, description="Número de chunks a recuperar")


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Consulta para buscar chunks relevantes")
    top_k: int = Field(default=3, ge=1, le=10, description="Número de chunks a recuperar")

class SearchResult(BaseModel):
    rank: int
    source: str
    page: int
    chunk_index: int | None = None
    similarity: float
    text: str = Field(..., description="Contenido completo del chunk recuperado")

class SearchResponse(BaseModel):
    results: list[SearchResult]
