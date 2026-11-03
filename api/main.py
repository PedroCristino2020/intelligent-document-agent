from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import qa

app = FastAPI(
    title="Intelligent Document Agent API",
    description="API para consultar documentos PDF indexados con RAG (Gemini + Ollama opcional)",
    version="1.0.0",
)

# Útil si luego creas un frontend (Vue/React) desde otro puerto
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción mejor restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa.router)


@app.get("/health")
def health():
    return {"status": "ok"}