"""
server.py - Servidor LangServe (API REST)

LangServe es una extension de LangChain que convierte cualquier cadena
en una API REST con FastAPI automaticamente.

Al agregar nuestra cadena con add_routes(), LangServe crea:
- POST /chat/invoke    -> envia una pregunta, recibe respuesta completa
- POST /chat/stream    -> envia una pregunta, recibe respuesta en streaming
- POST /chat/batch     -> envia multiples preguntas a la vez
- GET  /chat/playground -> UI visual interactiva para probar (gratis!)

Como ejecutar localmente:
    python -m app.server
    o
    uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload

Luego abrir: http://localhost:8000/chat/playground
"""
import os
import sys

# Fix Unicode encoding on Windows terminals (cp1252 doesn't support box-drawing chars used by LangServe)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from app.chain import rag_chain


class ChatInput(BaseModel):
    question: str
    chat_history: List[str] = []

# Crear la aplicacion FastAPI
app = FastAPI(
    title="Promtior RAG Chatbot",
    description=(
        "Chatbot con arquitectura RAG que responde preguntas sobre Promtior. "
        "Utiliza LangChain + LangServe + OpenAI + FAISS."
    ),
    version="1.0.0",
)

# Configurar CORS para permitir llamadas desde el frontend Streamlit
# En produccion, reemplazar "*" por el dominio real del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def root():
    """Endpoint raiz con informacion basica de la API."""
    return {
        "message": "Promtior RAG Chatbot API",
        "docs": "/docs",
        "playground": "/chat/playground",
        "health": "/health",
    }


@app.get("/health")
async def health():
    """Health check endpoint - util para AWS y Docker."""
    return {"status": "ok"}


# Agregar la cadena RAG como endpoint /chat
# LangServe hace todo el trabajo de crear los endpoints automaticamente
add_routes(
    app,
    rag_chain,
    path="/chat",
    input_type=ChatInput,
    output_type=str,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
