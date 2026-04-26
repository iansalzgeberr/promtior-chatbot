"""
ingest.py - Ingesta y procesamiento de documentos

Este modulo es responsable de:
1. Cargar documentos desde dos fuentes:
   - El sitio web de Promtior (scraping)
   - El PDF de la presentacion (fuente extra para puntos adicionales)
2. Dividir los documentos en fragmentos (chunks) manejables
3. Convertir esos fragmentos a vectores (embeddings) con OpenAI
4. Guardar el vector store ChromaDB localmente para reutilizarlo

Por que usamos ChromaDB?
- Base de datos vectorial que persiste en disco automaticamente
- Funciona localmente, sin necesidad de infraestructura externa
- Perfecta para proyectos de este tamano
"""
import os
import logging
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    PDF_PATH,
    PROMTIOR_URLS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scrape_website(urls: List[str]) -> List[Document]:
    """
    Hace scraping del sitio web de Promtior y retorna documentos LangChain.

    Usamos requests + BeautifulSoup para extraer el texto limpio de cada pagina.
    Cada pagina se convierte en un Document con metadata que incluye la URL.

    Args:
        urls: Lista de URLs a scrapear

    Returns:
        Lista de Documents con el contenido de cada pagina
    """
    documents = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    for url in urls:
        try:
            logger.info(f"Scrapeando: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Eliminar elementos que no tienen contenido util
            for tag in soup(["script", "style", "nav", "footer", "header", "meta"]):
                tag.decompose()

            # Extraer texto limpio
            text = soup.get_text(separator="\n", strip=True)

            # Eliminar lineas vacias consecutivas
            lines = [line for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            if clean_text:
                documents.append(
                    Document(
                        page_content=clean_text,
                        metadata={"source": url, "type": "website"},
                    )
                )
                logger.info(f"  -> {len(clean_text)} caracteres extraidos")

        except requests.RequestException as e:
            logger.warning(f"Error al scrapear {url}: {e}")

    return documents


def load_pdf(pdf_path: str) -> List[Document]:
    """
    Carga el PDF de la presentacion de Promtior.

    Usamos PyPDFLoader de LangChain que divide automaticamente el PDF
    en paginas, cada una como un Document separado.

    Args:
        pdf_path: Ruta al archivo PDF

    Returns:
        Lista de Documents (uno por pagina del PDF)
    """
    if not os.path.exists(pdf_path):
        logger.warning(f"PDF no encontrado en: {pdf_path}")
        logger.warning("Continuando sin el PDF. Para incluirlo, copialo a data/promtior_info.pdf")
        return []

    try:
        from langchain_community.document_loaders import PyPDFLoader

        logger.info(f"Cargando PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Agregar metadata de tipo para identificar la fuente
        for page in pages:
            page.metadata["type"] = "pdf"
            page.metadata["source"] = "promtior_presentation.pdf"

        logger.info(f"  -> {len(pages)} paginas cargadas del PDF")
        return pages

    except Exception as e:
        logger.error(f"Error al cargar el PDF: {e}")
        return []


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Divide los documentos en fragmentos (chunks) mas pequenos.

    Por que dividimos en chunks?
    - Los modelos de embeddings tienen limite de tokens
    - Chunks mas pequenos = busqueda mas precisa
    - El overlap evita cortar informacion importante entre dos chunks

    Args:
        documents: Lista de Documents completos

    Returns:
        Lista de Documents divididos en chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Primero intenta dividir por parrafos, luego por oraciones, luego por palabras
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documentos divididos: {len(documents)} -> {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Crea el vector store ChromaDB a partir de los chunks.

    Proceso:
    1. OpenAIEmbeddings convierte cada chunk de texto a un vector de numeros
    2. Chroma indexa esos vectores y los persiste en disco automaticamente
    3. En proximas ejecuciones, se carga desde disco sin recalcular

    Args:
        chunks: Lista de Documents en formato chunk

    Returns:
        Vector store Chroma listo para busquedas
    """
    logger.info("Generando embeddings con OpenAI...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    logger.info("Creando vector store ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH,
        collection_name="promtior",
    )
    logger.info(f"Vector store guardado en: {VECTORSTORE_PATH}")
    return vectorstore


def run_ingestion() -> Chroma:
    """
    Funcion principal que orquesta todo el proceso de ingesta.

    Si ya existe un vector store guardado, lo carga directamente.
    Si no, ejecuta el proceso completo de ingesta.

    Returns:
        Vector store Chroma listo para usar
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    # Si ya existe el vector store, cargarlo directamente
    chroma_db_file = os.path.join(VECTORSTORE_PATH, "chroma.sqlite3")
    if os.path.exists(chroma_db_file):
        logger.info("Cargando vector store existente...")
        return Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embeddings,
            collection_name="promtior",
        )

    logger.info("=== Iniciando proceso de ingesta ===")

    # 1. Cargar documentos de todas las fuentes
    all_documents = []

    web_docs = scrape_website(PROMTIOR_URLS)
    all_documents.extend(web_docs)
    logger.info(f"Documentos web cargados: {len(web_docs)}")

    pdf_docs = load_pdf(PDF_PATH)
    all_documents.extend(pdf_docs)
    logger.info(f"Documentos PDF cargados: {len(pdf_docs)}")


    if not all_documents:
        raise RuntimeError("No se pudo cargar ningun documento. Verifica la conexion a internet.")

    logger.info(f"Total documentos: {len(all_documents)}")

    # 2. Dividir en chunks
    chunks = split_documents(all_documents)

    # 3. Crear y guardar el vector store
    vectorstore = create_vectorstore(chunks)

    logger.info("=== Ingesta completada exitosamente ===")
    return vectorstore


if __name__ == "__main__":
    # Ejecutar ingesta directamente: python -m app.ingest
    run_ingestion()
    print("Ingesta completada. Vector store guardado en ./vectorstore/")
