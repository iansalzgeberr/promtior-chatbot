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


def _cluster_words_by_column(words: list, tolerance: int = 80) -> list:
    """
    Agrupa palabras en columnas por proximidad de x0.

    Compara cada palabra contra el x0 más cercano dentro de cada columna
    existente (no contra el primero). Esto permite manejar nombres como
    "Forestal Atlántico Sur" donde las palabras no están perfectamente alineadas.
    """
    columns = []
    for word in sorted(words, key=lambda w: w["x0"]):
        best_col, best_dist = None, float("inf")
        for col in columns:
            dist = min(abs(word["x0"] - w["x0"]) for w in col)
            if dist < tolerance and dist < best_dist:
                best_dist, best_col = dist, col
        if best_col is not None:
            best_col.append(word)
        else:
            columns.append([word])
    return columns


def _extract_client_grid(page) -> str:
    """
    Reconstruye la lista de clientes desde la página con grilla de logos.

    El PDF tiene logos en columnas múltiples. Usando coordenadas x/y de
    cada palabra, agrupamos por fila (saltos verticales > 100px) y luego
    por columna (proximidad x0 con tolerance=80px) para reconstruir cada
    nombre correctamente sin ningún fix hardcodeado.
    """
    words = page.extract_words(x_tolerance=5, y_tolerance=3)

    intro_words = [w for w in words if w["top"] < 400]
    grid_words  = [w for w in words if w["top"] >= 400]

    intro_text = " ".join(
        w["text"] for w in sorted(intro_words, key=lambda w: (w["top"], w["x0"]))
    )

    if not grid_words:
        return intro_text

    # Agrupar en filas por saltos verticales > 100px
    sorted_words = sorted(grid_words, key=lambda w: w["top"])
    rows, current_row = [], [sorted_words[0]]
    for word in sorted_words[1:]:
        if word["top"] - current_row[-1]["top"] > 100:
            rows.append(current_row)
            current_row = [word]
        else:
            current_row.append(word)
    rows.append(current_row)

    # Reconstruir nombres: una columna = un cliente
    client_names = []
    for row in rows:
        columns = _cluster_words_by_column(row, tolerance=80)
        for col in sorted(columns, key=lambda c: min(w["x0"] for w in c)):
            name = " ".join(
                w["text"] for w in sorted(col, key=lambda w: (w["top"], w["x0"]))
            )
            client_names.append(name)

    clients_text = "Promtior clients:\n" + "\n".join(f"- {n}" for n in client_names)
    return f"{intro_text}\n\n{clients_text}"


def load_pdf(pdf_path: str) -> List[Document]:
    """
    Carga el PDF de la presentacion de Promtior usando pdfplumber.

    pdfplumber extrae texto con conciencia posicional (coordenadas x/y),
    lo que permite reconstruir correctamente los nombres de clientes desde
    la pagina con grilla de logos — sin fixes hardcodeados.

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
        import pdfplumber

        logger.info(f"Cargando PDF: {pdf_path}")
        documents = []

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""

                # La pagina de clientes tiene grilla de logos — extraer con layout
                if "Bionic" in text and "Organizations" in text:
                    text = _extract_client_grid(page)

                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": "promtior_presentation.pdf",
                            "type": "pdf",
                            "page": i + 1,
                        },
                    ))

        logger.info(f"  -> {len(documents)} paginas cargadas del PDF")
        return documents

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
