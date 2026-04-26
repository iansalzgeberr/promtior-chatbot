"""
config.py - Configuracion centralizada del proyecto

Aqui cargamos las variables de entorno y definimos constantes globales.
python-dotenv lee el archivo .env automaticamente.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API key - se lee del archivo .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "No se encontro OPENAI_API_KEY. "
        "Crea un archivo .env con: OPENAI_API_KEY=tu-key-aqui"
    )

# Modelo de LLM a usar
# gpt-4o-mini: excelente balance costo/calidad (recomendado)
# gpt-3.5-turbo: mas economico
# gpt-4o: mas potente, mas caro
LLM_MODEL = "gpt-4o-mini"

# Modelo de embeddings de OpenAI
EMBEDDING_MODEL = "text-embedding-3-small"

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore")
PDF_PATH = os.path.join(BASE_DIR, "data", "promtior_info.pdf")

# URLs del sitio de Promtior para scraping
PROMTIOR_URLS = [
    "https://www.promtior.ai/",
    "https://www.promtior.ai/service",
    "https://www.promtior.ai/use-cases",
]

# Parametros del chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Cantidad de chunks a recuperar por query
RETRIEVER_K = 8
