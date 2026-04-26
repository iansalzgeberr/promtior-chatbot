"""
streamlit_app.py - Frontend del chatbot Promtior

Streamlit es un framework de Python para crear UIs web rapidamente.
No necesitas conocer HTML/CSS/JavaScript - todo se hace en Python.

Esta app:
- Muestra una interfaz de chat con historial
- Envia las preguntas al backend LangServe via HTTP
- Muestra las respuestas del chatbot
- Permite limpiar el historial

Como ejecutar:
    streamlit run frontend/streamlit_app.py

Nota: El backend (server.py) debe estar corriendo en localhost:8000
"""
from pathlib import Path
import httpx
import streamlit as st
import os

LANGSERVE_URL = os.getenv("LANGSERVE_URL", "http://backend:8000/chat/invoke")
FAV_PATH = Path(__file__).parent / "fav-icone.png"
CHAT_AVATAR_PATH = Path(__file__).parent / "fav-icone.png"

# Configuracion de la pagina
st.set_page_config(
    page_title="Promtior AI Assistant",
    page_icon=str(FAV_PATH),
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Estilos CSS personalizados para que se vea profesional
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #4CAF93;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #1e3a5f;
        border-left: 4px solid #4CAF93;
    }
    .assistant-message {
        background-color: #1a1a2e;
        border-left: 4px solid #7B68EE;
    }
    .source-tag {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1> Promtior AI Assistant</h1>
    <p>Preguntame sobre Promtior - servicios, historia, clientes y mas</p>
</div>
""", unsafe_allow_html=True)

# Inicializar historial de chat en session_state
# session_state persiste entre reruns de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Mensaje de bienvenida
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hola! Soy el asistente virtual de Promtior. "
            "Puedo responder preguntas sobre los servicios, historia y clientes de Promtior. "
            "¿En que puedo ayudarte?"
        ),
    })


def ask_chatbot(question: str, chat_history: list) -> str:
    """
    Envia una pregunta al backend LangServe junto con el historial de chat.

    El historial permite manejar preguntas de seguimiento como
    "decimelo en ingles" o "explica mas sobre eso".
    """
    # Construir historial como lista de strings alternando user/assistant
    # Formato: ["User: pregunta", "Assistant: respuesta", ...]
    history_strings = []
    for msg in chat_history[-8:]:  # Ultimos 8 mensajes
        if msg["content"].startswith("Hola!"):  # Saltar el mensaje de bienvenida
            continue
        prefix = "User" if msg["role"] == "user" else "Assistant"
        history_strings.append(f"{prefix}: {msg['content']}")

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                LANGSERVE_URL,
                json={"input": {"question": question, "chat_history": history_strings}},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("output", "No se pudo obtener respuesta.")

    except httpx.ConnectError:
        return (
            "Error: No se puede conectar al servidor. "
            "Asegurate de que el backend este corriendo en localhost:8000"
        )
    except httpx.TimeoutException:
        return "Error: El servidor tardo demasiado en responder. Intenta de nuevo."
    except Exception as e:
        return f"Error inesperado: {str(e)}"


# Mostrar historial de mensajes
for message in st.session_state.messages:
    avatar = str(CHAT_AVATAR_PATH) if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

# Input del usuario (siempre al final, como en ChatGPT)
if prompt := st.chat_input("Escribe tu pregunta sobre Promtior..."):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.write(prompt)

    # Obtener respuesta del chatbot
    with st.chat_message("assistant", avatar=str(CHAT_AVATAR_PATH)):
        with st.spinner("Buscando informacion..."):
            response = ask_chatbot(prompt, st.session_state.messages)
        st.write(response)

    # Agregar respuesta al historial
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar con info y boton para limpiar historial
with st.sidebar:
    st.markdown("### Sobre este chatbot")
    st.markdown("""
    Este chatbot usa **RAG (Retrieval Augmented Generation)**:

    1. Tu pregunta se convierte en un vector
    2. Se buscan los fragmentos mas relevantes en la base de conocimiento
    3. Esos fragmentos + tu pregunta se envian al LLM
    4. El LLM genera una respuesta basada en el contexto real

    **Fuentes de datos:**
    - Sitio web de Promtior
    - Presentacion institucional (PDF)

    **Stack tecnologico:**
    - LangChain + LangServe
    - OpenAI GPT-4o-mini
    - ChromaDB Vector Store
    - Streamlit
    """)

    st.divider()

    st.markdown("### Preguntas sugeridas")
    suggested_questions = [
        "What services does Promtior offer?",
        "When was the company founded?",
        "Who are Promtior's clients?",
        "What is Promtior's approach to AI?",
    ]

    for question in suggested_questions:
        if st.button(question, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            response = ask_chatbot(question, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    st.divider()

    if st.button("Limpiar chat", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Chat limpiado. Como puedo ayudarte?",
        }]
        st.rerun()
