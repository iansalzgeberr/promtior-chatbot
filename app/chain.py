"""
chain.py - Definicion de la cadena RAG (Retrieval Augmented Generation)

Flujo:
pregunta + historial
    -> retriever busca los chunks mas similares en ChromaDB (MMR)
    -> chunks + pregunta + historial se insertan en el prompt
    -> LLM genera respuesta basada UNICAMENTE en el contexto real
    -> respuesta devuelta al usuario
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

from app.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    RETRIEVER_K,
)
from app.ingest import run_ingestion


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def format_chat_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    return "\n".join(history[-8:])


def get_question(x):
    return x["question"] if isinstance(x, dict) else x.question


def get_history(x):
    history = x.get("chat_history", []) if isinstance(x, dict) else (x.chat_history or [])
    return format_chat_history(history)


def create_rag_chain():
    # 1. Cargar vector store
    vectorstore = run_ingestion()

    # 2. Retriever por similitud coseno
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    # 3. LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    # 4. Prompt de respuesta
    prompt = ChatPromptTemplate.from_template("""You are a virtual assistant specialized in answering questions about Promtior,
a technology and organizational consulting company specialized in generative AI.

IMPORTANT RULES:
1. Answer ONLY based on the provided context and/or conversation history.
2. If the user asks to translate, rephrase or repeat something from the conversation history, use the history to fulfill that request — do NOT say you lack information.
3. If the answer is not in the context, say honestly that you don't have that information.
4. Always respond in the same language the user used in their latest message.
5. Do NOT confuse document authorship with company founders. "Prepared by" in a document means who wrote that document, not who founded the company. If the only evidence for founders is a "Prepared by" field, say you don't have that information.

Context from knowledge base:
{context}

Conversation history:
{chat_history}

Question: {question}

Answer:""")

    # 5. Cadena RAG
    rag_chain = (
        RunnableParallel(
            context=RunnableLambda(get_question) | retriever | format_docs,
            question=RunnableLambda(get_question),
            chat_history=RunnableLambda(get_history),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = create_rag_chain()
