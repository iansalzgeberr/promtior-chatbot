# Project Overview - Promtior RAG Chatbot

## How I Approached the Challenge

When I read the technical test, the core challenge was clear: build a chatbot that can answer questions about Promtior by *retrieving* real information rather than relying on what a language model might have memorized. This is exactly the problem RAG (Retrieval Augmented Generation) solves.

My approach was to break the problem into two distinct phases:

1. **Ingestion phase** (runs once): Collect all available information about Promtior, process it, and store it in a way that enables fast semantic search.
2. **Query phase** (runs on every user question): Find the most relevant pieces of information and use them as context for the LLM to generate a grounded answer.

## Implementation Logic

### Data Sources
I used two sources of information about Promtior:
- **Website scraping**: Pages from promtior.ai scraped with `requests` and `BeautifulSoup`. This captures the most up-to-date public information about services, use cases, and the company.
- **PDF presentation**: The provided slide deck, loaded with `PyPDFLoader`. This contains structured information like founding date (May 2023), client list, and company philosophy that may not be as explicit on the website.

### The RAG Pipeline

```
User Question
     |
     v
[OpenAI Embeddings]       <- Convert question to vector
     |
     v
[ChromaDB Vector Store]   <- Find 8 most similar chunks
     |
     v
[Prompt Template]         <- Inject context + history + question
     |
     v
[OpenAI GPT-4o-mini]     <- Generate grounded answer
     |
     v
Response to User
```

### Text Processing
Documents are split into chunks of 1000 characters with 200-character overlap using `RecursiveCharacterTextSplitter`. The overlap ensures that information spanning a chunk boundary is not lost. Chunks are converted to vectors using OpenAI's `text-embedding-3-small` model and stored in a local ChromaDB index (persisted to disk).

### The Chain (LCEL)
The RAG chain is built with LangChain Expression Language (LCEL). It uses `RunnableParallel` to run three branches concurrently: one that retrieves relevant context from ChromaDB, one that passes the original question through, and one that formats the conversation history. All three feed into a prompt template that instructs the LLM to answer *only* from the provided context.

### Conversation Memory
Each request includes the last 8 messages of the conversation as `chat_history`. This allows the chatbot to handle follow-up questions and requests like "tell me that in Spanish" without losing context from earlier in the session.

### API Layer
LangServe wraps the chain in a FastAPI application, automatically exposing `/chat/invoke`, `/chat/stream`, and `/chat/playground` endpoints. This makes the chatbot accessible as a standard REST API.

### Frontend
A Streamlit interface provides a chat-style UI with message history, suggested questions, and a sidebar explaining the RAG architecture. It communicates with the LangServe backend via HTTP.

## Main Challenges and How I Overcame Them

### Challenge 1: Python version compatibility
The environment used Python 3.14, which had no pre-compiled wheels for FAISS. I switched to ChromaDB, which installs without compilation, persists to disk automatically, and integrates natively with LangChain.

### Challenge 2: Avoiding re-ingestion on every startup
Running embeddings on every server startup would be slow and expensive (OpenAI charges per token). I solved this by persisting the ChromaDB index to disk. On startup, the system checks if the index already exists and loads it directly, only running ingestion if the index is missing.

### Challenge 3: Ensuring answers stay grounded
Without careful prompt engineering, LLMs tend to fill gaps with hallucinated information. I addressed this with explicit instructions in the prompt: answer only from the provided context, and if the information is not there, say so honestly. Setting `temperature=0` further ensures deterministic, factual responses.

### Challenge 4: Retrieval coverage
With `k=6` some relevant chunks (particularly the client list, which uses the word "Organizations" instead of "clients") were not being retrieved. Increasing to `k=8` ensured broader coverage without significantly increasing the prompt size.

## Technologies Used

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | OpenAI GPT-4o-mini | Best cost/quality balance for Q&A |
| Embeddings | OpenAI text-embedding-3-small | High quality, cost-effective |
| Vector Store | ChromaDB | No compilation needed, persists to disk |
| Chain Framework | LangChain LCEL | Required by test, industry standard |
| API Server | LangServe + FastAPI | Required by test, auto-generates endpoints |
| Frontend | Streamlit | Fast to build, professional UI |
| Deployment | AWS EC2 + Docker | Reproducible, cloud-native |
