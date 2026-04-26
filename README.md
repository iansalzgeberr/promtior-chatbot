# Promtior RAG Chatbot

A conversational chatbot that uses **RAG (Retrieval Augmented Generation)** to answer questions about Promtior. Built with LangChain, LangServe, OpenAI, and Streamlit.

## What it does

The chatbot answers questions about Promtior by retrieving relevant information from two sources:
- **Promtior's website** (scraped automatically on first run)
- **The company presentation PDF** (included as an extra source for bonus points)

Instead of relying on what the LLM memorized during training, RAG ensures every answer is grounded in real, up-to-date content from those sources.

**Example questions it handles:**
- *"What services does Promtior offer?"*
- *"When was the company founded?"*
- *"Who are Promtior's clients?"*
- *"What is a Bionic Organization?"*

## Architecture

```
User
 │
 ▼
Streamlit UI (:8501)
 │  HTTP POST /chat/invoke
 ▼
LangServe API — FastAPI (:8000)
 │
 ▼
RAG Chain (LangChain LCEL)
 ├──────────────────────────────┐
 ▼                              ▼
Retriever                  GPT-4o-mini
 │                         (temperature=0)
 ▼
ChromaDB Vector Store
 │
 ▼
OpenAI Embeddings
(text-embedding-3-small)
 │
 ▼
Documents
 ├── Web scraping (promtior.ai)
 └── PDF presentation
```

**Ingestion flow** (runs once on first startup):
1. Scrape text from Promtior website pages
2. Load and parse the PDF presentation
3. Split all content into chunks (1000 chars, 200 overlap)
4. Generate embeddings via OpenAI API
5. Store vectors in ChromaDB (persisted to disk)

**Query flow** (on every user message):
1. Convert the question to a vector
2. Find the 8 most semantically similar chunks in ChromaDB
3. Inject chunks + conversation history + question into the prompt
4. GPT-4o-mini generates a grounded answer

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | GPT-4o-mini | Best cost/quality ratio for factual Q&A |
| Embeddings | text-embedding-3-small | Cost-effective, high quality |
| Vector Store | ChromaDB | No infra needed, persists to disk |
| Chain framework | LangChain LCEL | Required by test, industry standard |
| API server | LangServe + FastAPI | Required by test, auto-generates endpoints |
| Frontend | Streamlit | Clean chat UI with minimal code |
| Deployment | AWS EC2 + Docker | Cloud-native, reproducible |

## Prerequisites

- Python 3.11+ (tested on 3.14)
- OpenAI API key
- Docker + Docker Compose (for deployment)

## Quickstart (Windows)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/promtior-chatbot.git
cd promtior-chatbot

# 2. Create virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure your OpenAI API key
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...

# 4. Start everything (double-click or run from terminal)
start.bat
```

On first run, the server scrapes the website and builds the vector store (~30 seconds). Subsequent starts load from disk instantly.

**Services:**
| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| LangServe API | http://localhost:8000 |
| Interactive Playground | http://localhost:8000/chat/playground |
| Health check | http://localhost:8000/health |

## Quickstart (Linux/Mac)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then add your key

# Terminal 1 — backend
PYTHONIOENCODING=utf-8 python -m app.server

# Terminal 2 — frontend
streamlit run frontend/streamlit_app.py
```

## Docker

```bash
cp .env.example .env  # add your OPENAI_API_KEY

docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

## AWS EC2 Deployment

1. Launch an EC2 instance (Ubuntu 22.04, t2.micro or t3.small)
2. Open ports **8000** and **8501** in the Security Group inbound rules
3. SSH into the instance and run:

```bash
sudo apt update && sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu && newgrp docker

git clone https://github.com/YOUR_USERNAME/promtior-chatbot.git
cd promtior-chatbot
echo "OPENAI_API_KEY=sk-..." > .env

docker-compose up -d --build
```

4. Access at `http://YOUR_EC2_IP:8501`

See [`doc/aws_deploy_guide.txt`](doc/aws_deploy_guide.txt) for the full step-by-step guide including troubleshooting.

## API Reference

**Invoke (single question):**
```bash
curl -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "When was Promtior founded?", "chat_history": []}}'
```

**With conversation history:**
```bash
curl -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "question": "Tell me that in Spanish",
      "chat_history": [
        "User: What services does Promtior offer?",
        "Assistant: Promtior offers generative AI consulting and RAG implementation."
      ]
    }
  }'
```

## Project Structure

```
promtior-chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py           # Environment variables, model names, URLs
│   ├── ingest.py           # Web scraping + PDF loading + ChromaDB creation
│   ├── chain.py            # RAG chain with conversation memory (LCEL)
│   └── server.py           # LangServe FastAPI server
├── frontend/
│   └── streamlit_app.py    # Chat UI with history and suggested questions
├── data/
│   └── promtior_info.pdf   # Promtior presentation (extra source)
├── doc/
│   ├── project_overview.md     # Implementation approach and decisions
│   ├── component_diagram.png   # Architecture diagram
│   └── aws_deploy_guide.txt    # Full AWS deployment guide
├── vectorstore/            # ChromaDB index (auto-generated, gitignored)
├── Dockerfile              # Backend container
├── Dockerfile.frontend     # Frontend container
├── docker-compose.yml      # Orchestrates both services
├── start.bat               # Windows one-click launcher
├── requirements.txt
├── .env.example
└── .gitignore
```

## Documentation

See the [`/doc`](doc/) folder for:
- [`project_overview.md`](doc/project_overview.md) — approach, implementation logic, challenges and solutions
- [`component_diagram.png`](doc/component_diagram.png) — end-to-end architecture diagram
