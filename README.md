# Hybrid RAG Assistant

A hybrid retrieval-assisted generation (RAG) application built with Streamlit frontend and FastAPI backend. The project supports document ingestion, multi-strategy retrieval, metadata filtering, and streaming LLM responses with source citations.

## Architecture

Streamlit Frontend
  └─ calls FastAPI backend

FastAPI API Layer
  ├─ Document Upload Service
  │    ├─ PDF Parser
  │    ├─ DOCX Parser
  │    ├─ OCR Engine
  │    └─ Chunk Generator
  ├─ SQLite Metadata Store
  ├─ Hybrid Retrieval Engine
  │    ├─ BM25
  │    ├─ TF-IDF
  │    └─ Semantic Search
  ├─ Metadata Filtering
  ├─ Context Builder
  └─ LLM Layer
       ├─ OpenRouter
       └─ Ollama Fallback

Streaming Response → Source Citations

## Key Features

- Upload documents in formats: `pdf`, `docx`, `txt`, `csv`, `xlsx`, `json`, `md`, `png`, `jpg`, `jpeg`
- Parses text from files and scanned images
- Breaks documents into overlapping chunks for retrieval
- Uses BM25, TF-IDF, and semantic embeddings for hybrid retrieval
- Filters retrieval by document metadata when present in query
- Streams answers from an LLM with context-aware generation
- Returns source citations with document names and page references
- Persists indexed chunks and chat history to SQLite (`rag.db`)

## File Structure

- `frontend.py` — Streamlit frontend application
- `app/main.py` — FastAPI backend application
- `app/parser.py` — document parsing logic for supported formats
- `app/ocr.py` — OCR extraction for images and scanned PDFs
- `app/chunker.py` — text chunking and overlap generation
- `app/retriever.py` — hybrid retrieval using BM25, TF-IDF, and embeddings
- `app/database.py` — SQLite connection and schema setup
- `app/llm.py` — OpenRouter streaming, Ollama fallback, and prompt handling
- `app/agents.py` — task detection and optional query routing logic
- `docker-compose.yaml` — Docker Compose configuration for frontend and backend
- `Dockerfile` — backend container definition
- `Dockerfile.frontend` — frontend container definition
- `backend_requirements.txt` — backend Python dependencies
- `frontend_requirements.txt` — frontend Python dependencies
- `data/` — uploaded documents storage
- `rag.db` — SQLite metadata and history database

## Setup

### 1. Local Python environment

```powershell
cd "c:\new drive\HybridRag"
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r backend_requirements.txt
pip install -r frontend_requirements.txt
```

### 2. Environment variables

Create a `.env` file in the repository root with at least:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

If you use Ollama locally, ensure it is running and accessible at:

```text
http://host.docker.internal:11434
```

### 3. Run the backend

```powershell
cd "c:\new drive\HybridRag"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Run the frontend

In a second terminal:

```powershell
cd "c:\new drive\HybridRag"
streamlit run frontend.py --server.port=8501
```

### 5. Access the app

Open your browser at:

- `http://localhost:8501` for Streamlit frontend
- `http://localhost:8000/docs` for FastAPI documentation

## Docker Setup

Build and run both services with Docker Compose:

```powershell
docker-compose up --build
```

This starts:

- `backend` on port `8000`
- `frontend` on port `8501`

## Usage

1. Upload a supported document using the Streamlit sidebar.
2. The backend parses the file, chunks text, and stores metadata in SQLite.
3. Ask a question in the frontend.
4. The backend retrieves relevant chunks, builds context, and streams an answer.
5. The response includes cited source documents and page numbers.

## Notes

- `app/llm.py` first attempts OpenRouter models from `openrouter.ai`.
- If cloud models fail, it falls back to a local Ollama instance.
- `app/retriever.py` caches embeddings to `embeddings.pkl` for reuse.
- Uploaded documents and metadata are stored in `data/` and `rag.db`.

## Future Improvements

- Add explicit metadata filter UI in the frontend
- Include a document search or index browser
- Add better chunk de-duplication and similarity scoring
- Support additional LLM providers and configurable model selection
- Add authentication and persistent user sessions
