# RAG AI Agent

A Retrieval-Augmented Generation (RAG) API built with FastAPI, supporting multiple LLM providers (Google Gemini, HuggingFace) and local vector search.

## Features

-   **Multi-Model Support**: Switch between Google Gemini (e.g., `gemini-2.5-flash-lite`) and HuggingFace models.
-   **RAG Engine**: Upload PDF or TXT documents to build a knowledge base.
-   **Vector Search**: Uses `ChromaDB` for local vector storage and retrieval.
-   **Observability**: Integrated Prometheus metrics for monitoring (`/metrics`).
-   **Logging**: Comprehensive logging to console and `logs/app.log`.

## Prerequisites

-   Python 3.9+
-   Google API Key (for Gemini)
-   HuggingFace API Token (for HF models and embeddings)

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/maddineni108/rag-ai-agent.git
    cd rag-ai-agent
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_google_api_key
    HUGGINGFACEHUB_API_TOKEN=your_hf_token
    # Optional: LangChain Tracing
    # LANGCHAIN_TRACING_V2=true
    # LANGCHAIN_API_KEY=your_langchain_key
    ```

## Running the Application

Start the server using `uvicorn`:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Usage

### 1. Upload Documents
**POST** `/upload-doc`
Upload a PDF or TXT file to the knowledge base.

### 2. Chat
**POST** `/chat`

```json
{
  "query": "What is the content of the uploaded document?",
  "model_provider": "gemini",
  "model_name": "gemini-2.5-flash-lite"
}
```

### 3. Monitoring
Metrics are available at `/metrics` for Prometheus scraping.
-   `rag_chat_requests_total`: Total chat requests.
-   `rag_documents_uploaded_total`: Total uploads.
-   `rag_chat_generation_seconds`: Latency tracking.

## Project Structure

-   `main.py`: FastAPI application and endpoints.
-   `rag_engine.py`: Core RAG logic (Retrieval + Generation).
-   `models.py`: Pydantic data schemas.
-   `llm_factory.py`: Factory for initializing LLMs.
-   `logger_config.py`: Centralized logging configuration.
