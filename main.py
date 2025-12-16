from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
import shutil
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from rag_engine import rag_engine
from logger_config import setup_logger
from models import ChatRequest, ChatResponse, DocumentInfo
from prometheus_fastapi_instrumentator import Instrumentator

logger = setup_logger(__name__)

app = FastAPI(title="RAG AI Agent API")

# Setup Prometheus Instrumentation
Instrumentator().instrument(app).expose(app)

@app.get("/")
def read_root():
    return {"message": "Welcome to RAG AI Agent API"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Received chat request for model: {request.model_provider} / {request.model_name}")
    try:
        response = rag_engine.chat(
            prompt=request.query, 
            model_provider=request.model_provider, 
            model_name=request.model_name
        )
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-doc")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"Received upload request for file: {file.filename}")
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        documents = []
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
        elif file.filename.endswith(".txt"):
            loader = TextLoader(temp_file_path)
            documents = loader.load()
        else:
             raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and TXT supported.")
        
        # Add metadata (filename) to each doc if not present
        for doc in documents:
            doc.metadata["filename"] = file.filename

        rag_engine.ingest_documents(documents)
        
        return {"message": f"Successfully processed {file.filename}", "chunks": len(documents)}
    
    except Exception as e:
        logger.error(f"Upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/list-docs", response_model=List[DocumentInfo])
def list_documents():
    return rag_engine.get_all_documents()

@app.delete("/delete-doc/{doc_id}")
def delete_document(doc_id: str):
    try:
        rag_engine.delete_document(doc_id)
        logger.info(f"Deleted document via API: {doc_id}")
        return {"message": f"Document {doc_id} deleted"}
    except Exception as e:
        logger.error(f"Delete endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
