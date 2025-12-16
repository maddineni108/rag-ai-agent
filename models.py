from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    query: str
    model_provider: str = "gemini" # gemini or huggingface
    model_name: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

class DocumentInfo(BaseModel):
    id: str
    metadata: dict
