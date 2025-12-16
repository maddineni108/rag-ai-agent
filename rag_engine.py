from typing import List, Optional
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from logger_config import setup_logger
from llm_factory import ModelFactory

logger = setup_logger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "./chroma_db"

class RAGEngine:
    def __init__(self):
        logger.info("Initializing RAGEngine...")
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            logger.info(f"Embeddings model {EMBEDDING_MODEL_NAME} initialized.")
            
            self.vector_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=self.embeddings,
                collection_name="rag_collection"
            )
            logger.info(f"ChromaDB initialized at {PERSIST_DIRECTORY}")
            
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
        except Exception as e:
            logger.critical(f"Failed to initialize RAGEngine: {e}")
            raise e

    def ingest_documents(self, documents: List[Document]):
        """
        Add documents to the vector store.
        """
        if documents:
            try:
                self.vector_store.add_documents(documents)
                logger.info(f"Ingested {len(documents)} documents.")
            except Exception as e:
                logger.error(f"Error ingesting documents: {e}")

    def delete_collection(self):
        """
        Clears the vector store (use with caution).
        """
        try:
            self.vector_store.delete_collection()
            logger.warning("Vector store collection deleted.")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def get_all_documents(self) -> List[dict]:
        """
        List all documents in the collection (limited metadata).
        """
        try:
            data = self.vector_store.get()
            return [{"id": id, "metadata": meta} for id, meta in zip(data['ids'], data['metadatas'])]
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def delete_document(self, doc_id: str):
        try:
            self.vector_store.delete(ids=[doc_id])
            logger.info(f"Deleted document with ID: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")

    def chat(self, prompt: str, model_provider: str = "gemini", model_name: Optional[str] = None):
        """
        RAG Chat function.
        """
        logger.info(f"Chat request received. Provider: {model_provider}, Query: '{prompt}'")
        
        # 1. Retrieve Context
        try:
            docs = self.retriever.invoke(prompt)
            logger.info(f"Retrieved {len(docs)} documents for query.")
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            docs = []

        context_text = "\n\n".join(doc.page_content for doc in docs)

        # 2. Construct System Instruction (Context + Rules)
        system_instruction = f"""You are a helpful AI assistant, Created by Google and customized by Anil. 
Answer the user's question based ONLY on the provided context below. 
If the answer cannot be found in the context, politely state that you do not have that information.

---
CONTEXT:
{context_text}
---
"""

        # 3. Generate Response
        if model_provider.lower() == "gemini":
            try:
                model = ModelFactory.get_gemini_model(model_name, system_instruction)
                response = model.generate_content(prompt)
                logger.info("Gemini response generated successfully.")
                return response.text
            except Exception as e:
                logger.error(f"Error calling Gemini API: {str(e)}")
                return f"Error calling Gemini API: {str(e)}"
        
        elif model_provider.lower() == "huggingface":
            # For HF, use ModelFactory to get LLM, then invoke
            full_prompt = f"{system_instruction}\n\nUser Question: {prompt}"
            
            try:
                llm = ModelFactory.get_huggingface_model(model_name)
                if not llm:
                    return "Error: Could not initialize HuggingFace model."
                
                response = llm.invoke(full_prompt)
                logger.info("HuggingFace response generated successfully.")
                return response
            except Exception as e:
                logger.error(f"Error calling HuggingFace API: {str(e)}")
                return f"Error calling HuggingFace API: {str(e)}"
            
        else:
            logger.warning(f"Unsupported model provider requested: {model_provider}")
            return f"Unsupported model provider: {model_provider}"

# Singleton instance
rag_engine = RAGEngine()
