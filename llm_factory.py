import google.generativeai as genai
from langchain_huggingface import HuggingFaceEndpoint
from config import GOOGLE_API_KEY, HUGGINGFACEHUB_API_TOKEN
from logger_config import setup_logger

logger = setup_logger(__name__)

# Configure GenAI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.error("GOOGLE_API_KEY not found in environment variables.")

class ModelFactory:
    @staticmethod
    def get_gemini_model(model_name: str, system_instruction: str = None):
        """
        Returns a configured Gemini GenerativeModel instance.
        """
        try:
            target_model = model_name or "gemini-2.5-flash-lite"
            logger.info(f"Initializing Gemini model: {target_model}")
            
            model = genai.GenerativeModel(
                model_name=target_model,
                system_instruction=system_instruction
            )
            return model
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise e

    @staticmethod
    def get_huggingface_model(model_name: str):
        """
        Returns a configured HuggingFaceEndpoint instance.
        """
        try:
            repo_id = model_name or "mistralai/Mistral-7B-Instruct-v0.2"
            logger.info(f"Initializing HuggingFace model: {repo_id}")
            
            if not HUGGINGFACEHUB_API_TOKEN:
                 logger.error("HUGGINGFACEHUB_API_TOKEN not found for HuggingFace model.")
                 return None

            return HuggingFaceEndpoint(
                repo_id=repo_id,
                huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                temperature=0.1,
                max_new_tokens=512
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {e}")
            raise e
