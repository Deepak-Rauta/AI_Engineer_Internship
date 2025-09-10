import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
    
    GEMINI_MODEL = "gemini-1.5-flash"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CHUNKS = 5
    SIMILARITY_THRESHOLD = 0.7
    
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    
    ALLOWED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md']
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    VECTOR_STORE_PATH = "vector_store"
    SEARCH_RESULTS_COUNT = 5
    
    @classmethod
    def validate_config(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        return True
