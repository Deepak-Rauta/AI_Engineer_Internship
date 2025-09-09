import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the RAG Chatbot"""
    
    # API Keys - Set these as environment variables
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    
    # Model Configuration
    GEMINI_MODEL = "gemini-1.5-flash"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # RAG Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CHUNKS = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Response Configuration
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    
    # File Upload Configuration
    ALLOWED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md']
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Vector Store Configuration
    VECTOR_STORE_PATH = "vector_store"
    
    # Web Search Configuration
    SEARCH_RESULTS_COUNT = 5
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        required_keys = ['GEMINI_API_KEY']
        missing_keys = []
        
        for key in required_keys:
            if not getattr(cls, key) or getattr(cls, key) == f"your-{key.lower().replace('_', '-')}-here":
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration: {', '.join(missing_keys)}")
        
        return True
