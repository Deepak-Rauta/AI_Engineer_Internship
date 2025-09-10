import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
class Config:
    """Configuration class for managing API keys and settings"""
    
    def __init__(self):
        # API Keys - Replace with your actual keys or use environment variables
        self.GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.SERPAPI_KEY = os.getenv("SERPAPI_KEY")
        
        # Model Settings
        self.GEMINI_MODEL = "gemini-1.5-flash"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        
        # RAG Settings
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.MAX_CHUNKS = 5
        self.SIMILARITY_THRESHOLD = 0.7
        
        # Response Settings
        self.MAX_TOKENS = 1000
        self.TEMPERATURE = 0.7
        
        # File Upload Settings
        self.ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.md']
        self.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        
        # UI Settings
        self.APP_TITLE = "🤖 RAG Document Q&A Chatbot"
        self.APP_DESCRIPTION = "Upload documents and ask questions with AI-powered search"
    
    def get_gemini_config(self) -> Dict[str, Any]:
        """Get Gemini API configuration"""
        return {
            "api_key": self.GEMINI_API_KEY,
            "model": self.GEMINI_MODEL,
            "temperature": self.TEMPERATURE,
            "max_tokens": self.MAX_TOKENS,
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding model configuration"""
        return {
            "model_name": self.EMBEDDING_MODEL,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
        }
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration"""
        return {
            "serpapi_key": self.SERPAPI_KEY,
            "max_chunks": self.MAX_CHUNKS,
            "similarity_threshold": self.SIMILARITY_THRESHOLD,
        }

# Global config instance
config = Config()
