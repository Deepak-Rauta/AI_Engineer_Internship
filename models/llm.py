import google.generativeai as genai
from config.config import Config
import streamlit as st
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLLM:
    """Gemini AI Language Model wrapper"""
    
    def __init__(self):
        """Initialize Gemini LLM with API key"""
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
            logger.info("Gemini LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
            raise
    
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        response_mode: str = "detailed",
        **kwargs
    ) -> str:
        """
        Generate response using Gemini AI
        
        Args:
            prompt: User query
            context: Retrieved context from RAG or web search
            response_mode: 'concise' or 'detailed'
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
        try:
            # Prepare the full prompt
            full_prompt = self._prepare_prompt(prompt, context, response_mode)
            
            # Configure generation parameters
            generation_config = {
                'temperature': kwargs.get('temperature', Config.TEMPERATURE),
                'max_output_tokens': kwargs.get('max_tokens', Config.MAX_TOKENS),
            }
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return "I apologize, but I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"
    
    def _prepare_prompt(self, query: str, context: Optional[str], response_mode: str) -> str:
        """Prepare the complete prompt for Gemini"""
        
        # Base system prompt
        system_prompt = """You are an intelligent AI assistant with access to relevant information. 
        Provide helpful, accurate, and contextual responses based on the given information."""
        
        # Mode-specific instructions
        if response_mode == "concise":
            mode_instruction = "Provide a concise, brief response (2-3 sentences maximum)."
        else:
            mode_instruction = "Provide a detailed, comprehensive response with explanations and examples where appropriate."
        
        # Construct full prompt
        if context:
            full_prompt = f"""{system_prompt}
            
{mode_instruction}

Context Information:
{context}

User Question: {query}

Response:"""
        else:
            full_prompt = f"""{system_prompt}
            
{mode_instruction}

User Question: {query}

Response:"""
        
        return full_prompt
    
    def test_connection(self) -> bool:
        """Test if the Gemini API connection is working"""
        try:
            test_response = self.model.generate_content("Hello, this is a test.")
            return bool(test_response and test_response.text)
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

# Singleton instance
_gemini_instance = None

def get_gemini_llm() -> GeminiLLM:
    """Get singleton instance of Gemini LLM"""
    global _gemini_instance
    if _gemini_instance is None:
        _gemini_instance = GeminiLLM()
    return _gemini_instance
