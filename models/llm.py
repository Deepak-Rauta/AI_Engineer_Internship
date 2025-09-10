import google.generativeai as genai
from typing import Optional
import logging
from config.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLLM:
    """Google Gemini LLM wrapper"""
    
    def __init__(self):
        """Initialize Gemini LLM with API key"""
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
            logger.info(f"Initialized Gemini model: {config.GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

    def generate_response(self, prompt: str, context: Optional[str] = None, response_mode: str = "detailed") -> str:
        """Generate response using Gemini"""
        try:
            system_prompt = self._build_system_prompt(response_mode)
            full_prompt = self._build_full_prompt(system_prompt, prompt, context)
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.TEMPERATURE,
                    max_output_tokens=config.MAX_TOKENS,
                )
            )
            return response.text
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def _build_system_prompt(self, response_mode: str) -> str:
        base_prompt = (
            "You are a helpful AI assistant specialized in document Q&A and web search."
            " You provide accurate, relevant answers based on the provided context."
        )
        if response_mode == "concise":
            return base_prompt + "\n\nProvide concise, direct answers. Keep responses brief and to the point."
        else:
            return base_prompt + "\n\nProvide detailed, comprehensive answers with explanations and examples when helpful."
    
    def _build_full_prompt(self, system_prompt: str, question: str, context: Optional[str]) -> str:
        prompt_parts = [system_prompt]
        if context:
            prompt_parts.append(f"\nContext from documents:\n{context}")
        prompt_parts.append(f"\nUser Question: {question}")
        prompt_parts.append("\nResponse:")
        return "\n".join(prompt_parts)

    def generate_search_query(self, question: str) -> str:
        try:
            prompt = (
                f"Convert this question into an optimized web search query.\n"
                f"Extract key terms and concepts for better search results.\n\n"
                f"Question: {question}\n\n"
                f"Search Query:"
            )

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=100,
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            return question  # fallback

# Global LLM instance
llm = GeminiLLM()
