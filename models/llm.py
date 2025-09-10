import google.generativeai as genai
from config.config import Config
import logging

logger = logging.getLogger(__name__)

class GeminiLLM:
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
    
    def generate_response(self, prompt, context=None, response_mode="detailed", **kwargs):
        try:
            full_prompt = self._build_prompt(prompt, context, response_mode)
            
            config = {
                'temperature': Config.TEMPERATURE,
                'max_output_tokens': Config.MAX_TOKENS,
            }
            
            response = self.model.generate_content(full_prompt, generation_config=config)
            
            if response and response.text:
                return response.text.strip()
            else:
                return "Sorry, I couldn't generate a response right now."
                
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return f"Error: {str(e)}"
    
    def _build_prompt(self, query, context, response_mode):
        system_msg = "You are a helpful AI assistant. Answer questions based on the provided information."
        
        if response_mode == "concise":
            mode_msg = "Keep your answer brief and to the point."
        else:
            mode_msg = "Provide a detailed explanation with examples if helpful."
        
        if context:
            full_prompt = f"{system_msg}\n\n{mode_msg}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            full_prompt = f"{system_msg}\n\n{mode_msg}\n\nQuestion: {query}\n\nAnswer:"
        
        return full_prompt

_gemini_instance = None

def get_gemini_llm():
    global _gemini_instance
    if _gemini_instance is None:
        _gemini_instance = GeminiLLM()
    return _gemini_instance
