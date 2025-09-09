import requests
from typing import List, Dict, Any, Optional
import logging
from config.config import Config
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearcher:
    """Web search functionality using SerpAPI"""
    
    def __init__(self):
        self.api_key = Config.SERPAPI_KEY
        self.base_url = "https://serpapi.com/search"
        self.results_count = Config.SEARCH_RESULTS_COUNT
    
    def search(self, query: str, num_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform web search using SerpAPI
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search result dictionaries
        """
        try:
            if not self.api_key or self.api_key == "your-serpapi-key-here":
                logger.warning("SerpAPI key not configured, using fallback search")
                return self._fallback_search(query)
            
            num_results = num_results or self.results_count
            
            params = {
                'q': query,
                'api_key': self.api_key,
                'engine': 'google',
                'num': num_results,
                'format': 'json'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract organic results
            organic_results = data.get('organic_results', [])
            
            search_results = []
            for result in organic_results:
                search_results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', ''),
                    'link': result.get('link', ''),
                    'source': 'web_search'
                })
            
            logger.info(f"Found {len(search_results)} search results for query: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}")
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Fallback search when API is not available
        
        Args:
            query: Search query
            
        Returns:
            Mock search results
        """
        logger.info("Using fallback search results")
        
        # Return mock results indicating web search is not configured
        return [{
            'title': 'Web Search Not Configured',
            'snippet': f'Web search for "{query}" is not available. Please configure SerpAPI key in config/config.py to enable live web search functionality.',
            'link': 'https://serpapi.com/',
            'source': 'fallback'
        }]
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a readable string for LLM context
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted search results string
        """
        try:
            if not results:
                return "No search results found."
            
            formatted_results = "Web Search Results:\n\n"
            
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No Title')
                snippet = result.get('snippet', 'No description available')
                link = result.get('link', '')
                
                formatted_results += f"{i}. {title}\n"
                formatted_results += f"   {snippet}\n"
                if link:
                    formatted_results += f"   Source: {link}\n"
                formatted_results += "\n"
            
            return formatted_results.strip()
            
        except Exception as e:
            logger.error(f"Error formatting search results: {str(e)}")
            return "Error formatting search results."
    
    def is_configured(self) -> bool:
        """Check if web search is properly configured"""
        return bool(self.api_key and self.api_key != "your-serpapi-key-here")

# Singleton instance
_web_searcher_instance = None

def get_web_searcher() -> WebSearcher:
    """Get singleton instance of web searcher"""
    global _web_searcher_instance
    if _web_searcher_instance is None:
        _web_searcher_instance = WebSearcher()
    return _web_searcher_instance
