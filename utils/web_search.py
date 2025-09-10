"""
Web search utilities using SERPAPI
Handles real-time web search integration
"""
import requests
from typing import List, Dict, Any, Optional
import logging
from config.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearcher:
    """Handles web search using SERPAPI"""
    
    def __init__(self):
        self.api_key = config.SERPAPI_KEY
        self.base_url = "https://serpapi.com/search"
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform web search using SERPAPI
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            params = {
                'q': query,
                'api_key': self.api_key,
                'engine': 'google',
                'num': num_results,
                'gl': 'us',
                'hl': 'en'
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract organic results
            results = []
            if 'organic_results' in data:
                for result in data['organic_results']:
                    results.append({
                        'title': result.get('title', ''),
                        'link': result.get('link', ''),
                        'snippet': result.get('snippet', ''),
                        'source': result.get('source', '')
                    })
            
            logger.info(f"Found {len(results)} web search results for: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return []
    
    def get_search_context(self, query: str, num_results: int = 3) -> str:
        """
        Get formatted search context for LLM
        
        Args:
            query: Search query
            num_results: Number of results to include
            
        Returns:
            Formatted search context
        """
        try:
            results = self.search(query, num_results)
            
            if not results:
                return "No web search results found."
            
            context_parts = ["Web Search Results:"]
            
            for i, result in enumerate(results, 1):
                context_parts.append(f"\n{i}. {result['title']}")
                context_parts.append(f"   Source: {result['link']}")
                context_parts.append(f"   Summary: {result['snippet']}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting search context: {e}")
            return "Error retrieving web search results."

# Global web searcher instance
web_searcher = WebSearcher()