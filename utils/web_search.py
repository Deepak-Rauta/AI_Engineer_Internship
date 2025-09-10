import requests
import logging
from config.config import Config

logger = logging.getLogger(__name__)

class WebSearcher:
    def __init__(self):
        self.api_key = Config.SERPAPI_KEY
        self.base_url = "https://serpapi.com/search"
        self.results_count = Config.SEARCH_RESULTS_COUNT
    
    def search(self, query, num_results=None):
        if not self.api_key:
            return self._mock_results(query)
        
        num_results = num_results or self.results_count
        
        params = {
            'q': query,
            'api_key': self.api_key,
            'engine': 'google',
            'num': num_results,
            'format': 'json'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            organic_results = data.get('organic_results', [])
            search_results = []
            
            for result in organic_results:
                search_results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', ''),
                    'link': result.get('link', ''),
                    'source': 'web_search'
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return self._mock_results(query)
    
    def _mock_results(self, query):
        return [{
            'title': 'Web Search Not Available',
            'snippet': f'Search for "{query}" requires SerpAPI configuration.',
            'link': 'https://serpapi.com/',
            'source': 'fallback'
        }]
    
    def format_results(self, results):
        if not results:
            return "No search results found."
        
        formatted = "Web Search Results:\n\n"
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No Title')
            snippet = result.get('snippet', 'No description')
            link = result.get('link', '')
            
            formatted += f"{i}. {title}\n{snippet}\n"
            if link:
                formatted += f"Source: {link}\n"
            formatted += "\n"
        
        return formatted.strip()
    
    def is_configured(self):
        return bool(self.api_key)

_web_searcher_instance = None

def get_web_searcher():
    global _web_searcher_instance
    if _web_searcher_instance is None:
        _web_searcher_instance = WebSearcher()
    return _web_searcher_instance
