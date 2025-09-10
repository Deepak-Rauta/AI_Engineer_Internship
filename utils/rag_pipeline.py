"""
RAG Pipeline utilities
Combines document retrieval with web search
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
from models.embeddings import embedding_model
from utils.web_search import web_searcher
from models.llm import llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Handles the complete RAG pipeline"""
    
    def __init__(self):
        self.embedding_model = embedding_model
        self.web_searcher = web_searcher
        self.llm = llm
    
    def process_query(
        self, 
        query: str, 
        response_mode: str = "detailed",
        use_web_search: bool = True
    ) -> Dict[str, Any]:
        """
        Process user query through RAG pipeline
        
        Args:
            query: User question
            response_mode: 'concise' or 'detailed'
            use_web_search: Whether to use web search
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Step 1: Search documents
            doc_results = self.embedding_model.search_similar_documents(query)
            
            # Step 2: Prepare document context
            doc_context = self._format_document_context(doc_results)
            
            # Step 3: Web search if needed
            web_context = ""
            if use_web_search and (not doc_results or len(doc_results) < 2):
                search_query = self.llm.generate_search_query(query)
                web_context = self.web_searcher.get_search_context(search_query)
            
            # Step 4: Combine contexts
            full_context = self._combine_contexts(doc_context, web_context)
            
            # Step 5: Generate response
            response = self.llm.generate_response(query, full_context, response_mode)
            
            return {
                'response': response,
                'doc_sources': len(doc_results),
                'web_search_used': bool(web_context),
                'context_length': len(full_context)
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'response': f"Sorry, I encountered an error processing your query: {str(e)}",
                'doc_sources': 0,
                'web_search_used': False,
                'context_length': 0
            }
    
    def _format_document_context(self, doc_results: List[Tuple[str, float]]) -> str:
        """Format document search results for context"""
        if not doc_results:
            return ""
        
        context_parts = ["Relevant Document Excerpts:"]
        
        for i, (doc, score) in enumerate(doc_results, 1):
            context_parts.append(f"\n[Document {i}] (Relevance: {score:.2f})")
            context_parts.append(doc[:500] + "..." if len(doc) > 500 else doc)
        
        return "\n".join(context_parts)
    
    def _combine_contexts(self, doc_context: str, web_context: str) -> str:
        """Combine document and web search contexts"""
        contexts = []
        
        if doc_context:
            contexts.append(doc_context)
        
        if web_context:
            contexts.append(web_context)
        
        return "\n\n".join(contexts)
    
    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the RAG system"""
        try:
            if documents:
                self.embedding_model.build_vector_index(documents)
                logger.info(f"Added {len(documents)} documents to RAG system")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

# Global RAG pipeline instance
rag_pipeline = RAGPipeline()