from typing import List, Dict, Any, Optional, Tuple
import logging
from models.embeddings import get_embedding_model
from utils.vector_store import get_vector_store
from utils.document_processor import get_document_processor
from utils.web_search import get_web_searcher
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Complete RAG pipeline for document processing and retrieval"""
    
    def __init__(self):
        """Initialize RAG pipeline components"""
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()
        self.document_processor = get_document_processor()
        self.web_searcher = get_web_searcher()
        
        logger.info("RAG Pipeline initialized")
    
    def add_documents_from_files(self, uploaded_files) -> Dict[str, Any]:
        """
        Process uploaded files and add to vector store
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            Processing results dictionary
        """
        try:
            results = {
                'processed_files': [],
                'total_chunks': 0,
                'errors': []
            }
            
            all_chunks = []
            all_metadata = []
            
            for uploaded_file in uploaded_files:
                try:
                    # Process file
                    text = self.document_processor.process_uploaded_file(uploaded_file)
                    
                    # Chunk text
                    chunks = self.document_processor.chunk_text(text)
                    
                    # Create metadata for chunks
                    for chunk in chunks:
                        chunk_metadata = {
                            'file_name': uploaded_file.name,
                            'chunk_id': chunk['id'],
                            'source': 'uploaded_file',
                            'length': chunk['length']
                        }
                        all_chunks.append(chunk['text'])
                        all_metadata.append(chunk_metadata)
                    
                    results['processed_files'].append({
                        'name': uploaded_file.name,
                        'chunks': len(chunks),
                        'status': 'success'
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing {uploaded_file.name}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    results['processed_files'].append({
                        'name': uploaded_file.name,
                        'chunks': 0,
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Generate embeddings for all chunks
            if all_chunks:
                embeddings = self.embedding_model.encode_documents(all_chunks)
                
                # Add to vector store
                self.vector_store.add_documents(all_chunks, embeddings, all_metadata)
                
                results['total_chunks'] = len(all_chunks)
                logger.info(f"Successfully processed {len(all_chunks)} chunks from {len(uploaded_files)} files")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {str(e)}")
            raise
    
    def retrieve_context(self, query: str, use_web_search: bool = True) -> Tuple[str, str]:
        """
        Retrieve relevant context for a query using RAG and optionally web search
        
        Args:
            query: User query
            use_web_search: Whether to use web search as fallback
            
        Returns:
            Tuple of (context_text, source_info)
        """
        try:
            context_parts = []
            source_info = []
            
            # First, try RAG retrieval
            rag_results = self._retrieve_from_documents(query)
            
            if rag_results:
                context_parts.append("Document Knowledge:")
                for doc, similarity, metadata in rag_results:
                    context_parts.append(f"- {doc[:500]}...")
                    source_info.append(f"Document: {metadata.get('file_name', 'Unknown')} (similarity: {similarity:.3f})")
            
            # If no good RAG results and web search is enabled, try web search
            if (not rag_results or len(rag_results) < 2) and use_web_search:
                web_results = self._retrieve_from_web(query)
                
                if web_results:
                    if context_parts:
                        context_parts.append("\nWeb Search Results:")
                    else:
                        context_parts.append("Web Search Results:")
                    
                    for result in web_results:
                        context_parts.append(f"- {result['title']}: {result['snippet']}")
                        source_info.append(f"Web: {result['title']} ({result['link']})")
            
            # Combine context
            context_text = "\n".join(context_parts) if context_parts else ""
            source_text = "\n".join(source_info) if source_info else "No sources found"
            
            return context_text, source_text
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return "", f"Error retrieving context: {str(e)}"
    
    def _retrieve_from_documents(self, query: str) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Retrieve relevant documents using RAG"""
        try:
            logger.info(f"[RAG DEBUG] Query: {query}")
            
            # Check if vector store has documents
            stats = self.vector_store.get_stats()
            num_docs = stats.get('num_documents', 0)
            logger.info(f"[RAG DEBUG] Vector store has {num_docs} documents")
            
            if num_docs == 0:
                logger.warning("[RAG DEBUG] No documents in vector store!")
                return []
            
            # Encode query
            query_embedding = self.embedding_model.encode_text(query)[0]
            logger.info(f"[RAG DEBUG] Query embedding shape: {query_embedding.shape if hasattr(query_embedding, 'shape') else 'unknown'}")
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding, 
                top_k=Config.MAX_CHUNKS,
                threshold=Config.SIMILARITY_THRESHOLD
            )
            
            logger.info(f"[RAG DEBUG] Retrieved {len(results)} documents from RAG")
            for i, (doc, similarity, metadata) in enumerate(results):
                logger.info(f"[RAG DEBUG] Result {i+1}: similarity={similarity:.3f}, file={metadata.get('file_name', 'Unknown')}")
                logger.info(f"[RAG DEBUG] Content preview: {doc[:100]}...")
            
            if not results and Config.SIMILARITY_THRESHOLD > 0.5:
                logger.info(f"[RAG DEBUG] No results with threshold {Config.SIMILARITY_THRESHOLD}, trying lower threshold 0.3")
                results = self.vector_store.search(
                    query_embedding, 
                    top_k=Config.MAX_CHUNKS,
                    threshold=0.3
                )
                logger.info(f"[RAG DEBUG] With lower threshold: {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {str(e)}")
            return []

    def _retrieve_from_web(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve information using web search"""
        try:
            if not self.web_searcher.is_configured():
                logger.warning("Web search not configured")
                return []
            
            results = self.web_searcher.search(query, num_results=3)
            logger.info(f"Retrieved {len(results)} results from web search")
            return results
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        try:
            vector_stats = self.vector_store.get_stats()
            
            return {
                'vector_store': vector_stats,
                'web_search_configured': self.web_searcher.is_configured(),
                'embedding_model': self.embedding_model.model_name
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {str(e)}")
            return {}
    
    def clear_knowledge_base(self):
        """Clear all documents from the knowledge base"""
        try:
            self.vector_store.clear_store()
            logger.info("Knowledge base cleared")
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            raise

# Singleton instance
_rag_pipeline_instance = None

def get_rag_pipeline() -> RAGPipeline:
    """Get singleton instance of RAG pipeline"""
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline()
    return _rag_pipeline_instance
