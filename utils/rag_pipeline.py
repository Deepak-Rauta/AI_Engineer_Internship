import logging
from models.embeddings import get_embedding_model
from utils.vector_store import get_vector_store
from utils.document_processor import get_document_processor
from utils.web_search import get_web_searcher
from config.config import Config

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()
        self.document_processor = get_document_processor()
        self.web_searcher = get_web_searcher()
    
    def add_documents_from_files(self, uploaded_files):
        results = {
            'processed_files': [],
            'total_chunks': 0,
            'errors': []
        }
        
        all_chunks = []
        all_metadata = []
        
        for uploaded_file in uploaded_files:
            try:
                text = self.document_processor.process_uploaded_file(uploaded_file)
                chunks = self.document_processor.chunk_text(text)
                
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
                error_msg = f"Error with {uploaded_file.name}: {str(e)}"
                results['errors'].append(error_msg)
                results['processed_files'].append({
                    'name': uploaded_file.name,
                    'chunks': 0,
                    'status': 'error',
                    'error': str(e)
                })
        
        if all_chunks:
            embeddings = self.embedding_model.encode_documents(all_chunks)
            self.vector_store.add_documents(all_chunks, embeddings, all_metadata)
            results['total_chunks'] = len(all_chunks)
        
        return results
    
    def retrieve_context(self, query, use_web_search=True):
        context_parts = []
        source_info = []
        
        # Try document search first
        rag_results = self._search_documents(query)
        
        if rag_results:
            context_parts.append("Document Knowledge:")
            for doc, similarity, metadata in rag_results:
                context_parts.append(f"- {doc[:500]}...")
                source_info.append(f"Doc: {metadata.get('file_name', 'Unknown')} ({similarity:.3f})")
        
        # Try web search if needed
        if (not rag_results or len(rag_results) < 2) and use_web_search:
            web_results = self._search_web(query)
            
            if web_results:
                if context_parts:
                    context_parts.append("\nWeb Results:")
                else:
                    context_parts.append("Web Results:")
                
                for result in web_results:
                    context_parts.append(f"- {result['title']}: {result['snippet']}")
                    source_info.append(f"Web: {result['title']}")
        
        context_text = "\n".join(context_parts) if context_parts else ""
        source_text = "\n".join(source_info) if source_info else "No sources found"
        
        return context_text, source_text
    
    def _search_documents(self, query):
        try:
            query_embedding = self.embedding_model.encode_text(query)[0]
            results = self.vector_store.search(
                query_embedding, 
                top_k=Config.MAX_CHUNKS,
                threshold=Config.SIMILARITY_THRESHOLD
            )
            return results
        except Exception as e:
            logger.error(f"Document search error: {str(e)}")
            return []
    
    def _search_web(self, query):
        try:
            if not self.web_searcher.is_configured():
                return []
            
            results = self.web_searcher.search(query, num_results=3)
            return results
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []
    
    def get_pipeline_stats(self):
        vector_stats = self.vector_store.get_stats()
        return {
            'vector_store': vector_stats,
            'web_search_configured': self.web_searcher.is_configured(),
            'embedding_model': self.embedding_model.model_name
        }
    
    def clear_knowledge_base(self):
        self.vector_store.clear_store()

_rag_pipeline_instance = None

def get_rag_pipeline():
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline()
    return _rag_pipeline_instance
