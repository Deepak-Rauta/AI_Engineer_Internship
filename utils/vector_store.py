import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Simple vector store for RAG functionality"""
    
    def __init__(self, store_path: str = None):
        """
        Initialize vector store
        
        Args:
            store_path: Path to store vector data
        """
        self.store_path = store_path or Config.VECTOR_STORE_PATH
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        # Create store directory if it doesn't exist
        os.makedirs(self.store_path, exist_ok=True)
        
        # Load existing data if available
        self._load_store()
    
    def add_documents(
        self, 
        documents: List[str], 
        embeddings: np.ndarray, 
        metadata: List[Dict[str, Any]] = None
    ):
        """
        Add documents and their embeddings to the store
        
        Args:
            documents: List of document texts
            embeddings: Document embeddings array
            metadata: Optional metadata for each document
        """
        try:
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            
            # Initialize metadata if not provided
            if metadata is None:
                metadata = [{'id': i, 'source': 'unknown'} for i in range(len(documents))]
            
            # Add to existing data
            self.documents.extend(documents)
            self.metadata.extend(metadata)
            
            # Concatenate embeddings
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
            
            # Save updated store
            self._save_store()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5, 
        threshold: float = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (document, similarity, metadata) tuples
        """
        try:
            if self.embeddings is None or len(self.documents) == 0:
                logger.warning("Vector store is empty")
                return []
            
            # Compute similarities
            similarities = self._compute_similarities(query_embedding, self.embeddings)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Apply threshold if provided
            threshold = threshold or Config.SIMILARITY_THRESHOLD
            
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity >= threshold:
                    results.append((
                        self.documents[idx],
                        float(similarity),
                        self.metadata[idx]
                    ))
            
            logger.info(f"Found {len(results)} similar documents above threshold {threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def _compute_similarities(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between query and document embeddings"""
        try:
            # Normalize embeddings
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            
            # Compute cosine similarity
            similarities = np.dot(doc_norms, query_norm)
            return similarities
            
        except Exception as e:
            logger.error(f"Error computing similarities: {str(e)}")
            raise
    
    def _save_store(self):
        """Save vector store to disk"""
        try:
            store_data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }
            
            store_file = os.path.join(self.store_path, 'vector_store.pkl')
            with open(store_file, 'wb') as f:
                pickle.dump(store_data, f)
            
            logger.info(f"Vector store saved to {store_file}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def _load_store(self):
        """Load vector store from disk"""
        try:
            store_file = os.path.join(self.store_path, 'vector_store.pkl')
            
            if os.path.exists(store_file):
                with open(store_file, 'rb') as f:
                    store_data = pickle.load(f)
                
                self.documents = store_data.get('documents', [])
                self.embeddings = store_data.get('embeddings', None)
                self.metadata = store_data.get('metadata', [])
                
                logger.info(f"Loaded vector store with {len(self.documents)} documents")
            else:
                logger.info("No existing vector store found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            # Initialize empty store on error
            self.documents = []
            self.embeddings = None
            self.metadata = []
    
    def clear_store(self):
        """Clear all data from the vector store"""
        try:
            self.documents = []
            self.embeddings = None
            self.metadata = []
            
            # Remove store file
            store_file = os.path.join(self.store_path, 'vector_store.pkl')
            if os.path.exists(store_file):
                os.remove(store_file)
            
            logger.info("Vector store cleared")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'num_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'store_path': self.store_path
        }

# Singleton instance
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """Get singleton instance of vector store"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
