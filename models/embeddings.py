from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging
from config.config import Config
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Sentence Transformer based embedding model for RAG"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model '{self.model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text into embeddings
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if isinstance(text, str):
                text = [text]
            
            embeddings = self.model.encode(text, convert_to_numpy=True)
            logger.info(f"Encoded {len(text)} text(s) into embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode multiple documents into embeddings
        
        Args:
            documents: List of document texts
            
        Returns:
            Numpy array of document embeddings
        """
        try:
            embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
            logger.info(f"Encoded {len(documents)} documents into embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding documents: {str(e)}")
            raise
    
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding matrix
            
        Returns:
            Similarity scores array
        """
        try:
            # Normalize embeddings
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            
            # Compute cosine similarity
            similarities = np.dot(doc_norms, query_norm)
            return similarities
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise
    
    def find_most_similar(
        self, 
        query: str, 
        documents: List[str], 
        doc_embeddings: np.ndarray = None,
        top_k: int = 5,
        threshold: float = None
    ) -> List[tuple]:
        """
        Find most similar documents to query
        
        Args:
            query: Query text
            documents: List of document texts
            doc_embeddings: Pre-computed document embeddings (optional)
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (document, similarity_score, index) tuples
        """
        try:
            # Encode query
            query_embedding = self.encode_text(query)[0]
            
            # Encode documents if not provided
            if doc_embeddings is None:
                doc_embeddings = self.encode_documents(documents)
            
            # Compute similarities
            similarities = self.compute_similarity(query_embedding, doc_embeddings)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter by threshold if provided
            threshold = threshold or Config.SIMILARITY_THRESHOLD
            results = []
            
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity >= threshold:
                    results.append((documents[idx], float(similarity), int(idx)))
            
            logger.info(f"Found {len(results)} similar documents above threshold {threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            raise
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Embeddings saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file"""
        try:
            with open(filepath, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"Embeddings loaded from {filepath}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise

# Singleton instance
_embedding_instance = None

def get_embedding_model() -> EmbeddingModel:
    """Get singleton instance of embedding model"""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingModel()
    return _embedding_instance
