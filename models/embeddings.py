"""
RAG Embedding Models
Handles document embeddings and vector search
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import faiss
import pickle
import os
import logging
from config.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Handles document embeddings and vector search"""
    
    def __init__(self):
        """Initialize embedding model"""
        try:
            self.model = SentenceTransformer(config.EMBEDDING_MODEL)
            self.index = None
            self.documents = []
            self.embeddings = None
            logger.info(f"Initialized embedding model: {config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def create_embeddings(self, documents: List[str]) -> np.ndarray:
        """
        Create embeddings for documents
        
        Args:
            documents: List of document chunks
            
        Returns:
            Numpy array of embeddings
        """
        try:
            logger.info(f"Creating embeddings for {len(documents)} documents")
            embeddings = self.model.encode(documents, show_progress_bar=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def build_vector_index(self, documents: List[str]) -> None:
        """
        Build FAISS vector index from documents
        
        Args:
            documents: List of document chunks
        """
        try:
            self.documents = documents
            self.embeddings = self.create_embeddings(documents)
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            
            logger.info(f"Built vector index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            raise
    
    def search_similar_documents(
        self, 
        query: str, 
        k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Search for similar documents using vector similarity
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if k is None:
            k = config.MAX_CHUNKS
            
        try:
            if self.index is None or len(self.documents) == 0:
                logger.warning("No documents indexed")
                return []
            
            # Create query embedding
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            # Filter by similarity threshold
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= config.SIMILARITY_THRESHOLD:
                    results.append((self.documents[idx], float(score)))
            
            logger.info(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def save_index(self, filepath: str) -> None:
        """Save vector index and documents to file"""
        try:
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, filepath + '.faiss')
            
            logger.info(f"Saved index to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def load_index(self, filepath: str) -> None:
        """Load vector index and documents from file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Index file not found: {filepath}")
                return
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            
            # Load FAISS index
            faiss_path = filepath + '.faiss'
            if os.path.exists(faiss_path):
                self.index = faiss.read_index(faiss_path)
            
            logger.info(f"Loaded index from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")

# Global embedding model instance
embedding_model = EmbeddingModel()