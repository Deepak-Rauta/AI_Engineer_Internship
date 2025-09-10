from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from config.config import Config

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name=None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
    
    def encode_text(self, text):
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(text, convert_to_numpy=True)
        return embeddings
    
    def encode_documents(self, documents):
        embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    
    def compute_similarity(self, query_embedding, doc_embeddings):
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)
        return similarities
    
    def find_most_similar(self, query, documents, doc_embeddings=None, top_k=5, threshold=None):
        query_embedding = self.encode_text(query)[0]
        
        if doc_embeddings is None:
            doc_embeddings = self.encode_documents(documents)
        
        similarities = self.compute_similarity(query_embedding, doc_embeddings)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        threshold = threshold or Config.SIMILARITY_THRESHOLD
        results = []
        
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                results.append((documents[idx], float(similarity), int(idx)))
        
        return results

_embedding_instance = None

def get_embedding_model():
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingModel()
    return _embedding_instance
