import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, store_path=None):
        from config.config import Config
        
        self.store_path = store_path or Config.VECTOR_STORE_PATH
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        os.makedirs(self.store_path, exist_ok=True)
        self._load_data()
    
    def add_documents(self, documents, embeddings, metadata=None):
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings count mismatch")
        
        if metadata is None:
            metadata = [{'id': i, 'source': 'unknown'} for i in range(len(documents))]
        
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self._save_data()
    
    def search(self, query_embedding, top_k=5, threshold=None):
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        from config.config import Config
        
        similarities = self._calc_similarity(query_embedding, self.embeddings)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
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
        
        return results
    
    def _calc_similarity(self, query_embedding, doc_embeddings):
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)
        return similarities
    
    def _save_data(self):
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        
        store_file = os.path.join(self.store_path, 'data.npz')
        np.savez(store_file, 
                documents=np.array(self.documents, dtype=object),
                embeddings=self.embeddings if self.embeddings is not None else np.array([]),
                metadata=np.array(self.metadata, dtype=object))
    
    def _load_data(self):
        store_file = os.path.join(self.store_path, 'data.npz')
        
        if os.path.exists(store_file):
            try:
                data = np.load(store_file, allow_pickle=True)
                self.documents = data['documents'].tolist() if 'documents' in data else []
                self.embeddings = data['embeddings'] if 'embeddings' in data and data['embeddings'].size > 0 else None
                self.metadata = data['metadata'].tolist() if 'metadata' in data else []
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}")
                self.documents = []
                self.embeddings = None
                self.metadata = []
        else:
            self.documents = []
            self.embeddings = None
            self.metadata = []
    
    def clear_store(self):
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        store_file = os.path.join(self.store_path, 'data.npz')
        if os.path.exists(store_file):
            os.remove(store_file)
    
    def get_stats(self):
        return {
            'num_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'store_path': self.store_path
        }

_vector_store_instance = None

def get_vector_store():
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
