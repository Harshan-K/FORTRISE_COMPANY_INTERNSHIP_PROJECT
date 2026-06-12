import os
import pickle
import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from config import Config


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """Ensure 2D float32 array and L2-normalize in place."""
    embeddings = np.atleast_2d(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    return embeddings


class VectorStore:
    def __init__(self):
        Config.create_directories()
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        self.documents = []
        self.index_path = str(Config.FAISS_INDEX_PATH) + ".index"
        self.documents_path = Config.VECTORSTORE_DIR / "documents.pkl"
        self.load_index()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return np.atleast_2d(embeddings).astype('float32')

    def add_documents(self, chunks: List[Dict[str, Any]], source_file: str) -> int:
        texts = [chunk['content'] for chunk in chunks]
        embeddings = _normalize(self.generate_embeddings(texts))
        dimension = embeddings.shape[1]

        # Build or reset index based on dimension
        if self.index is None or self.index.d != dimension:
            if self.index is not None:
                print(f"Dimension mismatch (index={self.index.d}, model={dimension}). Resetting index.")
            self.index = faiss.IndexFlatIP(dimension)
            self.documents = []

        for i, chunk in enumerate(chunks):
            chunk.update({
                'source_file': source_file,
                'embedding_id': len(self.documents) + i
            })

        self.index.add(embeddings)
        self.documents.extend(chunks)
        self.save_index()
        return len(chunks)

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or len(self.documents) == 0:
            return []

        query_embedding = _normalize(self.generate_embeddings([query]))
        k = min(k, len(self.documents))
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(score)
                results.append(doc)
        return results

    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"Loaded {len(self.documents)} documents from existing index")
            except Exception as e:
                print(f"Error loading index, starting fresh: {e}")
                self.index = None
                self.documents = []

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.index.d if self.index else 0
        }

    def check_duplicate_question(self, question: str, threshold: float = 0.8) -> bool:
        if not self.documents:
            return False
        results = self.similarity_search(question, k=1)
        return bool(results and results[0]['similarity_score'] > threshold)
