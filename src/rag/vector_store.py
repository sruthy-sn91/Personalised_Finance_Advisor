import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from config.config import config

class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.store_path = config.VECTOR_STORE_PATH

    def load_or_create(self):
        if os.path.exists(os.path.join(self.store_path, "faiss_index.bin")):
            self._load()
        else:
            self.index = None
            self.documents = []

    def add_document(self, text, doc_id):
        embeddings = self.model.encode([text])
        if self.index is None:
            # dimension
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.documents.append((doc_id, text))

    def search(self, query, top_k=3):
        if self.index is None:
            return []
        query_embedding = self.model.encode([query])
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx_list in indices:
            for idx in idx_list:
                doc_id, doc_text = self.documents[idx]
                results.append(doc_text)
        return results

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.store_path, "faiss_index.bin"))
            with open(os.path.join(self.store_path, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)

    def _load(self):
        self.index = faiss.read_index(os.path.join(self.store_path, "faiss_index.bin"))
        with open(os.path.join(self.store_path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
