class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def get_relevant_docs(self, query, top_k=3):
        return self.vector_store.search(query, top_k=top_k)
