from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentIndexer:
    def __init__(self, model_name="sentence-transformers/distilbert-base-nli-mean-tokens"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def index(self, documents):
        self.documents = documents
        self.embeddings = self.model.encode(documents)

    
    def query_documents(self, query, n=5, threshold: float = 0.2):
        query_embedding = self.model.encode([query])[0]
        similarity_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
       
        relevant_indices = [i for i, score in enumerate(similarity_scores) if score >= threshold]
        sorted_indices = sorted(relevant_indices, key=lambda i: similarity_scores[i], reverse=True)[:n]

        results = [(self.documents[i], similarity_scores[i]) for i in sorted_indices]
        return results
    def retrieve_document(self, index):
        if 0 <= index < len(self.documents):
            return {"document": self.documents[index], "index": index}
        else:
            return {"error": "Document not found."}

# Global indexer instance
indexer = DocumentIndexer()
