from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class DocumentIndexer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.document_matrix = None
        self.documents = []

    def index(self, new_documents):
        if not hasattr(self, 'documents') or self.documents is None:
            self.documents = new_documents
        else:
            
            self.documents.extend(new_documents)
        self.document_matrix = self.tfidf_vectorizer.fit_transform(self.documents)

    def query_documents(self, query, n=5, threshold: float = 0.2):
        query_vec = self.tfidf_vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vec, self.document_matrix).flatten()
        
       
        related_docs_indices = [i for i, sim in enumerate(cosine_similarities) if sim >= threshold]
        related_docs_indices = sorted(related_docs_indices, key=lambda i: cosine_similarities[i], reverse=True)[:n]

        return [(self.documents[i], cosine_similarities[i]) for i in related_docs_indices]
    def retrieve_document(self, index):
        if 0 <= index < len(self.documents):
            return self.documents[index]
        else:
            return None

# Global indexer instance
indexer = DocumentIndexer()