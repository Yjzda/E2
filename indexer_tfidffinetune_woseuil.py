from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics.pairwise import cosine_similarity

# Define custom scoring function
def cosine_similarity_score(y_true, y_pred):
    return cosine_similarity(y_true.reshape(1, -1), y_pred.reshape(1, -1))[0][0]

# Make scorer from custom scoring function
cosine_similarity_scorer = make_scorer(cosine_similarity_score)

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

    def fine_tune(self, param_grid):
        grid_search = GridSearchCV(self.tfidf_vectorizer, param_grid, cv=3, scoring=cosine_similarity_scorer)
        grid_search.fit(self.documents)
        self.tfidf_vectorizer = grid_search.best_estimator_

    def query_documents(self, query, n=5):
        query_vec = self.tfidf_vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vec, self.document_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-n-1:-1]
        return [(self.documents[i], cosine_similarities[i]) for i in related_docs_indices]
    def retrieve_document(self, index):
        if 0 <= index < len(self.documents):
            return self.documents[index]
        else:
            return None

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Grid of hyperparameters for fine-tuning
param_grid = {
    'max_features': [1000, 2000, 3000],
    'ngram_range': [(1, 1), (1, 2), (2, 2)],
    'stop_words': [None, 'english'],
    'max_df': [0.5, 0.7, 1.0],  # Example values for max_df
    'min_df': [1, 2, 3]       
}

# Global indexer instance
indexer = DocumentIndexer()
indexer.index(documents)
indexer.fine_tune(param_grid)
