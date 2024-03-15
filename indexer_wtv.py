from gensim.models import Word2Vec
import numpy as np
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2.T)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity
class DocumentIndexer:
    def __init__(self):
        self.word2vec_model = Word2Vec(vector_size=300, window=5, min_count=1, workers=4)
        self.documents = []

    def index(self, new_documents):
        tokenized_documents = [document.split() for document in new_documents]
        self.word2vec_model.build_vocab(tokenized_documents)
        self.word2vec_model.train(tokenized_documents, total_examples=len(tokenized_documents), epochs=10)
        self.documents.extend(new_documents)

    def query_documents(self, query, n=5):
        # Split the query into individual words
        query_words = query.split()

        # Filter out words that are not present in the vocabulary
        query_words_in_vocab = [word for word in query_words if word in self.word2vec_model.wv]

        # Check if there are any words in the query that are present in the vocabulary
        if not query_words_in_vocab:
            # If none of the words in the query are present in the vocabulary, return an empty list
            return []

        # Compute the average word embedding vector for the query
        query_vector = np.mean([self.word2vec_model.wv[word] for word in query_words_in_vocab], axis=0)

        # Compute the average word embedding vectors for all documents
        document_vectors = np.array([np.mean([self.word2vec_model.wv[word] for word in doc.split() if word in self.word2vec_model.wv], axis=0)
                                    for doc in self.documents])

        # Calculate cosine similarities between the query vector and document vectors
        cosine_similarities = cosine_similarity(query_vector, document_vectors).flatten()

        # Get indices of documents sorted by similarity
        related_docs_indices = cosine_similarities.argsort()[:-n-1:-1]

        # Return the most similar documents along with their similarity scores
        return [(self.documents[i], cosine_similarities[i]) for i in related_docs_indices]
    def retrieve_document(self, index):
        if 0 <= index < len(self.documents):
            return self.documents[index]
        else:
            return None

# Global indexer instance
indexer_wtv = DocumentIndexer()
