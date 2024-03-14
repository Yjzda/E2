from gensim.models import Word2Vec
import numpy as np

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

        # Compute the average vector for the words in the query that are present in the vocabulary
        query_vector = np.mean([self.word2vec_model.wv[word] for word in query_words_in_vocab], axis=0)

        # Find the most similar words to the query vector
        similar_words = self.word2vec_model.wv.most_similar(positive=[query_vector], topn=n)

        # Return the most similar words along with their similarity scores
        return [(self.documents[int(index)], sim) for index, sim in similar_words]

    def retrieve_document(self, index):
        if 0 <= index < len(self.documents):
            return self.documents[index]
        else:
            return None

# Global indexer instance
indexer = DocumentIndexer()
