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
       
        query_words = query.split()
 
        query_words_in_vocab = [word for word in query_words if word in self.word2vec_model.wv]

        
        if not query_words_in_vocab:
            # If none of the words in the query are present in the vocabulary, return an empty list
            return []

        
        query_vector = np.mean([self.word2vec_model.wv[word] for word in query_words_in_vocab], axis=0)

       
        document_vectors = np.array([np.mean([self.word2vec_model.wv[word] for word in doc.split() if word in self.word2vec_model.wv], axis=0)
                                    for doc in self.documents])

        
        cosine_similarities = cosine_similarity(query_vector, document_vectors).flatten()

        
        related_docs_indices = cosine_similarities.argsort()[:-n-1:-1]

       
        return [(self.documents[i], cosine_similarities[i]) for i in related_docs_indices]
    def retrieve_document(self, index):
        if 0 <= index < len(self.documents):
            return self.documents[index]
        else:
            return None


indexer_wtv = DocumentIndexer()
