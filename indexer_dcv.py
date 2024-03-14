from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np

class DocumentIndexer:
    def __init__(self):
        self.doc2vec_model = Doc2Vec(vector_size=300, window=5, min_count=1, workers=4)
        self.documents = []

    def index(self, new_documents):
        tagged_documents = [TaggedDocument(words=document.split(), tags=[str(i)]) for i, document in enumerate(new_documents)]
        self.doc2vec_model.build_vocab(tagged_documents)
        self.doc2vec_model.train(tagged_documents, total_examples=self.doc2vec_model.corpus_count, epochs=10)
        self.documents.extend(new_documents)

    # def query_documents(self, query, n=5, threshold: float = 0.0):
    #     query_vector = self.doc2vec_model.infer_vector(query.split())
    #     similarities = self.doc2vec_model.docvecs.most_similar([query_vector], topn=n)
    #     related_docs_indices = [int(index) for index, _ in similarities if _ >= threshold]
    #     return [(self.documents[i], _) for i, _ in related_docs_indices]
    def query_documents(self, query, n=5, threshold: float = 0.1):
        query_vector = self.doc2vec_model.infer_vector(query.split())
        similarities = self.doc2vec_model.docvecs.most_similar([query_vector], topn=n)
        # cosine_similarities = linear_kernel(query_vec, self.document_matrix).flatten()
        similarities_array = np.array(similarities)
        
        # Use argsort() method on the NumPy array
        related_docs_indices = similarities_array[:, 0].astype(int)
        # related_docs_indices = similarities.argsort()[:-n-1:-1]
        return [(self.documents[i], similarities[i]) for i in related_docs_indices]

    def retrieve_document(self, index):
        if 0 <= index < len(self.documents):
            return self.documents[index]
        else:
            return None

# Global indexer instance
indexer = DocumentIndexer()
