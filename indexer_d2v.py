from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from gensim.utils import simple_preprocess
class DocumentIndexer:
    def __init__(self, vector_size=300, window=5, min_count=1, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.documents = []
        self.doc2vec_model = None

    def index(self, new_documents):
        self.documents.extend(new_documents)
        tagged_documents = [TaggedDocument(simple_preprocess(doc), [i]) for i, doc in enumerate(new_documents)]
        if self.doc2vec_model is None:
            self.doc2vec_model = Doc2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count)
            self.doc2vec_model.build_vocab(tagged_documents)
        else:
            self.doc2vec_model.build_vocab(tagged_documents, update=True)
        self.doc2vec_model.train(tagged_documents, total_examples=len(tagged_documents), epochs=self.epochs)

    def query_documents(self, query, n=5, threshold: float = 0.001):
        query_vector = self.doc2vec_model.infer_vector(query.split())
        similarities = self.doc2vec_model.docvecs.most_similar([query_vector], topn=n)
        return [(self.documents[i], similarity) for i, similarity in similarities]
    def retrieve_document(self, index):
        if 0 <= index < len(self.documents):
            return self.documents[index]
        else:
            return None

# Global indexer instance
indexer = DocumentIndexer()
