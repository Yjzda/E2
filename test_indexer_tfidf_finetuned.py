import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics.pairwise import cosine_similarity
from indexer_tfidffine_tune import DocumentIndexer

@pytest.fixture
def documents():
    return [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]

@pytest.fixture
def param_grid():
    return {
        'max_features': [1000, 2000, 3000],
        'ngram_range': [(1, 1), (1, 2), (2, 2)],
        'stop_words': [None, 'english'],
        'max_df': [0.5, 0.7, 1.0],  
        'min_df': [1, 2, 3]       
    }

@pytest.fixture
def indexer(documents, param_grid):
    indexer = DocumentIndexer()
    indexer.index(documents)
    indexer.fine_tune(param_grid)
    return indexer

def test_indexing(documents, indexer):
    assert indexer.documents == documents

def test_fine_tune(indexer):
    assert isinstance(indexer.tfidf_vectorizer, TfidfVectorizer)

def test_query_documents(indexer):
    query = "This is the first document."
    results = indexer.query_documents(query)
    assert len(results) > 0

def test_retrieve_document(indexer):
    document_index = 0
    assert indexer.retrieve_document(document_index) == "This is the first document."

if __name__ == "__main__":
    pytest.main()
