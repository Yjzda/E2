import pytest
from sentence_transformers import SentenceTransformer
from indexer_sentencetr import DocumentIndexer

@pytest.fixture
def documents():
    return [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]

@pytest.fixture
def indexer():
    return DocumentIndexer()

def test_indexing(indexer, documents):
    indexer.index(documents)
    assert len(indexer.embeddings) == len(documents)

def test_query_documents(indexer, documents):
    indexer.index(documents)
    query = "This is the first document."
    results = indexer.query_documents(query)
    assert len(results) > 0

def test_retrieve_document(indexer, documents):
    indexer.index(documents)
    document_index = 0
    result = indexer.retrieve_document(document_index)
    assert result["document"] == documents[document_index]

if __name__ == "__main__":
    pytest.main()
