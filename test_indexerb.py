import pytest
from sklearn.exceptions import NotFittedError
from indexerb import DocumentIndexer

@pytest.fixture
def indexer():
    return DocumentIndexer()

def test_indexing(indexer):
    documents = ["This is the first document", "This is the second document", "Another document"]
    indexer.index(documents)
    assert indexer.documents == documents

def test_indexing_multiple_times(indexer):
    documents1 = ["This is the first document", "This is the second document"]
    documents2 = ["Another document"]
    expected_documents = documents1 + documents2
    indexer.index(documents1)
    indexer.index(documents2)
    assert indexer.documents == expected_documents

def test_query_documents(indexer):
    documents = ["This is the first document", "This is the second document", "Another document"]
    indexer.index(documents)
    results = indexer.query_documents("first document", n=2)
    assert len(results) == 2
    assert results[0][0] == "This is the first document"

def test_query_documents_threshold(indexer):
    documents = ["This is the first document", "This is the second document", "Another document"]
    indexer.index(documents)
    results = indexer.query_documents("first document", threshold=0.9)
    assert len(results) == 0

def test_query_documents_empty_indexer(indexer):
    with pytest.raises(NotFittedError):
        indexer.query_documents("query")

def test_retrieve_document(indexer):
    documents = ["This is the first document", "This is the second document", "Another document"]
    indexer.index(documents)
    assert indexer.retrieve_document(0) == "This is the first document"
    assert indexer.retrieve_document(3) == None

if __name__ == "__main__":
    pytest.main()
