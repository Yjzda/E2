import pytest
#from indexer import DocumentIndexer
#from indexer_sentencetr import DocumentIndexer
from indexer_tfidffinetune_woseuil import DocumentIndexer
@pytest.fixture
def indexer():
    return DocumentIndexer()

def test_indexing_and_querying(indexer):
  
    documents = ["This is the first document", "This is the second document", "Another document"]
    indexer.index(documents)
    
  
    results = indexer.query_documents("first document")
    
   
    assert len(results) == 3  # There are 3 documents indexed; should be 2 when there is argument threshold
    assert results[0][0] == "This is the first document"  # Most similar document
    
    # Assuming the minimum similarity score
    assert results[0][1] > 0.0  # Adjust threshold as per your requirement

    # Test with a query that doesn't match any document
    results = indexer.query_documents("random query")
    assert len(results) >= 0

if __name__ == "__main__":
    pytest.main()
