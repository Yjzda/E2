from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from indexer_tfidffinetune_woseuil import indexer

app = FastAPI()

class DocumentList(BaseModel):
    documents: List[str]

@app.post("/index-documents")
async def index_documents(documents: DocumentList):
    indexer.index(documents.documents)
    param_grid = {
    'max_features': [1000, 2000, 3000],
    'ngram_range': [(1, 1), (1, 2), (2, 2)],
    'stop_words': [None, 'english'],
    'max_df': [0.5, 0.7, 1.0],  # Example values for max_df
    'min_df': [1, 2, 3]       
}
    indexer.fine_tune(param_grid)
    return {"message": "Documents indexed successfully."}

@app.get("/query-document")
async def query_document(query: str, n: int = 5):
    if indexer.document_matrix is None:
        raise HTTPException(status_code=404, detail="No indexed documents found.")
    results = indexer.query_documents(query, n)
    return {"query": query, "results": results}


@app.get("/retrieve-document/{index}")
async def retrieve_document(index: int):
    document = indexer.retrieve_document(index)
    if document is not None:
        return {"document": document}
    else:
        raise HTTPException(status_code=404, detail="Document not found.")