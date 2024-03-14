from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from indexer import indexer

app = FastAPI()

class DocumentList(BaseModel):
    documents: List[str]

@app.post("/index-documents")
async def index_documents(documents: DocumentList):
    indexer.index(documents.documents)
    return {"message": "Documents indexed successfully."}

@app.get("/query-document")
async def query_document(query: str, n: int = 5,threshold:float=0.5):
    if indexer.document_matrix is None:
        raise HTTPException(status_code=404, detail="No indexed documents found.")
    results = indexer.query_documents(query, n,threshold)
    return {"query": query, "results": results}


@app.get("/retrieve-document/{index}")
async def retrieve_document(index: int):
    document = indexer.retrieve_document(index)
    if document is not None:
        return {"document": document}
    else:
        raise HTTPException(status_code=404, detail="Document not found.")