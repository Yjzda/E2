from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from indexer_wtv import indexer  # Mise à jour de l'import

app = FastAPI()

class DocumentList(BaseModel):
    documents: List[str]

@app.post("/index-documents")
async def index_documents(documents: DocumentList):
    indexer.index(documents.documents)
    return {"message": "Documents indexed successfully."}

@app.get("/query-document")
async def query_document(query: str, n: int = 5):
    if not indexer.word2vec_model.wv.key_to_index:  # Vérifier si le modèle est bien entraîné
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
