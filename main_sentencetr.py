from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from indexer_sentencetr import indexer
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI()

class DocumentList(BaseModel):
    documents: List[str]

@app.post("/index-documents")
async def index_documents(documents: DocumentList):
    indexer.index(documents.documents)
    return {"message": "Documents indexed successfully."}

@app.get("/query-document")
async def query_document(query: str, n: int = 5):
    if indexer.embeddings is None:
        raise HTTPException(status_code=404, detail="No indexed documents found.")
    
    results = indexer.query_documents(query, n)
    
   
    results = [(doc, float(similarity)) for doc, similarity in results]
    

    json_results = [{"document": doc, "similarity": similarity} for doc, similarity in results]
    
    
    encoded_results = jsonable_encoder(json_results)
    
    return JSONResponse(content={"query": query, "results": encoded_results})

@app.get("/retrieve-document/{index}")
async def retrieve_document(index: int):
    document = indexer.retrieve_document(index)
    if document is not None:
        return document
    else:
        raise HTTPException(status_code=404, detail="Document not found.")
