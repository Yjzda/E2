import pytest
from fastapi.testclient import TestClient
from main_d2v import app

client = TestClient(app)


documents = {
    "documents": [
        "The cat is blue. It's from Egypt.",
        "The hat is red."
    ]
}


def test_index_documents():
    response = client.post("/index-documents", json=documents)
    assert response.status_code == 200
    assert response.json() == {"message": "Documents indexed successfully."}


def test_query_hat():
    query = "How is the hat."
    response = client.get(f"/query-document?query={query}&n=5")
    assert response.status_code == 200
  
    assert any("hat" in result[0].lower() for result in response.json()["results"])

def test_query_where_cat_from():
    query = "where does the cat from"
    response = client.get(f"/query-document?query={query}&n=5")
    assert response.status_code == 200

    assert any("cat" in result[0].lower() for result in response.json()["results"])


def test_retrieve_document():
    response = client.get("/retrieve-document/0")
    assert response.status_code == 200
    assert response.json()["document"] == "The cat is blue. It's from Egypt."

