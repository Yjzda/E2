import os
import requests
import sys

base_url = os.getenv('baseurl', "http://localhost:8000")

def index_folder_contents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())

    response = requests.post(f"{base_url}/index-documents", json={"documents": documents})
    if response.status_code == 200:
        print("Documents indexed successfully.")
    else:
        print("Failed to index documents:", response.text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index_folder.py <folder_path>")
        sys.exit(1)
    folder_path = sys.argv[1]
    index_folder_contents(folder_path)