import openai
import os

class DocumentIndexer:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.documents = []

    def index(self, new_documents):
        self.documents.extend(new_documents)

    def query_documents(self, query, n=5):
        embeddings = []
        for document in self.documents:
            response = openai.Completion.create(
                engine="text-embedding-ada-002",
                prompt=document,
                max_tokens=100,
                logprobs=0
            )
            if response.choices and hasattr(response.choices[0], 'embedding'):
                embeddings.append(response.choices[0].embedding)
            else:
                print("Warning: No embedding found for document:", document)

        query_response = openai.Completion.create(
            engine="text-embedding-ada-002",
            prompt=query,
            max_tokens=100,
            return_prompt=False
        )
        if query_response.choices and hasattr(query_response.choices[0], 'embedding'):
            query_embedding = query_response.choices[0].embedding
        else:
            print("Error: No embedding found for query.")
        return []

        similarities = []
        for e in embeddings:
            # Calculer la similarité entre les embeddings
            similarity = e.cosine_similarity(query_embedding)
            similarities.append(similarity)

        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        return [(self.documents[i], similarities[i]) for i in sorted_indices[:n]]


    def retrieve_document(self, index):
        if 0 <= index < len(self.documents):
            return self.documents[index]
        else:
            return None

# Global indexer instance
indexer = DocumentIndexer()
