import requests
from typing import List, Union
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class LocalEmbedding(Embeddings):
    """
    A wrapper for interacting with a locally hosted FastAPI embedding service.
    Works similarly to HuggingFaceEmbeddings, so it can be used in LangChain components.
    """

    def __init__(self, api_url: str = "http://localhost:8002/v1/embeddings"):
        self.api_url = api_url
        # print(f"🚀 Local Embedding API URL: {self.api_url}")

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single text query."""
        
        return self._send_request(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        # texts = [doc.page_content for doc in documents]
        return self._send_request(texts)

    def _send_request(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Handles API request and response parsing."""
        try:
            print(self.api_url)
            response = requests.post(self.api_url, json={"input": text}, proxies={"http": None, "https": None})
            
            if response.status_code != 200:
                raise Exception(f"API Error ({response.status_code}): {response.text}")

            response_json = response.json()
            if "data" not in response_json:
                raise Exception(f"API Error: Missing 'data' field in response: {response_json}")

            return response_json["data"]  # List of embeddings
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")


if __name__ == "__main__":
    local_embedding_model = LocalEmbedding()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # documents = text_splitter.create_documents(["This is a sample chunk.", "Another chunk of text for testing."])
    documents = ["This is a sample chunk.", "Another chunk of text for testing."]
    # char_documents = text_splitter.split_documents(data)
    print(documents)
    embedding = local_embedding_model.embed_documents(documents)
    # embedding = local_embedding_model.embed_query(["Tis is test", "This is test2"])
    print(f"Embedding dimension: {len(embedding[0])}")

