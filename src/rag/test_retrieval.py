import logging
import sys
import os

from langchain_community.vectorstores import FAISS
# CORREÇÃO: Usando a nova biblioteca
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_retrieval(index_path: str = "data/processed/faiss_index.bin", model_name: str = "BAAI/bge-small-en-v1.5"):
    try:
        logger.info("--- Verification Started ---")
        embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
        
        if not os.path.exists(index_path):
             raise FileNotFoundError("Index not found. Run ingest.py first.")

        vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        query = "How do I manage user accounts?"
        results = vector_store.similarity_search(query, k=3)

        logger.info(f"Query: {query}")
        for i, doc in enumerate(results, 1):
            print(f"\nResult #{i}:")
            print(f"Control ID: {doc.metadata.get('control_id')}")
            print(f"Content: {doc.page_content[:100]}...")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_retrieval()
