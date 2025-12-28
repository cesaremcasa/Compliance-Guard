import logging
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# CORREÇÃO: Document mudou para langchain_core.documents
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceRetriever:
    def __init__(self, index_path: str = "data/processed/faiss_index.bin"):
        self.index_path = index_path
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.vector_store = None
        self._load_index()

    def _load_index(self):
        """Loads the FAISS index into memory."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found at {self.index_path}. Run ingest.py first.")
        
        logger.info("Initializing Embeddings Model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info(f"Loading Vector Store from {self.index_path}...")
        self.vector_store = FAISS.load_local(
            self.index_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        logger.info("Vector Store loaded successfully.")

    def search(self, query: str, k: int = 3) -> list[dict]:
        """
        Performs a similarity search and formats the output.
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized.")

        logger.info(f"Searching for: '{query}' with top_k={k}")
        
        # Execute search
        results = self.vector_store.similarity_search(query, k=k)
        
        # Format results for JSON response
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "control_id": doc.metadata.get('control_id', 'UNKNOWN'),
                "source": doc.metadata.get('source', 'unknown'),
                "content": doc.page_content,
                "score": None
            })
            
        return formatted_results

# Singleton instance to be loaded at startup
retriever_instance = ComplianceRetriever()
