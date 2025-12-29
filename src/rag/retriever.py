import logging
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityException(Exception):
    pass

class ComplianceRetriever:
    def __init__(self, index_path: str = "data/processed/faiss_index.bin"):
        self.index_path = index_path
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.vector_store = None
        self._load_index_securely()
    
    def _load_index_securely(self):
        logger.info(f"Checking if index exists at {self.index_path}...")
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found at {self.index_path}")
        
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
        logger.info("✅ Vector Store loaded successfully.")
    
    def retrieve(self, query: str, top_k: int = 3, k: int = None) -> list[dict]:
        """Método principal chamado pela API - aceita tanto top_k quanto k"""
        # Se k não for especificado, usa top_k
        num_results = k if k is not None else top_k
        return self.search(query, num_results)
    
    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized.")
        
        logger.info(f"Searching for: '{query}'")
        results = self.vector_store.similarity_search(query, k=k)
        
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "control_id": doc.metadata.get('control_id', 'UNKNOWN'),
                "source": doc.metadata.get('source', 'unknown'),
                "content": doc.page_content,
            })
        
        return formatted_results

# Criar instância global do retriever
retriever_instance = ComplianceRetriever()
