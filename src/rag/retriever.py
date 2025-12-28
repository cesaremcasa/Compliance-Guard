import logging
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityException(Exception):
    pass

class IndexIntegrityChecker:
    @staticmethod
    def generate_hash(file_path: str) -> str:
        import hashlib
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def save_hash(file_path: str, hash_value: str) -> None:
        hash_file = file_path + ".sha256"
        with open(hash_file, "w") as f:
            f.write(hash_value)

    @staticmethod
    def verify(file_path: str) -> bool:
        hash_file = file_path + ".sha256"
        if not os.path.exists(hash_file):
            logger.warning("No hash file found. Cannot verify.")
            return False
        with open(hash_file, "r") as f:
            stored_hash = f.read().strip()
        current_hash = IndexIntegrityChecker.generate_hash(file_path)
        return current_hash == stored_hash

class ComplianceRetriever:
    def __init__(self, index_path: str = "data/processed/faiss_index.bin"):
        self.index_path = index_path
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.vector_store = None
        self._load_index_securely()

    def _load_index_securely(self):
        logger.info("Performing Security Integrity Check...")
        is_safe = IndexIntegrityChecker.verify(self.index_path)
        if not is_safe:
            raise SecurityException("Index integrity verification failed.")
        
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
        logger.info("✅ Vector Store loaded securely.")

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
