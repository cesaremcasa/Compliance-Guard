import hashlib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndexIntegrityChecker:
    
    @staticmethod
    def _get_real_file_path(target_path: str) -> str:
        """
        FAISS save_local creates a directory. We need to hash the specific 
        index file inside it.
        """
        if os.path.isdir(target_path):
            # It's a directory, target the main FAISS index file
            return os.path.join(target_path, "index.faiss")
        return target_path

    @staticmethod
    def generate_hash(file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        
        # Resolve real file path
        real_path = IndexIntegrityChecker._get_real_file_path(file_path)
        
        if not os.path.exists(real_path):
            raise FileNotFoundError(f"Cannot find index file at {real_path}")

        with open(real_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def save_hash(file_path: str, hash_value: str) -> None:
        # We save the .sha256 file alongside the directory, not inside it
        # to avoid confusing the FAISS loader
        hash_file = file_path + ".sha256"
        
        with open(hash_file, "w") as f:
            f.write(hash_value)
        logger.info(f"Saved integrity hash to {hash_file}")

    @staticmethod
    def verify(file_path: str) -> bool:
        hash_file = file_path + ".sha256"
        
        if not os.path.exists(hash_file):
            logger.warning("No hash file found. Cannot verify integrity.")
            return False
            
        with open(hash_file, "r") as f:
            stored_hash = f.read().strip()
            
        current_hash = IndexIntegrityChecker.generate_hash(file_path)
        
        if current_hash == stored_hash:
            logger.info("✅ Index Integrity Verified: Hash Match.")
            return True
        else:
            logger.critical(f"❌ SECURITY ALERT: Hash Mismatch!")
            logger.critical(f"Stored: {stored_hash} | Current: {current_hash}")
            return False
