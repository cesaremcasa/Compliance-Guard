import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.download_nist import download_official_nist
from src.rag.ingest import ComplianceSmartIngestor
from src.rag.retriever import IndexIntegrityChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting BLOCK 4: Production Ingestion Pipeline")
    
    # 1. Download Data
    logger.info("STEP 1: Acquiring Official NIST Data...")
    if not download_official_nist():
        logger.error("Failed to download PDF. Halting.")
        return False
        
    # 2. Ingestion
    logger.info("STEP 2: Running Ingestion (CPU Optimized)...")
    ingestor = ComplianceSmartIngestor("data/official_nist.pdf")
    try:
        ingestor.run_pipeline("data/processed/faiss_index.bin")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # 3. Generate Security Hash
    logger.info("STEP 3: Generating Integrity Hash...")
    try:
        index_path = "data/processed/faiss_index.bin"
        hash_val = IndexIntegrityChecker.generate_hash(index_path)
        IndexIntegrityChecker.save_hash(index_path, hash_val)
        logger.info(f"Hash generated: {hash_val[:16]}...")
    except Exception as e:
        logger.error(f"Hashing failed: {e}")
        return False
        
    logger.info("✅ Pipeline Completed Successfully.")
    logger.info("Please run the test query manually or start the API.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
