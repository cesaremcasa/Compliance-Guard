import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NIST_URL = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf"
DESTINATION = "data/official_nist.pdf"
MIN_FILE_SIZE_BYTES = 1000 * 1000 # 1MB

def download_official_nist():
    logger.info(f"Initiating secure download from: {NIST_URL}")
    os.makedirs(os.path.dirname(DESTINATION), exist_ok=True)
    
    try:
        response = requests.get(NIST_URL, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"File size detected: {total_size / (1024*1024):.2f} MB")
        
        if total_size < MIN_FILE_SIZE_BYTES:
            raise ValueError(f"File too small ({total_size} bytes).")
            
        with open(DESTINATION, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"✅ Successfully downloaded official NIST data.")
        return True
        
    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to download. {e}")
        if os.path.exists(DESTINATION):
            os.remove(DESTINATION)
        return False
