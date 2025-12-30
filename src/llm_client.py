import os
import requests
import logging
import time

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.gpu_url = os.getenv("GPU_URL", "http://localhost:8000")
        self.endpoint = f"{self.gpu_url}/v1/completions"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.1,
        }
        
        try:
            start_time = time.time()
            
            # TIMEOUT AUMENTADO PARA 180 SEGUNDOS (3 MINUTOS)
            response = requests.post(self.endpoint, json=payload, timeout=180)
            
            response.raise_for_status()
            data = response.json()
            latency = (time.time() - start_time) * 1000
            
            logger.info(
                "GPU Request Success",
                extra={"props": {"latency_ms": round(latency, 2), "status_code": response.status_code}}
            )
            
            return data["choices"][0]["text"].strip()
            
        except Exception as e:
            logger.error(
                "GPU Inference Exception",
                exc_info=True,
                extra={"props": {"error": str(e), "gpu_url": self.gpu_url}}
            )
            raise Exception(f"GPU Inference Failed: {str(e)}")
