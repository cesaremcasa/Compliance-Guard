import os
import requests

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
            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Erro ao conectar GPU: {e}")
            raise Exception("GPU Inference Failed")
