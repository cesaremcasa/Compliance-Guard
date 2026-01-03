import os
import time
import logging
import hashlib
import json
from datetime import datetime
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response, JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel

BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
LORA_ADAPTER_PATH = "models/checkpoints"
FORBIDDEN_KEYWORDS = ["ignore", "jailbreak", "system", "instructions"]
MAX_INPUT_LENGTH = 500
RATE_LIMIT = "10/minute"
CACHE_DIR = "/tmp/compliance_cache"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComplianceGuardv3_Final")

REQUEST_COUNT = Counter('compliance_request_total', 'Total requests', ['method', 'endpoint', 'status'])
LATENCY_HISTOGRAM = Histogram('compliance_latency_seconds', 'LLM Inference Latency')
CACHE_HIT_COUNTER = Counter('cache_hit_total', 'Cache hits')
CACHE_MISS_COUNTER = Counter('cache_miss_total', 'Cache misses')
FEEDBACK_COUNT = Counter('user_feedback_total', 'User feedback', ['rating'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Active requests')

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class GenerateRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    request_id: str
    rating: int
    comment: Optional[str] = None

class SimpleCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[dict]:
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                CACHE_HIT_COUNTER.inc()
                logger.info(f"Cache HIT: {cache_key[:8]}")
                return json.load(f)
        CACHE_MISS_COUNTER.inc()
        return None
    
    def set(self, text: str, response: dict):
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, 'w') as f:
            json.dump(response, f)

cache = SimpleCache(CACHE_DIR)

class FeedbackStore:
    def __init__(self, feedback_dir: str = "/tmp/feedback"):
        self.feedback_dir = feedback_dir
        os.makedirs(feedback_dir, exist_ok=True)
        self.feedback_file = os.path.join(feedback_dir, "feedback.jsonl")
    
    def save_feedback(self, feedback: dict):
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback) + '\n')
        logger.info(f"Feedback saved: {feedback['request_id']}")

feedback_store = FeedbackStore()

logger.info("Loading Base Model with 4-bit Quantization...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    if os.path.exists(LORA_ADAPTER_PATH):
        logger.info(f"Loading LoRA Adapter from {LORA_ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    else:
        logger.warning("LoRA path not found. Using base 4-bit model.")
        model = base_model
        
    model.eval()
    logger.info("Model Loaded Successfully on Device: %s", device)

except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None 

@app.get("/")
async def root():
    return {"status": "ok", "model": "Mistral-7B-4bit", "version": "v3.1-enhanced"}

@app.get("/health")
async def health():
    return {"status": "healthy" if model is not None else "unhealthy", "model_loaded": model is not None}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate")
@limiter.limit(RATE_LIMIT)
async def generate_text(request: Request, body: GenerateRequest):
    input_text = body.text
    request_id = hashlib.md5(f"{input_text}{time.time()}".encode()).hexdigest()[:16]
    
    ACTIVE_REQUESTS.inc()
    
    try:
        if len(input_text) > MAX_INPUT_LENGTH:
            REQUEST_COUNT.labels(method='POST', endpoint='/generate', status='400').inc()
            raise HTTPException(status_code=400, detail=f"Input exceeds {MAX_INPUT_LENGTH} chars")
        
        lower_input = input_text.lower()
        found_forbidden = [kw for kw in FORBIDDEN_KEYWORDS if kw in lower_input]
        if found_forbidden:
            REQUEST_COUNT.labels(method='POST', endpoint='/generate', status='400').inc()
            raise HTTPException(status_code=400, detail=f"Forbidden keywords: {found_forbidden}")

        if model is None:
            return JSONResponse(status_code=503, content={"detail": "Model not loaded"})

        cached_response = cache.get(input_text)
        if cached_response:
            return {
                "request_id": request_id,
                "generated_text": cached_response["generated_text"],
                "cached": True,
                "latency_seconds": 0.0
            }

        start_time = time.time()
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        duration = time.time() - start_time
        
        response_data = {"generated_text": generated_text}
        cache.set(input_text, response_data)
        
        LATENCY_HISTOGRAM.observe(duration)
        REQUEST_COUNT.labels(method='POST', endpoint='/generate', status='200').inc()
        
        return {
            "request_id": request_id,
            "generated_text": generated_text,
            "cached": False,
            "latency_seconds": round(duration, 4)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        REQUEST_COUNT.labels(method='POST', endpoint='/generate', status='500').inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        ACTIVE_REQUESTS.dec()

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    if feedback.rating < 1 or feedback.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    feedback_data = {
        "request_id": feedback.request_id,
        "rating": feedback.rating,
        "comment": feedback.comment,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    feedback_store.save_feedback(feedback_data)
    FEEDBACK_COUNT.labels(rating=str(feedback.rating)).inc()
    
    return {"status": "success", "message": "Feedback received. Thank you!"}

@app.get("/stats")
async def get_stats():
    cache_files = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
    return {
        "cache_entries": cache_files,
        "model_loaded": model is not None,
        "device": device if model else None
    }
