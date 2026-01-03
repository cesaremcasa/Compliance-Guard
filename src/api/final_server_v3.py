import os
import time
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response, JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel

# --- Configuration ---
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
LORA_ADAPTER_PATH = "models/lora_adapter"
FORBIDDEN_KEYWORDS = ["ignore", "jailbreak", "system", "instructions"]
MAX_INPUT_LENGTH = 500
RATE_LIMIT = "10/minute"

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComplianceGuardv3_Final")

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter('compliance_request_total', 'Total requests', ['method', 'endpoint', 'status'])
LATENCY_HISTOGRAM = Histogram('compliance_latency_seconds', 'LLM Inference Latency')

# --- Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Models ---
class GenerateRequest(BaseModel):
    text: str

# --- Model Loading (Optimized with 4-bit) ---
logger.info("Loading Base Model with 4-bit Quantization...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Optimized Config for G4DN Memory Constraints
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # Load with 4-bit config to reduce VRAM usage from ~15GB to ~5GB
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    if os.path.exists(LORA_ADAPTER_PATH):
        logger.info(f"Loading LoRA Adapter from {LORA_ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        # Merging adapter is optional for inference but often helps performance
        # Keeping separate to allow unloading if needed, but here we load into base
    else:
        logger.warning("LoRA path not found. Using base 4-bit model.")
        model = base_model
        
    model.eval()
    logger.info("Model Loaded Successfully on Device: %s", device)

except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None 

# --- Endpoints ---
@app.get("/")
async def root():
    return {"status": "ok", "model": "Mistral-7B-4bit", "version": "v3.0"}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate")
@limiter.limit(RATE_LIMIT)
async def generate_text(request: Request, body: GenerateRequest):
    input_text = body.text
    
    # Validation
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

    try:
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
        LATENCY_HISTOGRAM.observe(duration)
        REQUEST_COUNT.labels(method='POST', endpoint='/generate', status='200').inc()
        
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"Inference error: {e}")
        REQUEST_COUNT.labels(method='POST', endpoint='/generate', status='500').inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
