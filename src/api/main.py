import logging
import sys
import os
import json
import uuid
from datetime import datetime
from typing import Callable
import requests

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag.retriever import retriever_instance
# Nota: Não usamos mais LLMClient local, chamamos o vLLM direto via requests

# --- 1. Structured JSON Logging Configuration ---

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service_name": "compliance-guard",
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, 'props'):
            log_record.update(record.props)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)

# Configure Logger
logger = logging.getLogger("compliance-guard")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(JSONFormatter())
logger.addHandler(ch)

# --- 2. Configuration & Initialization ---

# CONFIGURAÇÃO ATUALIZADA: Aponta para o modelo Fine-Tuned
MODEL_NAME = "compliance-trained" 
VLLM_API_URL = "http://localhost:8000/v1/completions"

app = FastAPI(title="Compliance Guard API", version="3.0")

# --- 3. Middleware ---

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        
        # Log request
        logger.info("Incoming request", extra={
            "props": {"request_id": request_id, "path": request.url.path, "method": request.method}
        })

        response = await call_next(request)
        
        # Add ID to response
        response.headers["X-Request-ID"] = request_id
        return response

app.add_middleware(RequestContextMiddleware)

# --- 4. Pydantic Models ---

class ComplianceRequest(BaseModel):
    query: str = Field(..., description="The compliance question to ask")

class ComplianceResponse(BaseModel):
    request_id: str
    answer: str
    source: str

# --- 5. Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/compliance", response_model=ComplianceResponse)
async def get_compliance_guidance(request: Request, body: ComplianceRequest):
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    try:
        # 1. RAG: Retrieve Context
        rag_docs = retriever_instance.retrieve(body.query, top_k=2)
        context_text = "\n".join([doc['text'] for doc in rag_docs])
        
        # 2. LLM: Construct Prompt
        prompt = f"Context: {context_text}\n\nQuestion: {body.query}\n\nAnswer:"
        
        # 3. LLM: Call vLLM (Fine-Tuned Model)
        payload = {
            "model": MODEL_NAME, # Usa o adapter "compliance-trained"
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.1
        }
        
        # Chama o vLLM
        llm_response = requests.post(VLLM_API_URL, json=payload, timeout=30)
        llm_response.raise_for_status()
        llm_data = llm_response.json()
        
        answer_text = llm_data['choices'][0]['text'].strip()
        
        logger.info("Request processed successfully", extra={"props": {"request_id": request_id}})
        
        return ComplianceResponse(
            request_id=request_id,
            answer=answer_text,
            source="RAG + Fine-Tuned vLLM"
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"vLLM connection error: {e}", extra={"props": {"request_id": request_id}})
        raise HTTPException(status_code=503, detail="LLM Service Unavailable")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", extra={"props": {"request_id": request_id}})
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
