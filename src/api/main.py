import logging
import sys
import os
import json
import uuid
from datetime import datetime
from typing import Callable

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag.retriever import retriever_instance
from src.llm_client import LLMClient

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

# Configure root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

for handler in logger.handlers[:]:
    logger.removeHandler(handler)
    
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(JSONFormatter())
logger.addHandler(stream_handler)

# --- 2. Middleware for Request ID ---

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        req_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        request.state.request_id = req_id
        
        response = await call_next(request)
        
        process_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        response.headers["X-Request-ID"] = req_id
        
        logger.info(
            "Request processed", 
            extra={
                "props": {
                    "request_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(process_time, 2),
                    "user_ip": request.client.host if request.client else "unknown"
                }
            }
        )
        
        return response

app = FastAPI(
    title="Compliance-Guard API",
    description="API for retrieving compliance controls using RAG.",
    version="0.3.0"
)

app.add_middleware(LoggingMiddleware)

llm_client = LLMClient()

# --- Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query about compliance standards.", min_length=3)
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to return.")

class ControlResult(BaseModel):
    control_id: str
    source: str
    content: str

class QueryResponse(BaseModel):
    query: str
    results: list[ControlResult]

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Compliance-Guard API v3.0 (SRE Edition)"}

@app.post("/compliance")
def check_compliance(request: QueryRequest, http_request: Request):
    req_id = http_request.state.request_id
    
    try:
        logger.info(
            "Compliance check started",
            extra={"props": {"request_id": req_id, "query": request.query}}
        )

        # 1. RAG - CORREÇÃO USANDO k=k
        rag_results = retriever_instance.search(query=request.query, k=request.top_k)
        
        context_str = "\n".join([doc["content"] for doc in rag_results])

        # 2. Prompt
        system_instruction = (
            "[INST] Você é um oficial de compliance especialista. "
            "Analise o contexto fornecido e a entrada do usuário. "
            "Determine se a entrada está em conformidade. "
            "Responda APENAS em JSON válido: "
            '{ "is_compliant": true/false, "reason": "string" }. '
            "Não use markdown. [/INST]"
        )
        
        final_prompt = f"{system_instruction}\n\nContexto:\n{context_str}\n\nEntrada do Usuário:\n{request.query}\n\nResposta:"

        # 3. GPU
        try:
            llm_response_text = llm_client.generate(final_prompt)
            gpu_status = "success"
        except Exception as e:
            logger.error(
                "GPU connection failed",
                exc_info=True,
                extra={"props": {"request_id": req_id, "error": str(e)}}
            )
            llm_response_text = '{"is_compliant": false, "reason": "GPU Service Unavailable"}'
            gpu_status = "failed"

        # 4. Parse JSON
        try:
            clean_text = llm_response_text.replace("```json", "").replace("```", "").strip()
            result_json = json.loads(clean_text)
        except json.JSONDecodeError:
            logger.warning(
                "GPU failed to return valid JSON",
                extra={"props": {"request_id": req_id}}
            )
            result_json = {
                "is_compliant": False, 
                "reason": "Erro de processamento: A GPU não retornou um JSON válido."
            }

        logger.info(
            "Compliance check completed",
            extra={"props": {"request_id": req_id, "gpu_status": gpu_status}}
        )
        
        return {
            "status": "success",
            "request_id": req_id,
            "query": request.query,
            "rag_context_count": len(rag_results),
            "compliance": result_json,
            "gpu_used": True
        }

    except Exception as e:
        logger.error(
            "Critical API error",
            exc_info=True,
            extra={"props": {"request_id": req_id}}
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_controls(request: QueryRequest):
    req_id = str(uuid.uuid4())
    try:
        # CORREÇÃO USANDO k=k
        results = retriever_instance.search(query=request.query, k=request.top_k)
        formatted_results = [
            ControlResult(control_id=r['id'], source=r['source'], content=r['content'])
            for r in results
        ]
        logger.info("Legacy query executed", extra={"props": {"request_id": req_id}})
        return QueryResponse(query=request.query, results=formatted_results)
    except Exception as e:
        logger.error("Legacy query failed", exc_info=True, extra={"props": {"request_id": req_id}})
        raise HTTPException(status_code=500, detail=str(e))
