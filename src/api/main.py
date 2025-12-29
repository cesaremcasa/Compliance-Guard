from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import sys
import os
import json
import requests

# Add parent directory to path to import rag modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag.retriever import retriever_instance
from src.llm_client import LLMClient  # <--- IMPORTAÇÃO DO CLIENTE GPU

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Compliance-Guard API",
    description="API for retrieving compliance controls using RAG.",
    version="0.2.0"
)

# --- Request/Response Models ---
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

# Inicializa o Cliente GPU
llm_client = LLMClient()

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Compliance-Guard API v2.0 Running with GPU Integration"}

@app.post("/compliance", response_model=dict)
def check_compliance(request: QueryRequest):
    """
    Endpoint Principal: Recupera contexto via RAG, envia para GPU e retorna verificação.
    """
    try:
        # 1. RAG - Recuperar Documentos
        logger.info(f"Received query: {request.query}")
        rag_results = retriever_instance.retrieve(query=request.query, top_k=request.top_k)
        
        context_str = "\n".join([doc["content"] for doc in rag_results])

        # 2. Construir Prompt para Mistral
        system_instruction = (
            "[INST] Você é um oficial de compliance especialista. "
            "Analise o contexto fornecido e a entrada do usuário. "
            "Determine se a entrada está em conformidade. "
            "Responda APENAS em JSON válido: "
            '{ "is_compliant": true/false, "reason": "string" }. '
            "Não use markdown. [/INST]"
        )
        
        final_prompt = f"{system_instruction}\n\nContexto:\n{context_str}\n\nEntrada do Usuário:\n{request.query}\n\nResposta:"

        # 3. Chamada GPU (via LLMClient)
        llm_response_text = llm_client.generate(final_prompt)
        
        logger.info(f"GPU Response: {llm_response_text}")

        # 4. Parse do JSON (Tratamento simples de erro)
        try:
            # Limpa markdown se existir
            clean_text = llm_response_text.replace("```json", "").replace("```", "").strip()
            result_json = json.loads(clean_text)
        except json.JSONDecodeError:
            # Fallback se o modelo falhar no JSON
            logger.error("GPU failed to return valid JSON")
            result_json = {
                "is_compliant": False, 
                "reason": "Erro de processamento: A GPU não retornou um JSON válido."
            }

        # 5. Retornar Resultado Combinado
        return {
            "status": "success",
            "query": request.query,
            "rag_context_count": len(rag_results),
            "compliance": result_json,
            "gpu_used": True
        }

    except Exception as e:
        logger.error(f"Error in compliance endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_controls(request: QueryRequest):
    """
    Endpoint Legado Apenas para RAG (sem GPU).
    """
    try:
        results = retriever_instance.retrieve(query=request.query, top_k=request.top_k)
        formatted_results = [
            ControlResult(control_id=r['id'], source=r['source'], content=r['content'])
            for r in results
        ]
        return QueryResponse(query=request.query, results=formatted_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
