from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import uvicorn

app = FastAPI()

# Configurações - USANDO O MODELO QUE VOCÊ JÁ TEM
BASE_MODEL = "mistralai/Mistral-7B-v0.1"  # Modelo base (não Instruct)
LORA_PATH = "/home/ubuntu/compliance-guard-gpu/models/checkpoints"

print("=" * 60)
print("🚀 Iniciando servidor de API com modelo treinado")
print("=" * 60)

# Carregar modelo e tokenizer
print("📥 Carregando modelo base (já baixado)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,  # Corrigido: usar 'dtype' em vez de 'torch_dtype'
    device_map="auto",
    low_cpu_mem_usage=True  # Otimização de memória
)

print("🔧 Carregando adaptador LoRA treinado...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print("✅ Modelo carregado com sucesso!")
print("=" * 60)

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9

class CompletionResponse(BaseModel):
    model: str
    choices: list

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "model": "compliance-trained",
        "base_model": BASE_MODEL,
        "lora_path": LORA_PATH
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "compliance-trained",
                "object": "model",
                "owned_by": "organization",
                "created": 1704240000
            }
        ]
    }

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    try:
        print(f"📝 Recebendo prompt: {request.prompt[:100]}...")
        
        # Tokenizar input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        # Gerar resposta
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decodificar
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remover o prompt da resposta
        response_text = generated_text[len(request.prompt):].strip()
        
        print(f"✅ Resposta gerada: {response_text[:100]}...")
        
        return {
            "id": f"cmpl-{hash(request.prompt) % 10000}",
            "object": "text_completion",
            "created": 1704240000,
            "model": request.model,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(outputs[0]) - len(inputs.input_ids[0]),
                "total_tokens": len(outputs[0])
            }
        }
    
    except Exception as e:
        print(f"❌ Erro ao gerar resposta: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🌐 Servidor rodando em http://0.0.0.0:8000")
    print("📚 Documentação disponível em http://0.0.0.0:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
