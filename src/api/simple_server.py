import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import PeftModel
import torch
import os

app = FastAPI()

# Config
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
ADAPTER_PATH = "/home/ubuntu/compliance-guard-gpu/models/checkpoints"

print("Carregando Modelo e Adapter... (Isso pode levar 1-2 minutos)")

# 1. Carregar Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# 2. Carregar Base em 4-bit (Para caber na GPU)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

print("Carregando pesos do modelo base...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# 3. Carregar LoRA Adapter
print("Carregando LoRA Adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("✅ Modelo Carregado e pronto para servir!")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: PromptRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True
        )
    
    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove o prompt original da resposta (apenas o texto gerado)
    return {"result": result_text[len(request.prompt):]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
