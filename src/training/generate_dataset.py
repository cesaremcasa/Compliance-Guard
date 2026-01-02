import json
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def extract_topic(chunk_text):
    """Extrai as primeiras 5 palavras como tópico."""
    clean_text = " ".join(chunk_text.split())
    first_sentence = clean_text.split('.')[0]
    words = first_sentence.split()
    return " ".join(words[:5])

def generate_dataset(output_path="dataset/train.jsonl", num_samples=50):
    print(f"--- Gerando {num_samples} amostras de treinamento ---")
    
    # 1. Carregar o Índice Real (RAG)
    print("Carregando índice FAISS real...")
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    vectorstore = FAISS.load_local("data/processed/faiss_index.bin", embeddings, allow_dangerous_deserialization=True)
    
    samples = []
    
    # Consultas genéricas para NIST
    queries = [
        "access control", "incident response", "encryption", "risk management",
        "physical security", "audit logs", "malware", "network security"
    ]
    
    import itertools
    all_queries = itertools.cycle(queries)
    
    for _ in range(num_samples):
        query = next(all_queries)
        
        # Busca no RAG (Dados Reais)
        results = vectorstore.similarity_search(query, k=1)
        
        if not results:
            continue
            
        context = results[0].page_content
        topic = extract_topic(context)
        
        # Formato Instrucao-Resposta
        prompt = f"Question: What are the NIST guidelines regarding {topic}?\n\nContext: {context}\n\nAnswer:"
        completion = context
        
        entry = {
            "prompt": prompt,
            "completion": completion
        }
        samples.append(entry)
        
    # Salvar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for entry in samples:
            f.write(json.dumps(entry) + '\n')
            
    print(f"✅ Sucesso! {len(samples)} amostras salvas em {output_path}")

if __name__ == "__main__":
    generate_dataset()
