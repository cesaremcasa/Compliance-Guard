# Deployment Handover Notes

## Current Environment (T3 Logic)
- **Instance Type:** AWS T3 (CPU)
- **Role:** RAG Ingestion & Core API Testing
- **API Endpoint:** http://:8000
- **Data Location:** `data/processed/faiss_index.bin`

## Status
- [x] Block 1: RAG Ingestion Complete
- [x] Block 2: Core API (FastAPI) Running
- [x] Block 3: Basic Frontend Integrated

## Next Steps (Block 4: GPU Integration)
1. Launch **g6.xlarge** instance.
2. Update Security Group on the new GPU instance to allow inbound traffic on **Port 8000**.
3. Clone this repository to the GPU instance.
4. Update `requirements.txt` to include `torch` (CUDA version) and `accelerate`.
