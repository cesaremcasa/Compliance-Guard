FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch transformers accelerate peft bitsandbytes fastapi uvicorn prometheus-client slowapi
COPY . /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api.final_server_v3:app", "--host", "0.0.0.0", "--port", "8000"]
