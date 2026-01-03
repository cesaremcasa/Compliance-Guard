# Base Image
FROM python:3.9-slim

# Set Working Directory
WORKDIR /app

# Install System Dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends     curl     && rm -rf /var/lib/apt/lists/*

# Install Python Dependencies
# Installing directly as requested to ensure clean build
RUN pip install --no-cache-dir torch transformers accelerate peft bitsandbytes fastapi uvicorn prometheus-client slowapi

# Copy Application Code
COPY . /app

# Expose API Port
EXPOSE 8000

# Start Command
# Note: Ensure your entrypoint matches your actual file (simple_server or simple_server_v2)
CMD ["uvicorn", "src.api.simple_server_v2:app", "--host", "0.0.0.0", "--port", "8000"]
